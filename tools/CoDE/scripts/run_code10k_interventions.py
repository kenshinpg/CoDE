"""Frozen-test citation interventions; reuse legacy geometry, selectors and estimators."""
import argparse,json,os,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prepare_code10k import sha,dump,rows,unique
from run_code10k_sources import structural_record
from run_code10k_rq1 import verify,RetrievalSources,read,emit
from run_code10k_rq2 import prepare,SOURCES
from run_coverage_controls import coverage
from code_eval.tree_subset import Encoding,checked_contributions
from code_eval.fixed_diagnosis_g0 import geometry,select


def legal_variants(full,groups,cited):
    return {op:geometry(full,groups,cited,op) for op in ['add','delete']}


def worker(a):
    import torch
    from code_eval.frozen_bank import FrozenBank
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fm,fit,dictionary=verify(a);labels=fm['labels'];enc=Encoding(dictionary);cfg=read(a.intervention_config)
    cls=read(a.rq1/'class_shap.json')
    if sha(a.rq1/'class_shap.json')!=read(a.config)['class_shap_sha256'] or cls['labels']!=labels or cls['columns']!=enc.columns:raise ValueError('Class attribution drift')
    means=np.asarray(cls['mean_absolute'])
    paths={n:a.models/p for n,p in {'lightgbm':'source/tree/mask_augmented.txt','encoder':'encoder/encoder.pt','grasp':'formal/grasp.pt','cohd':'formal/cohd_counts.json'}.items()}
    bank=FrozenBank(paths,enc,labels,a.threads);retrieval=RetrievalSources(a,fit,labels,dictionary)
    split=unique(a.protocol/'patient_split_PRIVATE.jsonl','patient_id');patients=unique(a.sample/'data/patients_public.jsonl','patient_id')
    if sha(a.out/'explanation_ledger.jsonl')!=read(a.out/'protocol_lock.json')['explanation_ledger_sha256']:raise ValueError('Prepared ledger drift')
    by=defaultdict(list)
    for e in rows(a.out/'explanation_ledger.jsonl'):by[e['patient_id']].append(e)
    chosen=sorted(by)[a.shard::a.shards];folder=a.out/f'shard_{a.shard}';folder.mkdir();state={};started=time.time()
    outputs={n:(folder/(n+'.jsonl')).open('w',encoding='utf-8') for n in ['attempts','interventions','variant_scores','eligibility','precision_pairs']}
    try:
        for i,pid in enumerate(chosen):
            r=structural_record(patients[pid],split[pid]);tokens={f['fact_id']:f['provenance']['source_token'] for f in patients[pid]['facts'] if f.get('visible',True) and f['coverage_scope']=='symptoms_antecedents'}
            full=set(r['tokens']);groups=enc.groups(r['tokens']);minus={g:tuple(sorted(full-set(ts))) for g,ts in groups.items()}
            tasks=[];keys={tuple(sorted(full))};keys.update(minus.values())
            for e in by[pid]:
                reason='mapping_unavailable' if not e['reviewed_available'] else ('target_unsupported' if not e['target_supported'] else None)
                emit(outputs['eligibility'],dict(explanation_id=e['explanation_id'],patient_id=pid,reason=reason,included=reason is None,diagnosis_correct=e['diagnosis_correct']))
                if reason:continue
                if not set(e['cited_fact_ids'])<=tokens.keys():raise ValueError('Nonpatient citation')
                E={tokens[f] for f in e['cited_fact_ids']};geo=legal_variants(full,groups,E)
                variants={'original':sorted(E)};variants.update({op+'|'+g:v['fact_ids'] for op,gs in geo.items() for g,v in gs.items()})
                keys.update(tuple(fs) for fs in variants.values());tasks.append((e,E,geo,variants))
            if not tasks:continue
            keys=sorted(keys);probs={n:[] for n in SOURCES}
            for start in range(0,len(keys),64):
                batch=[list(k) for k in keys[start:start+64]];v=bank.predict(batch,r['context']);v.update(retrieval.predict(batch,r))
                for n in SOURCES:probs[n].extend(v[n])
            cache={n:dict(zip(keys,v)) for n,v in probs.items()};F=tuple(sorted(full));fp={n:cache[n][F] for n in SOURCES}
            shap=checked_contributions(bank.tree,enc.encode(r['tokens'],r['context'])[None],len(labels))[0];cols={g:sorted({enc.token(t)[0] for t in ts}) for g,ts in groups.items()}
            def support(n,fs,target):
                p=cache[n][tuple(sorted(fs))];return float(p[labels.index(target)]) if p is not None else None
            def pred(n,fs):
                p=cache[n][tuple(sorted(fs))];return labels[int(p.argmax())] if p is not None else None
            for e,E,geo,variants in tasks:
                d=e['diagnosis'];di=labels.index(d);y=r['label'];eid=e['explanation_id']
                w={n:{g:float(fp[n][di])-float(cache[n][v][di]) for g,v in minus.items()} if fp[n] is not None and all(cache[n][v] is not None for v in minus.values()) else None for n in SOURCES}
                w['local_shap']={g:float(shap[di,cs].sum()) for g,cs in cols.items()};w['class_shap']={g:float(means[di,cs].sum()) for g,cs in cols.items()}
                for n in SOURCES:
                    p=cache[n][tuple(sorted(E))]
                    if p is not None and fp[n] is not None:emit(outputs['precision_pairs'],dict(explanation_id=eid,source=n,absolute=float(p[di]),full=float(fp[n][di]),legacy_retained=float(p[di]-fp[n][di]),corrected_retained=float(p[di])-float(fp[n][di])))
                for vid,fs in variants.items():
                    scores={'B1_fact':len(fs)/len(full),'B1_group':coverage({g:1. for g in groups},groups,set(fs))}
                    for n in SOURCES:
                        suffix='case' if n=='voting' else n
                        scores['B0_'+suffix]=support(n,full,d);scores['B2_A_'+suffix]=support(n,fs,d)
                        scores['B3_'+suffix]=coverage({g:max(0.,v) for g,v in w[n].items()},groups,set(fs)) if w[n] is not None else None
                    for n in ['local_shap','class_shap']:scores['B3_'+n]=coverage({g:max(0.,v) for g,v in w[n].items()},groups,set(fs))
                    probe={pr:dict(q_d=support(pr,fs,d),q_y=support(pr,fs,y),prediction=pred(pr,fs)) for pr in ['encoder','lightgbm']}
                    if any(p['q_d'] is None for p in probe.values()):raise ValueError('Unscorable probe variant; explicit missingness handling required')
                    emit(outputs['variant_scores'],dict(patient_id=pid,pattern_id=r['pattern_id'],explanation_id=eid,variant=vid,fact_ids=fs,fact_id_namespace='source_token',actual_fact_count=len(fs),d=d,y=y,scores=scores,probe=probe))
                for op,candidates in geo.items():
                    for selector,directions in cfg['selectors'].items():
                        for direction in directions:
                            seed=f"{cfg['seed']}|{pid}|{eid}|{op}|{selector}|{direction}"
                            s=select(candidates,w[selector],direction,seed,cfg['random_repeats'],cfg['zero_tolerance'])
                            base=dict(patient_id=pid,pattern_id=r['pattern_id'],explanation_id=eid,d=d,selector=selector,operation=op,direction=direction,split='test')
                            emit(outputs['attempts'],dict(base,**s))
                            if not s['available']:continue
                            for condition,g,repeat in [('high',s['high'],None),('low',s['low'],None)]+[('random',g,j) for j,g in enumerate(s['random'])]:
                                v=candidates[g];fs=v['fact_ids'];probe={}
                                for pr in ['encoder','lightgbm']:
                                    probe[pr]=dict(available=True,q_d_before=support(pr,E,d),q_d_after=support(pr,fs,d),q_y_before=support(pr,E,y),q_y_after=support(pr,fs,y),prediction_before=pred(pr,E),prediction_after=pred(pr,fs))
                                emit(outputs['interventions'],dict(base,condition=condition,repeat=repeat,group_id=g,actual_fact_count=v['actual_fact_count'],original_fact_ids=sorted(E),fact_ids=fs,changed_fact_ids=v['changed_fact_ids'],y=y,diagnosis_correct=d==y,source_weight=w[selector][g],probe=probe))
            if (i+1)%20==0:print('patients',i+1,'/',len(chosen),'seconds',round(time.time()-started,1),flush=True)
        state=dict(status='complete',patients=len(chosen))
    except Exception:state=dict(status='failed',error=traceback.format_exc());raise
    finally:
        for f in outputs.values():f.close()
        dump(folder/'run_manifest.json',state)

def export_blind(a):
    plan=read(ROOT/'configs/code10k_mapping_audit_20261007.json')
    if sha(a.out/'explanation_ledger.jsonl')!=plan['rq2_ledger_sha256']:raise ValueError('Audit ledger does not match accepted RQ2')
    patients=unique(a.sample/'data/patients_public.jsonl','patient_id');wanted={t['explanation_id']:t for t in plan['tasks']}
    records={g['id']:g for g in rows(a.sample/'generation/results.jsonl') if g['id'] in wanted}
    if set(records)!=set(wanted):raise ValueError('Audit explanations absent')
    folder=a.out/'reviewer_blind';folder.mkdir()
    with (folder/'tasks.jsonl').open('w',encoding='utf-8') as f:
        for t in plan['tasks']:
            g=records[t['explanation_id']];p=patients[t['patient_id']]
            if g['patient_id']!=t['patient_id']:raise ValueError('Audit patient join drift')
            emit(f,dict(audit_id=t['audit_id'],explanation=g['parsed']['explanation'],demographics=p['demographics'],patient_facts=p['facts']))
    (folder/'evidence_dictionary.json').write_bytes((a.sample/'data/evidence_dictionary.json').read_bytes())
    dump(a.out/'audit_key_PRIVATE.json',plan)
    dump(folder/'review_instructions.json',dict(scope='Independently identify all stated patient findings, including mismatches and uncertainty; map exact quotes to dictionary and observed values. Do not infer omitted mentions from the patient record.',blinding='No automatic citations, model scores, reference diagnosis or sampling-stratum flags provided.',required_fields=['audit_id','reviewer','review_date','review_type','status','mentions: quote/feature_id/asserted_value/matched_fact_ids/error_type','reviewed_cited_fact_ids','unresolved','notes'],empty_set='Only valid after explicit human completion',delivery='Raw material for spreadsheet preparation; do not send private key to annotator'))
    return dict(random=50,targeted_failures=len(plan['targeted_ids']),unique=len(plan['tasks']),human_review_completed=False)


def main_run(a):
    import torch,lightgbm
    fm,_,_=verify(a)
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_code10k_interventions.py','-v'],cwd=ROOT,check=True)
    ledger=prepare(a,fm);cfg=read(a.intervention_config)
    dump(a.out/'protocol_lock.json',dict(config=cfg,protocol_sha256=sha(a.intervention_config),explanation_ledger_sha256=sha(a.out/'explanation_ledger.jsonl'),fitted_manifest_sha256=sha(a.models/'fitted_sources.json'),class_shap_sha256=sha(a.rq1/'class_shap.json'),scope='fixed diagnosis structured citation add/delete; not edited natural language; original proposals, no label-based selection'))
    review=export_blind(a);print('Blind review materials',review,flush=True)
    processes=[];ids=a.gpus.split(',')
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate GPU')
    try:
        for i,g in enumerate(ids):
            cmd=[sys.executable,'-u',__file__,'--shard',str(i),'--shards',str(len(ids)),'--threads',str(a.threads)]
            for k in ['models','protocol','sample','rq1','config','out','intervention-config']:cmd+=['--'+k,str(getattr(a,k.replace('-','_')))]
            log=(a.out/f'worker_{i}.log').open('w',encoding='utf-8');processes.append((subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=g),stdout=log,stderr=subprocess.STDOUT),log))
        while any(p.poll() is None for p,_ in processes):
            print('Workers',[(i,p.poll()) for i,(p,_) in enumerate(processes)],flush=True);time.sleep(30)
        if any(p.returncode for p,_ in processes):raise RuntimeError('Worker failed; inspect logs')
    finally:
        for p,log in processes:
            if p.poll() is None:p.terminate();p.wait()
            log.close()
    counts={}
    for n in ['attempts','interventions','variant_scores','eligibility','precision_pairs']:
        count=0
        with (a.out/(n+'.jsonl')).open('w',encoding='utf-8') as f:
            for i in range(len(ids)):
                for r in rows(a.out/f'shard_{i}/{n}.jsonl'):emit(f,r);count+=1
        counts[n]=count
    eligibility=list(rows(a.out/'eligibility.jsonl'))
    if len(eligibility)!=6024 or {r['explanation_id'] for r in eligibility}!={r['explanation_id'] for r in ledger}:raise ValueError('Eligibility census drift')
    origins=[r for r in rows(a.out/'variant_scores.jsonl') if r['variant']=='original']
    if len(origins)!=sum(r['included'] for r in eligibility) or {r['explanation_id'] for r in origins}!={r['explanation_id'] for r in eligibility if r['included']}:raise ValueError('Missing original variants')
    dump(a.out/'scoring_manifest.json',dict(status='complete',output_sha256={n:sha(a.out/n) for n in ['protocol_lock.json','attempts.jsonl','interventions.jsonl','variant_scores.jsonl']},counts=counts))
    print('CPU intervention summaries',flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/analyze_code10k_intervention_effects.py'),'--source-dir',str(a.out),'--out',str(a.out/'effects')],cwd=ROOT,check=True)
    print('CPU version ranking summaries',flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/analyze_code10k_versions.py'),'--source-dir',str(a.out),'--config',str(ROOT/'configs/code10k_versions_20261007.json'),'--out',str(a.out/'versions')],cwd=ROOT,check=True)
    return dict(status='complete',counts=counts,review=review,models_trained=False,api_calls=0,scope='Structured citation interventions and versions on frozen test; mapping independent review pending')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key,default in [('models','artifacts/code10k_sources_v1'),('protocol','artifacts/code10k_protocol_v1'),('rq1','artifacts/code10k_rq1_v1_retry1'),('out','artifacts/code10k_interventions_v1'),('sample','inputs/ddxplus_sample'),('config',str(ROOT/'configs/code10k_rq2_20261007.json')),('intervention-config',str(ROOT/'configs/code10k_interventions_20261007.json'))]:p.add_argument('--'+key,type=Path,default=Path(default))
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--threads',type=int,default=4);p.add_argument('--shard',type=int);p.add_argument('--shards',type=int)
    a=p.parse_args()
    for k in ['models','protocol','rq1','out','sample','config','intervention_config']:setattr(a,k,getattr(a,k).resolve())
    if a.shard is not None:worker(a)
    else:
        a.out.mkdir(parents=True,exist_ok=False);state={};start=time.time()
        try:state=main_run(a)
        except Exception:state=dict(status='failed',error=traceback.format_exc());raise
        finally:
            state.update(seconds=time.time()-start,output_sha256={f.relative_to(a.out).as_posix():sha(f) for f in a.out.rglob('*') if f.is_file()},code_sha256={f.relative_to(ROOT).as_posix():sha(f) for d in ['scripts','src'] for f in (ROOT/d).rglob('*.py')});dump(a.out/'run_manifest.json',state)
            archive=a.out.with_name(a.out.name+'_reports.tar.gz')
            with tarfile.open(archive,'w:gz') as tar:
                for f in a.out.rglob('*'):
                    if f.is_file():tar.add(f,arcname=f.relative_to(a.out).as_posix())
            print('Return report:',archive,flush=True)
