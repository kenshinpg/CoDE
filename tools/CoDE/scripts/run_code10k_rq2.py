"""RQ2 natural evidence evaluation on the frozen 6024-explanation test ledger."""
import argparse,json,os,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prepare_code10k import sha,dump,rows,unique,stable
from run_code10k_sources import LEDGER,structural_record
from run_code10k_rq1 import verify,RetrievalSources,read,emit
from run_coverage_controls import coverage,permute
from code_eval.tree_subset import Encoding,checked_contributions
from code_eval.rq2_summary import SOURCES,FAMILIES,panel,add_pairwise_families

def prepare(a,fm):
    if sha(a.protocol/'explanation_ledger_PRIVATE.jsonl')!=LEDGER:raise ValueError('Citation snapshot drift')
    report=read(a.protocol/'cohort_report.json');gp=a.sample/'generation/results.jsonl'
    expected=[h for p,h in report['input_sha256'].items() if Path(p).name=='results.jsonl']
    if expected!=[sha(gp)]:raise ValueError('Original generation changed')
    ledger=unique(a.protocol/'explanation_ledger_PRIVATE.jsonl','explanation_id');out=[];seen=set()
    for g in rows(gp):
        eid=g['id']
        if eid in seen or eid not in ledger:raise ValueError('Generation ID mismatch')
        seen.add(eid);r=ledger[eid]
        if r['partition']!='test':continue
        diagnoses=g['parsed']['ranked_diagnoses'];text=g['parsed']['explanation']
        if g['patient_id']!=r['patient_id'] or stable(text)!=r['text_sha256'] or stable(diagnoses)!=r['diagnoses_sha256']:raise ValueError('Explanation join drift')
        if (diagnoses[0]==r['disease'])!=r['diagnosis_correct']:raise ValueError('Correctness sidecar mismatch')
        out.append(dict(r,diagnosis=diagnoses[0],target_supported=diagnoses[0] in fm['labels']))
    if seen!=set(ledger) or len(out)!=6024 or sum(r['reviewed_available'] for r in out)!=5382:raise ValueError('Frozen explanation census changed')
    with (a.out/'explanation_ledger.jsonl').open('w',encoding='utf-8') as f:
        for r in sorted(out,key=lambda r:r['explanation_id']):emit(f,r)
    return out

def worker(a):
    import torch
    from code_eval.frozen_bank import FrozenBank
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fm,fit,dictionary=verify(a);labels=fm['labels'];enc=Encoding(dictionary)
    if sha(a.rq1/'class_shap.json')!=read(a.config)['class_shap_sha256']:raise ValueError('Accepted class SHAP drift')
    cls=read(a.rq1/'class_shap.json');means=np.asarray(cls['mean_absolute'])
    if cls['labels']!=labels or cls['columns']!=enc.columns or cls['train_sha256']!=fm['voting_bank_sha256']:raise ValueError('Class SHAP provenance mismatch')
    paths={n:a.models/p for n,p in {'lightgbm':'source/tree/mask_augmented.txt','encoder':'encoder/encoder.pt','grasp':'formal/grasp.pt','cohd':'formal/cohd_counts.json'}.items()}
    bank=FrozenBank(paths,enc,labels,a.threads);retrieval=RetrievalSources(a,fit,labels,dictionary);gain=bank.tree.feature_importance(importance_type='gain')
    split=unique(a.protocol/'patient_split_PRIVATE.jsonl','patient_id');patients=unique(a.sample/'data/patients_public.jsonl','patient_id')
    if sha(a.out/'explanation_ledger.jsonl')!=read(a.out/'evaluation_lock.json')['explanation_ledger_sha256']:raise ValueError('Prepared ledger changed')
    by=defaultdict(list)
    for r in rows(a.out/'explanation_ledger.jsonl'):by[r['patient_id']].append(r)
    chosen=sorted(by)[a.shard::a.shards];folder=a.out/f'shard_{a.shard}';folder.mkdir();state={};started=time.time()
    try:
        with (folder/'scores.jsonl').open('w',encoding='utf-8') as dest:
            for i,pid in enumerate(chosen):
                r=structural_record(patients[pid],split[pid]);tokens={f['fact_id']:f['provenance']['source_token'] for f in patients[pid]['facts'] if f.get('visible',True) and f['coverage_scope']=='symptoms_antecedents'}
                groups=enc.groups(r['tokens']);full=set(r['tokens']);keys={tuple(sorted(full))};minus={g:tuple(sorted(full-set(ts))) for g,ts in groups.items()};keys.update(minus.values())
                cited={}
                for e in by[pid]:
                    if e['reviewed_available']:
                        if not set(e['cited_fact_ids'])<=tokens.keys():raise ValueError('Nonpatient citation')
                        cited[e['explanation_id']]=tuple(sorted({tokens[f] for f in e['cited_fact_ids']}));keys.add(cited[e['explanation_id']])
                keys=sorted(keys);probs=bank.predict([list(k) for k in keys],r['context']);probs.update(retrieval.predict([list(k) for k in keys],r));cache={n:dict(zip(keys,v)) for n,v in probs.items()}
                fp={n:cache[n][tuple(sorted(full))] for n in SOURCES};both=all(fp[n] is not None and labels[int(fp[n].argmax())]==r['label'] for n in ['encoder','lightgbm'])
                shap=checked_contributions(bank.tree,enc.encode(r['tokens'],r['context'])[None],len(labels))[0];cols={g:sorted({enc.token(t)[0] for t in ts}) for g,ts in groups.items()}
                for e in by[pid]:
                    mapped=e['reviewed_available'];ts=cited.get(e['explanation_id']);scores={};reasons={};d=labels.index(e['diagnosis']) if e['target_supported'] else None
                    if mapped:
                        cs=set(ts);scores.update(fact_coverage=len(cs)/len(full),group_coverage=coverage({g:1. for g in groups},groups,cs),words=e['word_count'])
                        for name in SOURCES:
                            w={g:max(0.,float(fp[name][d]-cache[name][v][d])) for g,v in minus.items()} if d is not None and fp[name] is not None and all(cache[name][v] is not None for v in minus.values()) else None
                            p=cache[name][ts]
                            scores[name+'_coverage']=coverage(w,groups,cs) if w is not None else None
                            scores[name+'_absolute']=float(p[d]) if d is not None and p is not None else None
                            scores[name+'_retained']=float(p[d]-fp[name][d]) if d is not None and p is not None and fp[name] is not None else None
                            reasons[name+'_coverage']=None if scores[name+'_coverage'] is not None else ('unsupported_target' if d is None else ('zero_positive_mass' if w is not None else 'source_unavailable'))
                        if d is not None:
                            weights={'tree_gain':{g:float(gain[c].sum()) for g,c in cols.items()},'tree_positive_shap':{g:max(0.,float(shap[d,c].sum())) for g,c in cols.items()},'tree_absolute_shap':{g:float(np.abs(shap[d,c]).sum()) for g,c in cols.items()},'tree_class_shap':{g:float(means[d,c].sum()) for g,c in cols.items()}}
                            for name,w in weights.items():scores[name]=coverage(w,groups,cs)
                            for j in range(10):scores[f'permuted_{j:02d}']=coverage(permute(weights['tree_positive_shap'],f"261006|{pid}|{e['diagnosis']}|{j}"),groups,cs)
                    for probe in ['encoder','lightgbm']:
                        pp=cache[probe][ts] if mapped else None;available=pp is not None and fp[probe] is not None
                        emit(dest,dict(explanation_id=e['explanation_id'],patient_id=pid,pattern_id=r['pattern_id'],component=r['component'],reference_diagnosis=r['label'],diagnosis=e['diagnosis'],diagnosis_correct=e['diagnosis_correct'],target_supported=e['target_supported'],
                            mapping_available=mapped,repair_status=e['repair_status'],cited_fact_count=len(ts) if mapped else None,cited_tokens=list(ts) if mapped else None,
                            probe=probe,probe_available=available,both_full_correct=both,full_correct=bool(fp[probe] is not None and labels[int(fp[probe].argmax())]==r['label']),
                            reference_failure=bool(labels[int(pp.argmax())]!=r['label']) if available else None,proposal_failure=bool(labels[int(pp.argmax())]!=e['diagnosis']) if available else None,
                            full_probabilities=fp[probe].tolist() if fp[probe] is not None else None,cited_probabilities=pp.tolist() if pp is not None else None,scores=scores,coverage_unavailability=reasons))
                if (i+1)%20==0:print('Shard',a.shard,i+1,'/',len(chosen),'patients; seconds',round(time.time()-started,1),flush=True)
        state=dict(status='complete',patients=len(chosen))
    except Exception:state=dict(status='failed',error=traceback.format_exc());raise
    finally:dump(folder/'run_manifest.json',state)

def calculate(task):
    path,probe,stratum,endpoint,family=task;add_pairwise_families()
    return panel(list(rows(Path(path))),probe,stratum,endpoint,family)

def main_run(a):
    fm,_,_=verify(a)
    if sha(a.rq1/'class_shap.json')!=read(a.config)['class_shap_sha256']:raise ValueError('RQ1 class attribution changed')
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_code10k_rq2.py','-v'],cwd=ROOT,check=True)
    ledger=prepare(a,fm);add_pairwise_families()
    dump(a.out/'evaluation_lock.json',dict(fitted_manifest_sha256=sha(a.models/'fitted_sources.json'),class_shap_sha256=sha(a.rq1/'class_shap.json'),original_ledger_sha256=LEDGER,explanation_ledger_sha256=sha(a.out/'explanation_ledger.jsonl'),labels=fm['labels'],families=FAMILIES,bootstrap=1000,seed=261006,primary='correct proposals; both full probes correct; reference-recovery failure',sensitivity='incorrect/all proposals; proposal agreement as separate nonclinical endpoint',models_trained=False,semantic_audit='not performed by this script'))
    ids=a.gpus.split(',');processes=[]
    if len(set(ids))!=len(ids):raise ValueError('Distinct GPUs required')
    try:
        for i,g in enumerate(ids):
            cmd=[sys.executable,'-u',__file__,'--shard',str(i),'--shards',str(len(ids)),'--threads',str(a.threads)]
            for key in ['models','protocol','sample','rq1','config','out']:cmd+=['--'+key,str(getattr(a,key))]
            log=(a.out/f'worker_{i}.log').open('w',encoding='utf-8');p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=g),stdout=log,stderr=subprocess.STDOUT);processes.append((p,log))
        while any(p.poll() is None for p,_ in processes):
            print('Scoring workers',[(i,'running' if p.poll() is None else p.returncode) for i,(p,_) in enumerate(processes)],flush=True);time.sleep(30)
        if any(p.returncode for p,_ in processes):raise RuntimeError('Scoring worker failed; inspect logs')
    finally:
        for p,log in processes:
            if p.poll() is None:p.terminate();p.wait()
            log.close()
    records=[]
    for i in range(len(ids)):records.extend(rows(a.out/f'shard_{i}/scores.jsonl'))
    if len(records)!=12048 or len({(r['explanation_id'],r['probe']) for r in records})!=12048 or {r['explanation_id'] for r in records}!={r['explanation_id'] for r in ledger}:raise ValueError('Incomplete score ledger')
    scorepath=a.out/'scores.jsonl'
    with scorepath.open('w',encoding='utf-8') as f:
        for r in sorted(records,key=lambda r:(r['explanation_id'],r['probe'])):emit(f,r)
    flow={}
    for probe in ['encoder','lightgbm']:
        rs=[r for r in records if r['probe']==probe];groups=defaultdict(Counter)
        for r in rs:
            key='mapping_unavailable' if not r['mapping_available'] else ('target_unsupported' if not r['target_supported'] else ('probe_unavailable' if not r['probe_available'] else ('full_probe_not_both_correct' if not r['both_full_correct'] else 'eligible_before_score_availability')))
            for group in ['all','disease:'+r['reference_diagnosis'],'component:'+r['component'],'proposal:'+('correct' if r['diagnosis_correct'] else 'incorrect')]:groups[group][key]+=1
        flow[probe]={k:dict(v) for k,v in groups.items()}
    dump(a.out/'population_flow.json',flow)
    tasks=[(str(scorepath),probe,stratum,'reference',family) for probe in ['encoder','lightgbm'] for stratum in ['correct','incorrect','all'] for family in FAMILIES]
    tasks +=[(str(scorepath),probe,stratum,'proposal',family) for probe in ['encoder','lightgbm'] for stratum in ['incorrect','all'] for family in ['coverage_sources','tree_rules','direct_support','quantity_only']]
    print('Patient bootstrap panels:',len(tasks),flush=True);panels=[]
    with ProcessPoolExecutor(max_workers=a.summary_workers) as pool:
        for i,result in enumerate(pool.map(calculate,tasks),1):
            panels.append(result);dump(a.out/'rq2_summary.json',panels);print('Summary',i,'/',len(tasks),result['family'],result['probe'],result['stratum'],'common',result['common'],'failures',result['failures'],flush=True)
    dump(a.out/'rq2_information_audit.json',[{k:r[k] for k in ['probe','stratum','endpoint','family','information_audit']} for r in panels])
    dump(a.out/'review_requirements_status.json',dict(automatic_comparisons='completed',population_and_event_audit='completed',
        leave_one_disease='completed: descriptive correct-proposal reference endpoint paired panels',
        independent_semantic_review='pending',corrected_mapping_sensitivity='pending independent reviewed mappings',
        sampling_origin_and_actual_extractor_identity='requires source documentation',controlled_experiments='separate RQ3 task'))
    return dict(status='complete',patients=1506,planned_explanations=6024,mapped_explanations=5382,score_rows=len(records),panels=len(panels),models_trained=False,api_calls=0,scope='RQ2 automatic natural mappings only; independent mapping audit and clinical explanation validity not established')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key,default in [('models','artifacts/code10k_sources_v1'),('protocol','artifacts/code10k_protocol_v1'),('rq1','artifacts/code10k_rq1_v1_retry1'),('out','artifacts/code10k_rq2_v1'),('sample','inputs/ddxplus_sample'),('config',str(ROOT/'configs/code10k_rq2_20261007.json'))]:p.add_argument('--'+key,type=Path,default=Path(default))
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--threads',type=int,default=4);p.add_argument('--summary-workers',type=int,default=8);p.add_argument('--shard',type=int);p.add_argument('--shards',type=int)
    a=p.parse_args()
    for key in ['models','protocol','rq1','out','sample','config']:setattr(a,key,getattr(a,key).resolve())
    if a.shard is not None:worker(a)
    else:
        a.out.mkdir(parents=True,exist_ok=False);state={};start=time.time()
        try:state=main_run(a)
        except Exception:state=dict(status='failed',error=traceback.format_exc());raise
        finally:
            state.update(seconds=time.time()-start,output_sha256={f.relative_to(a.out).as_posix():sha(f) for f in a.out.rglob('*') if f.is_file()},code_sha256={f.relative_to(ROOT).as_posix():sha(f) for folder in ['scripts','src'] for f in (ROOT/folder).rglob('*.py')});dump(a.out/'run_manifest.json',state)
            archive=a.out.with_name(a.out.name+'_reports.tar.gz')
            with tarfile.open(archive,'w:gz') as tar:
                for f in a.out.rglob('*'):
                    if f.is_file():tar.add(f,arcname=f.relative_to(a.out).as_posix())
            print('Return report:',archive,flush=True)
