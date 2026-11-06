"""TRACE RQ2: reused code10k scoring; frozen full-match adapter and policy sensitivities."""
import argparse,json,os,sys,time,traceback,subprocess,tarfile,shutil,hashlib
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_trace2400_evaluation import inputs,setup,emit
from run_trace2400_sources import read,rows,dump,sha
from run_coverage_controls import coverage,permute
from code_eval.tree_subset import checked_contributions
from code_eval.trace2400.rq2_summary import SOURCES,FAMILIES,panel,add_pairwise_families
from code_eval.trace2400.natural_contract import index,adapt,policy_records,census


def prepare(a):
    train,labels,enc,paths,split,query=inputs(a)
    mm=read(a.match/'manifest.json');rm=read(a.release/'release_manifest.json')
    if sha(a.release/'release_manifest.json')!=mm['source_manifest_sha256']:raise ValueError('Release identity drift')
    if sha(a.view/'patients_model_input_PRIVATE.jsonl')!=mm['source_files']['frozen_input/patients_model_input_PRIVATE.jsonl']:raise ValueError('Frozen patient input drift')
    for name in ['generation_records_PRIVATE.jsonl','explanations_PRIVATE.jsonl']:
        if sha(a.release/name)!=rm['outputs'][name]:raise ValueError('Release file drift: '+name)
    generations=index(rows(a.release/'generation_records_PRIVATE.jsonl'),'task_id')
    original=index(rows(a.release/'explanations_PRIVATE.jsonl'),'task_id')
    patients=index(rows(a.view/'patients_model_input_PRIVATE.jsonl'),'patient_id')
    matched=index(rows(a.match/'export/matching_candidates_PRIVATE.jsonl'),'task_id')
    if set(matched)!=set(original) or set(generations)!=set(original) or len(matched)!=9600:raise ValueError('Full task census mismatch')
    if Counter(r['patient_id'] for r in matched.values())!=Counter({p:4 for p in patients}):raise ValueError('Four tasks per admission required')
    observed=Counter(r['visible_fact_scan_status'] for r in matched.values())
    if dict(observed)!=read(a.match/'export/report.json')['explanation_statuses']:raise ValueError('Matching report drift')
    output=[]
    for tid,m in matched.items():
        pid=m['patient_id'];s=split[pid]
        if original[tid]['patient_id']!=pid or m['partition']!=s['partition']:raise ValueError('Join drift')
        r=adapt(m,generations[tid],patients[pid],s,labels)
        if s['partition']==a.partition:output.append(r)
    if len(output)!=4*len(query):raise ValueError('Partition census failed')
    with (a.out/'explanation_ledger.jsonl').open('w',encoding='utf-8') as f:
        for r in sorted(output,key=lambda r:r['explanation_id']):emit(f,r)
    cls=read(a.rq1/'class_shap.json')
    if cls['train_sha256']!=sha(a.view/'train_records_PRIVATE.jsonl') or cls['tree_sha256']!=sha(paths['lightgbm']):raise ValueError('Class SHAP fit provenance mismatch')
    previous=read(a.rq1/'evaluation_lock.json')
    if previous['class_shap_sha256']!=sha(a.rq1/'class_shap.json'):raise ValueError('Accepted class SHAP drift')
    shutil.copy2(a.rq1/'class_shap.json',a.out/'class_shap.json')
    lock=dict(dataset='TRACE2400-v1',partition=a.partition,input_mode='model_operational_candidate',
        class_shap_sha256=sha(a.out/'class_shap.json'),explanation_ledger_sha256=sha(a.out/'explanation_ledger.jsonl'),
        code_sha256={str(f.relative_to(ROOT)):sha(f) for folder in ['scripts','src'] for f in (ROOT/folder).rglob('*.py')},
        source_files={str(x):sha(x) for x in [a.models/'fitted_sources.json',a.match/'manifest.json',a.match/'export/matching_candidates_PRIVATE.jsonl',a.release/'generation_records_PRIVATE.jsonl']},
        policies=['complete_only','include_partial_exact'],primary='complete_only; correct proposal; both full probes correct; reference recovery',
        sensitivity='partial scans use only exact decisions; incorrect/all proposal strata; proposal agreement; leave-one-disease',
        bootstrap=1000,seed=261006,semantic_review='pending independent audit; operational labels are not gold',models_trained=False,api_calls=0)
    dump(a.out/'evaluation_lock.json',lock)
    dump(a.out/'input_audit.json',dict(all_tasks=len(matched),all_statuses=dict(observed),partition=a.partition,planned=census(output),
        statuses={k:census([r for r in output if r['repair_status']==k]) for k in ['COMPLETE','PARTIAL','UNAVAILABLE']},
        by_disease={d:census([r for r in output if r['reference_diagnosis']==d]) for d in labels}))
    return output

def worker(a):
    bank,retrieval,labels,enc,means,query=setup(a)
    gain=bank.tree.feature_importance(importance_type='gain')
    patients=index(rows(a.view/'patients_model_input_PRIVATE.jsonl'),'patient_id')
    if sha(a.out/'explanation_ledger.jsonl')!=read(a.out/'evaluation_lock.json')['explanation_ledger_sha256']:raise ValueError('Prepared ledger drift')
    by=defaultdict(list)
    for r in rows(a.out/'explanation_ledger.jsonl'):by[r['patient_id']].append(r)
    chosen=sorted(by)[a.shard::a.shards];folder=a.out/f'shard_{a.shard}';folder.mkdir();state={};started=time.time()
    try:
        with (folder/'scores.jsonl').open('w',encoding='utf-8') as dest:
            for i,pid in enumerate(chosen):
                r=query[pid];tokens={f['fact_id']:f['feature_id']+'_@_'+f['state'] for f in patients[pid]['facts']}
                if set(tokens.values())!=set(r['tokens']):raise ValueError('Visible fact/token mismatch')
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
                        emit(dest,dict(subject_group=r['subject_group'],scan_status=e['repair_status'],annotation_protocols=e['annotation_protocols'],explanation_id=e['explanation_id'],patient_id=pid,pattern_id=r['pattern_id'],component=r['component'],reference_diagnosis=r['label'],diagnosis=e['diagnosis'],diagnosis_correct=e['diagnosis_correct'],target_supported=e['target_supported'],
                            mapping_available=mapped,repair_status=e['repair_status'],cited_fact_count=len(ts) if mapped else None,cited_tokens=list(ts) if mapped else None,
                            probe=probe,probe_available=available,both_full_correct=both,full_correct=bool(fp[probe] is not None and labels[int(fp[probe].argmax())]==r['label']),
                            reference_failure=bool(labels[int(pp.argmax())]!=r['label']) if available else None,proposal_failure=bool(labels[int(pp.argmax())]!=e['diagnosis']) if available else None,
                            full_probabilities=fp[probe].tolist() if fp[probe] is not None else None,cited_probabilities=pp.tolist() if pp is not None else None,scores=scores,coverage_unavailability=reasons))
                if (i+1)%20==0:print('Shard',a.shard,i+1,'/',len(chosen),'patients; seconds',round(time.time()-started,1),flush=True)
        state=dict(status='complete',patients=len(chosen))
    except Exception:state=dict(status='failed',error=traceback.format_exc());raise
    finally:dump(folder/'run_manifest.json',state)


def calculate(task):
    path,policy,probe,stratum,endpoint,family=task;add_pairwise_families()
    rs=policy_records(rows(Path(path)),policy)
    out=panel(rs,probe,stratum,endpoint,family)
    out['mapping_policy']=policy
    out['interpretation']='Frozen TRACE natural-explanation comparison; inferred-subject bootstrap; unadjusted intervals; endpoint is predictive support, not clinical quality.'
    return out


def summarize(a):
    if sha(a.out/'explanation_ledger.jsonl')!=read(a.out/'evaluation_lock.json')['explanation_ledger_sha256']:raise ValueError('Summary ledger drift')
    records=[]
    for folder in sorted(a.out.glob('shard_*')):
        if read(folder/'run_manifest.json')['status']!='complete':raise ValueError('Incomplete shard')
        records.extend(rows(folder/'scores.jsonl'))
    ledger=rows(a.out/'explanation_ledger.jsonl');expected={(r['explanation_id'],p) for r in ledger for p in ['encoder','lightgbm']}
    if len(records)!=len(expected) or {(r['explanation_id'],r['probe']) for r in records}!=expected:raise ValueError('Score census mismatch')
    path=a.out/'scores_PRIVATE.jsonl'
    with path.open('w',encoding='utf-8') as f:
        for r in sorted(records,key=lambda r:(r['explanation_id'],r['probe'])):emit(f,r)
    add_pairwise_families();tasks=[];flow={}
    for policy in ['complete_only','include_partial_exact']:
        rs=policy_records(records,policy);flow[policy]={}
        for probe in ['encoder','lightgbm']:
            selected=[r for r in rs if r['probe']==probe];steps=[]
            for name,pred in [('planned',lambda r:True),('mapping_available',lambda r:r['mapping_available']),('target_supported',lambda r:r['target_supported']),('probe_available',lambda r:r['probe_available']),('both_full_correct',lambda r:r['both_full_correct']),('correct_proposal',lambda r:r['diagnosis_correct'])]:
                selected=[r for r in selected if pred(r)];steps.append(dict(stage=name,**census(selected),failures=census([r for r in selected if r['reference_failure'] is True])))
            flow[policy][probe]=steps
            for stratum in ['correct','incorrect','all']:
                for family in FAMILIES:tasks.append((str(path),policy,probe,stratum,'reference',family))
            for stratum in ['incorrect','all']:
                for family in ['coverage_sources','tree_rules','direct_support','quantity_only']:tasks.append((str(path),policy,probe,stratum,'proposal',family))
    dump(a.out/'population_flow.json',flow)
    print('Bootstrap panels',len(tasks),flush=True);panels=[]
    with ProcessPoolExecutor(max_workers=a.summary_workers) as pool:
        for i,result in enumerate(pool.map(calculate,tasks),1):
            panels.append(result);dump(a.out/'rq2_summary.json',panels)
            print('Summary',i,'/',len(tasks),result['mapping_policy'],result['family'],'common',result['common'],'failures',result['failures'],flush=True)
    dump(a.out/'rq2_information_audit.json',[{k:r[k] for k in ['mapping_policy','probe','stratum','endpoint','family','information_audit']} for r in panels])
    dump(a.out/'review_requirements_status.json',dict(population_and_event_audit='complete',automatic_mapping_policy_sensitivity='complete',
        independent_semantic_review='pending',corrected_mapping_sensitivity='pending reviewed mappings',clinical_validity='not established by this script'))
    return dict(status='complete',partition=a.partition,planned_explanations=len(ledger),score_rows=len(records),panels=len(panels),api_calls=0,models_trained=False)


def main(a):
    if a.summarize_only:return summarize(a)
    import torch,lightgbm
    ids=a.gpus.split(',')
    if not torch.cuda.is_available() or torch.cuda.device_count()<len(ids):raise RuntimeError('Need visible C550 GPUs in vendor torch environment')
    if len(set(ids))!=len(ids):raise ValueError('Duplicate GPUs')
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_trace_natural.py','-v'],check=True,cwd=ROOT)
    prepare(a)
    # Reuse accepted parity path. Its artifacts are kept separate from RQ2 lock.
    parity=a.out/'parity';parity.mkdir()
    cmd=[sys.executable,'-u',str(ROOT/'scripts/run_trace2400_evaluation.py'),'--worker','parity','--partition',a.partition,'--out',str(parity),'--threads',str(a.threads)]
    for k in ['view','models','first']:cmd+=['--'+k,str(getattr(a,k))]
    subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=ids[0]),check=True)
    jobs=[]
    try:
        for i,gpu in enumerate(ids):
            cmd=[sys.executable,'-u',__file__,'--shard',str(i),'--shards',str(len(ids)),'--partition',a.partition,'--threads',str(a.threads)]
            for k in ['view','models','first','match','release','rq1','out']:cmd+=['--'+k,str(getattr(a,k))]
            log=(a.out/f'worker_{i}.log').open('w',encoding='utf-8')
            jobs.append((subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT),log))
        while any(p.poll() is None for p,l in jobs):
            print('Scoring workers',[(i,p.poll()) for i,(p,l) in enumerate(jobs)],flush=True);time.sleep(20)
        if any(p.returncode for p,l in jobs):raise RuntimeError('Scoring failed; inspect worker logs')
    finally:
        for p,l in jobs:
            if p.poll() is None:p.terminate();p.wait()
            l.close()
    return summarize(a)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    base='inputs/trace_production/'
    for key,default in [('view','artifacts/trace2400_view_v2_names'),('models','artifacts/trace2400_extended_sources_v1'),('first','artifacts/trace2400_sources_v1'),('rq1','artifacts/trace2400_evaluation_v3'),('match',base+'trace_match_full_v1'),('release',base+'trace_release_v1/run_20261015_005327_147'),('out','artifacts/trace2400_rq2_v1')]:p.add_argument('--'+key,type=Path,default=Path(default))
    p.add_argument('--partition',choices=['development','test'],default='test');p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--threads',type=int,default=4);p.add_argument('--summary-workers',type=int,default=8)
    p.add_argument('--shard',type=int);p.add_argument('--shards',type=int);p.add_argument('--summarize-only',action='store_true')
    a=p.parse_args()
    for k in ['view','models','first','rq1','match','release','out']:setattr(a,k,getattr(a,k).resolve())
    if a.shard is not None:worker(a)
    else:
        if not a.summarize_only:a.out.mkdir(parents=True,exist_ok=False)
        state={};start=time.time()
        try:state=main(a)
        except Exception:state=dict(status='failed',error=traceback.format_exc());raise
        finally:
            state['seconds']=time.time()-start;dump(a.out/'run_manifest.json',state)
            archive=a.out.with_name(a.out.name+'_reports.tar.gz')
            with tarfile.open(archive,'w:gz') as tar:
                # Aggregate results only; patient-level score/ledger files remain on server.
                for name in ['run_manifest.json','input_audit.json','evaluation_lock.json','population_flow.json','rq2_summary.json','rq2_information_audit.json','review_requirements_status.json']:
                    f=a.out/name
                    if f.exists():tar.add(f,arcname=name)
                for f in (a.out/'parity').glob('*.json'):tar.add(f,arcname='parity/'+f.name)
                for f in a.out.glob('worker_*.log'):tar.add(f,arcname=f.name)
            print('Return report:',archive,flush=True)
