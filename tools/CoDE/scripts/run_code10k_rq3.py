"""RQ3 controlled equal-fact-count subsets on all frozen test patients."""
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

from run_controlled_sets import sampled_sets

def support_scores(subset_probability, full_probability):
    # Convert BEFORE subtraction: float(np.float32(a)-np.float32(b)) has
    # already rounded the difference and need not satisfy A-R == b in float64.
    a=float(subset_probability);b=float(full_probability)
    return a,a-b

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
    chosen=sorted(pid for pid,r in split.items() if r['partition']=='test')[a.shard::a.shards];folder=a.out/f'shard_{a.shard}';folder.mkdir();state={};started=time.time()
    try:
        with (folder/'scores.jsonl').open('w',encoding='utf-8') as dest:
            for i,pid in enumerate(chosen):
                r=structural_record(patients[pid],split[pid]);tokens={f['fact_id']:f['provenance']['source_token'] for f in patients[pid]['facts'] if f.get('visible',True) and f['coverage_scope']=='symptoms_antecedents'}
                groups=enc.groups(r['tokens']);full=set(r['tokens']);keys={tuple(sorted(full))};minus={g:tuple(sorted(full-set(ts))) for g,ts in groups.items()};keys.update(minus.values())
                cited={};tasks=[]
                for fraction in [.25,.5,.75]:
                    count,sets=sampled_sets(groups,261006,pid,fraction,10)
                    for j,ts in enumerate(sets):
                        eid=f'{pid}|{fraction}|{j}';cited[eid]=tuple(ts);keys.add(tuple(ts))
                        tasks.append(dict(explanation_id=eid,reviewed_available=True,target_supported=True,word_count=None,repair_status='NOT_APPLICABLE_STRUCTURED_SUBSET',fraction=fraction,requested_sets=10,unique_sets=len(sets),anchor_fact_count=count))
                keys=sorted(keys);probs=bank.predict([list(k) for k in keys],r['context']);probs.update(retrieval.predict([list(k) for k in keys],r));cache={n:dict(zip(keys,v)) for n,v in probs.items()}
                fp={n:cache[n][tuple(sorted(full))] for n in SOURCES};both=all(fp[n] is not None and labels[int(fp[n].argmax())]==r['label'] for n in ['encoder','lightgbm'])
                shap=checked_contributions(bank.tree,enc.encode(r['tokens'],r['context'])[None],len(labels))[0];cols={g:sorted({enc.token(t)[0] for t in ts}) for g,ts in groups.items()}
                target=labels[int(fp['lightgbm'].argmax())]
                for e in tasks:
                    e.update(diagnosis=target,diagnosis_correct=target==r['label'])
                    mapped=e['reviewed_available'];ts=cited.get(e['explanation_id']);scores={};reasons={};d=labels.index(e['diagnosis']) if e['target_supported'] else None
                    if mapped:
                        cs=set(ts);scores.update(fact_coverage=len(cs)/len(full),group_coverage=coverage({g:1. for g in groups},groups,cs),words=e['word_count'])
                        for name in SOURCES:
                            w={g:max(0.,float(fp[name][d]-cache[name][v][d])) for g,v in minus.items()} if d is not None and fp[name] is not None and all(cache[name][v] is not None for v in minus.values()) else None
                            p=cache[name][ts]
                            scores[name+'_coverage']=coverage(w,groups,cs) if w is not None else None
                            scores[name+'_absolute']=float(p[d]) if d is not None and p is not None else None
                            scores[name+'_retained']=support_scores(p[d],fp[name][d])[1] if d is not None and p is not None and fp[name] is not None else None
                            reasons[name+'_coverage']=None if scores[name+'_coverage'] is not None else ('unsupported_target' if d is None else ('zero_positive_mass' if w is not None else 'source_unavailable'))
                        if d is not None:
                            weights={'tree_gain':{g:float(gain[c].sum()) for g,c in cols.items()},'tree_positive_shap':{g:max(0.,float(shap[d,c].sum())) for g,c in cols.items()},'tree_absolute_shap':{g:float(np.abs(shap[d,c]).sum()) for g,c in cols.items()},'tree_class_shap':{g:float(means[d,c].sum()) for g,c in cols.items()}}
                            for name,w in weights.items():scores[name]=coverage(w,groups,cs)
                            for j in range(10):scores[f'permuted_{j:02d}']=coverage(permute(weights['tree_positive_shap'],f"261006|{pid}|{e['diagnosis']}|{j}"),groups,cs)
                    for probe in ['encoder','lightgbm']:
                        pp=cache[probe][ts] if mapped else None;available=pp is not None and fp[probe] is not None
                        emit(dest,dict(explanation_id=e['explanation_id'],patient_id=pid,pattern_id=r['pattern_id'],component=r['component'],fraction=e['fraction'],requested_sets=e['requested_sets'],unique_sets=e['unique_sets'],anchor_fact_count=e['anchor_fact_count'],reference_diagnosis=r['label'],diagnosis=e['diagnosis'],diagnosis_correct=e['diagnosis_correct'],target_supported=e['target_supported'],
                            mapping_available=mapped,repair_status=e['repair_status'],cited_fact_count=len(ts) if mapped else None,cited_tokens=list(ts) if mapped else None,
                            probe=probe,probe_available=available,both_full_correct=both,full_correct=bool(fp[probe] is not None and labels[int(fp[probe].argmax())]==r['label']),
                            reference_failure=bool(labels[int(pp.argmax())]!=r['label']) if available else None,proposal_failure=bool(labels[int(pp.argmax())]!=e['diagnosis']) if available else None,
                            full_probabilities=fp[probe].tolist() if fp[probe] is not None else None,cited_probabilities=pp.tolist() if pp is not None else None,scores=scores,coverage_unavailability=reasons))
                if (i+1)%20==0:print('Shard',a.shard,i+1,'/',len(chosen),'patients; seconds',round(time.time()-started,1),flush=True)
        state=dict(status='complete',patients=len(chosen))
    except Exception:state=dict(status='failed',error=traceback.format_exc());raise
    finally:dump(folder/'run_manifest.json',state)


CONTROL_FAMILIES=['coverage_sources','tree_rules','direct_support','retrieval','quantity_only',
 'paired_tree_positive_shap_vs_lightgbm_absolute','paired_cohd_coverage_vs_lightgbm_absolute',
 'paired_cohd_coverage_vs_cohd_absolute','paired_tabr_coverage_vs_group_coverage',
 'paired_tabr_absolute_vs_voting_absolute','paired_tabr_absolute_vs_no_retrieval_absolute']

def register_families():
    add_pairwise_families()
    for baseline in ['voting_absolute','no_retrieval_absolute']:
        FAMILIES['paired_tabr_absolute_vs_'+baseline]=['fact_coverage','group_coverage','tabr_absolute',baseline]

def controlled_panel(records,probe,fraction, family,bootstrap=1000):
    register_families()
    selected=[r for r in records if r['fraction']==fraction]
    result=panel(selected,probe,'all','reference',family,bootstrap)
    result.update(fraction=fraction,kind='controlled',target='full-record LightGBM prediction; equals reference after primary both-full-correct filter',
                  interpretation='Structured equal-fact-count subsets, not natural rationales or edited text. Primary both-full-correct population. Unadjusted intervals; self-source cells descriptive.')
    return result

def calculate(task):
    path,probe,fraction,family=task
    return controlled_panel(list(rows(Path(path))),probe,fraction,family)

def validate_records(records,expected_patients):
    if {r['patient_id'] for r in records}!=set(expected_patients):raise ValueError('Test census differs')
    if len({(r['explanation_id'],r['probe']) for r in records})!=len(records):raise ValueError('Duplicate score record')
    blocks=defaultdict(list);offsets={}
    for r in records:
        blocks[(r['patient_id'],r['fraction'],r['probe'])].append(r)
        if r['both_full_correct'] and not r['diagnosis_correct']:raise ValueError('Primary target not reference')
        for n in SOURCES:
            a=r['scores'].get(n+'_absolute');v=r['scores'].get(n+'_retained')
            if a is not None and v is not None:
                key=(r['patient_id'],r['diagnosis'],n);offset=a-v
                if key in offsets and not np.isclose(offsets[key],offset,atol=1e-12,rtol=0):raise ValueError(f'A/R is not a constant shift: key={key}, prior={offsets[key]:.17g}, current={offset:.17g}, delta={offset-offsets[key]:.3g}')
                offsets[key]=offset
    audit=[]
    for pid in sorted(expected_patients):
        for fraction in [.25,.5,.75]:
            left=blocks[(pid,fraction,'encoder')];right=blocks[(pid,fraction,'lightgbm')]
            if not left or {r['explanation_id'] for r in left}!={r['explanation_id'] for r in right}:raise ValueError('Incomplete probe/budget block')
            if len({tuple(r['cited_tokens']) for r in left})!=len(left):raise ValueError('Duplicate sampled subset')
            if any(r['cited_fact_count']!=r['anchor_fact_count'] or r['unique_sets']!=len(left) for r in left+right):raise ValueError('Sampling count mismatch')
            audit.append(dict(patient_id=pid,fraction=fraction,requested=10,unique=len(left),fact_count=left[0]['anchor_fact_count']))
    return dict(patients=len(expected_patients),unique_sets=sum(r['unique'] for r in audit),blocks=audit,ar_constant_shift_checked=True)

def main_run(a):
    # Fail before launching GPU workers if the interpreter lacks a dependency.
    import torch,lightgbm
    fm,_,_=verify(a)
    if sha(a.rq1/'class_shap.json')!=read(a.config)['class_shap_sha256']:raise ValueError('Class SHAP drift')
    for test in ['test_code10k_rq3.py','test_reserved_evaluation.py']:
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p',test,'-v'],cwd=ROOT,check=True)
    expected={r['patient_id'] for r in rows(a.protocol/'patient_split_PRIVATE.jsonl') if r['partition']=='test'}
    if len(expected)!=1506:raise ValueError('Test census changed')
    register_families()
    dump(a.out/'evaluation_lock.json',dict(fitted_manifest_sha256=sha(a.models/'fitted_sources.json'),class_shap_sha256=sha(a.rq1/'class_shap.json'),
        split_sha256=fm['split_sha256'],labels=fm['labels'],budgets=[.25,.5,.75],seed=261006,requested_sets=10,bootstrap=1000,
        families={k:FAMILIES[k] for k in CONTROL_FAMILIES},target='full-input LightGBM argmax; before subset outcomes',
        sampling='legacy sampled_sets; random group-order anchor, exact fact-count matches, ten draws deduplicated; no label or score-based selection',
        primary='both full-input probes correct',natural_mapping_required=False,models_trained=False))
    processes=[];ids=a.gpus.split(',')
    if len(set(ids))!=len(ids):raise ValueError('Duplicate GPU')
    try:
        for i,g in enumerate(ids):
            cmd=[sys.executable,'-u',__file__,'--shard',str(i),'--shards',str(len(ids)),'--threads',str(a.threads)]
            for k in ['models','protocol','sample','rq1','config','out']:cmd+=['--'+k,str(getattr(a,k))]
            log=(a.out/f'worker_{i}.log').open('w',encoding='utf-8')
            processes.append((subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=g),stdout=log,stderr=subprocess.STDOUT),log))
        while any(p.poll() is None for p,_ in processes):
            print('Scoring workers',[(i,p.poll()) for i,(p,_) in enumerate(processes)],flush=True);time.sleep(30)
        if any(p.returncode for p,_ in processes):raise RuntimeError('Worker failed; see worker logs')
    finally:
        for p,log in processes:
            if p.poll() is None:p.terminate();p.wait()
            log.close()
    records=[r for i in range(len(ids)) for r in rows(a.out/f'shard_{i}/scores.jsonl')]
    audit=validate_records(records,expected);dump(a.out/'sampling_audit.json',audit)
    path=a.out/'scores.jsonl'
    with path.open('w',encoding='utf-8') as f:
        for r in sorted(records,key=lambda r:(r['explanation_id'],r['probe'])):emit(f,r)
    tasks=[(str(path),probe,fraction,family) for probe in ['encoder','lightgbm'] for fraction in [.25,.5,.75] for family in CONTROL_FAMILIES]
    panels=[]
    print('CPU summary panels',len(tasks),flush=True)
    with ProcessPoolExecutor(max_workers=a.summary_workers) as pool:
        for i,r in enumerate(pool.map(calculate,tasks),1):
            panels.append(r);dump(a.out/'rq3_summary.json',panels);print('Summary',i,'/',len(tasks),r['probe'],r['fraction'],r['family'],flush=True)
    return dict(status='complete',patients=len(expected),unique_sets=audit['unique_sets'],score_rows=len(records),panels=len(panels),api_calls=0,models_trained=False,
        scope='RQ3 controlled subsets only; fixed-diagnosis citation additions/deletions and natural mapping audit pending')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key,default in [('models','artifacts/code10k_sources_v1'),('protocol','artifacts/code10k_protocol_v1'),('rq1','artifacts/code10k_rq1_v1_retry1'),('out','artifacts/code10k_rq3_v1'),('sample','inputs/ddxplus_sample'),('config',str(ROOT/'configs/code10k_rq2_20261007.json'))]:p.add_argument('--'+key,type=Path,default=Path(default))
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
