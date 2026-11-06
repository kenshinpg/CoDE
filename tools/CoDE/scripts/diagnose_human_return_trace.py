"""Read saved human-return scores; diagnose TRACE replay on affected patients only.

No changes to fitted models, frozen workers, F, thresholds or formal results.
"""
import argparse,os,sys,json,time,traceback,subprocess,tarfile,platform
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from code_eval.human_return import read,rows,sha,dump,emit,table
from run_human_return_sensitivity import configs,verify_files

SOURCES=['tabr','no_retrieval','grasp']
ENDPOINTS=['patient_id','diagnosis','reference_diagnosis','probe_available','both_full_correct','reference_failure','proposal_failure','cited_fact_count']

def verified_saved(source,relative,manifest):
    p=source/relative
    if manifest['output_sha256'].get(relative)!=sha(p):raise ValueError('Saved file changed: '+relative)
    return p

def select_plan(drift,ledger):
    affected=sorted({r['patient_id'] for r in drift['details']})
    reviewed=sorted({r['patient_id'] for r in ledger})
    controls=[p for p in reviewed if p not in affected][:2]
    return dict(affected=affected,controls=controls,patients=sorted(set(affected+controls)),
        anchors=affected[:2],selection='all replay-drift patients plus first two unaffected reviewed IDs; first two affected IDs repeated on each GPU')

def environment(torch):
    p=torch.cuda.get_device_properties(0)
    return dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda_version=torch.version.cuda,
        device=str(p),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),query_mode=os.environ.get('CODE_TRACE_TABR_QUERY_MODE'),
        threads=torch.get_num_threads(),matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        deterministic=torch.are_deterministic_algorithms_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark)

def prepare(a,c):
    binding=verify_files(a,c);saved=read(a.source/'run_manifest.json')
    if read(a.source/'input_binding.json')!=binding:raise ValueError('Input binding changed')
    drift=read(verified_saved(a.source,'TRACE/automatic_score_drift_PRIVATE.json',saved))
    ledger=rows(verified_saved(a.source,'TRACE/explanation_ledger.jsonl',saved))
    for name in ['class_shap.json','evaluation_lock.json']:verified_saved(a.source,'TRACE/'+name,saved)
    original={(r['explanation_id'],r['probe']):r for r in rows(Path(c['datasets']['TRACE']['rq2_scores']))}
    rr=[];shards=sorted((a.source/'TRACE').glob('shard_*'))
    if not shards:raise ValueError('Keep the original score shard directories')
    for folder in shards:
        if read(folder/'run_manifest.json')['status']!='complete':raise ValueError('Incomplete saved shard')
        rr+=rows(verified_saved(a.source,folder.relative_to(a.source).as_posix()+'/scores.jsonl',saved))
    expected={(r['explanation_id'],p) for r in ledger for p in ['encoder','lightgbm']}
    if len(rr)!=len(expected) or {(r['explanation_id'],r['probe']) for r in rr}!=expected:raise ValueError('Saved score census mismatch')
    changes=[]
    for r in rr:
        eid,version=r['explanation_id'].rsplit('::human_return::',1)
        if version!='automatic':continue
        old=original[eid,r['probe']]
        for f in ENDPOINTS:
            if r[f]!=old[f]:changes.append(dict(explanation_id=eid,probe=r['probe'],field=f,original=old[f],replayed=r[f]))
    dump(a.out/'saved_endpoint_audit.json',dict(automatic_rows=sum(r['explanation_id'].endswith('::automatic') for r in rr),changes=changes))
    emit(a.out/'saved_TRACE_scores_PRIVATE.jsonl',rr)
    plan=select_plan(drift,ledger);dump(a.out/'diagnostic_plan_PRIVATE.json',plan)
    dump(a.out/'input_binding.json',binding)
    # Compare the actual source dependencies with the previously accepted run's code inventory.
    historical=Path(c['datasets']['TRACE']['rq2_scores']).parents[1]/'run_manifest.json'
    deps=['src/code_eval/tabr_adapter.py','src/code_eval/tabr_official_core.py','src/code_eval/trace2400/bank.py',
        'src/code_eval/trace2400/retrieval.py','src/code_eval/trace2400/encoding.py','src/code_eval/trace2400/math_policy.py',
        'src/code_eval/context_grasp.py','src/code_eval/context_encoder.py','scripts/run_coverage_controls.py']
    inventory=read(historical).get('code_sha256',{}) if historical.exists() else {}
    report=[]
    for name in deps:
        old=inventory.get(name);now=sha(ROOT/name)
        report.append(dict(file=name,accepted_sha256=old,current_sha256=now,status='missing_historical_identity' if old is None else 'match' if old==now else 'changed'))
    dump(a.out/'dependency_identity.json',dict(historical_manifest=str(historical),files=report))
    if any(r['status']=='changed' for r in report):raise ValueError('Historical model dependency changed; inspect dependency_identity.json before GPU replay')
    if changes:raise ValueError('Saved endpoint drift; inspect saved_endpoint_audit.json before GPU replay')
    return plan

def tabr_neighbors(bank,tokens,pid):
    """Capture real forward encodings; compute its exact retrieval boundary for diagnosis."""
    import numpy as np
    from unittest.mock import patch
    from code_eval.trace2400.math_policy import source_math
    torch=bank.torch;model,bx,by,bp,k=bank.retrieval['tabr'];captured=[];encode=model._encode
    def capture(x):
        value=encode(x);captured.append(value[1].detach());return value
    xx=torch.tensor(np.stack([bank.dense.encode(tokens,{})]),device=bank.device)
    qp=torch.tensor([bank.components[pid]],device=bank.device)
    with source_math(torch,'tabr'),torch.no_grad(),patch.object(model,'_encode',capture):
        prediction=model(xx,bx,by,qp,bp,k,use_retrieval=True).softmax(1).cpu().numpy()[0]
        q,candidate=captured[:2]
        distance=q.square().sum(1,keepdim=True)-2*q@candidate.T+candidate.square().sum(1)[None]
        distance.masked_fill_(qp[:,None]==bp[None,:],float('inf'))
        order=torch.argsort(distance,dim=1,stable=True)[0]
        ds=distance[0,order[:k+1]].cpu().tolist();ix=order[:k].cpu().tolist()
    gap=ds[k]-ds[k-1] if len(ds)>k and all(np.isfinite(ds)) else None
    import hashlib
    return dict(probabilities=prediction.tolist(),indices=ix,cutoff=ds[k-1],next_distance=ds[k] if len(ds)>k and np.isfinite(ds[k]) else None,
        cutoff_gap=gap,candidate_keys_sha256=hashlib.sha256(candidate.cpu().numpy().tobytes()).hexdigest())

def worker(a,c):
    import numpy as np,torch
    from run_trace2400_evaluation import setup
    from run_coverage_controls import coverage
    cfg=c['datasets']['TRACE'];arg=SimpleNamespace(out=a.source/'TRACE',view=Path(cfg['view']),models=Path(cfg['models']),
        first=Path(cfg['first']),partition='test',threads=a.threads)
    bank,_,labels,enc,_,query=setup(arg)
    dump(a.out/f'environment_{a.shard}.json',environment(torch))
    plan=read(a.out/'diagnostic_plan_PRIVATE.json')
    ids=sorted(set(plan['patients'][a.shard::a.shards])|set(plan['anchors']))
    full_ledger=rows(Path(cfg['ledger']));review_ledger=rows(a.source/'TRACE/explanation_ledger.jsonl')
    original={(r['explanation_id'],r['probe']):r for r in rows(Path(cfg['rq2_scores']))}
    patients={r['patient_id']:r for r in rows(Path(cfg['view'])/'patients_model_input_PRIVATE.jsonl')}
    scores=[];probabilities=[];neighbors=[]
    for n,pid in enumerate(ids,1):
        r=query[pid];full=tuple(sorted(r['tokens']));groups=enc.groups(r['tokens'])
        minus={g:tuple(sorted(set(full)-set(ts))) for g,ts in groups.items()}
        facts={f['fact_id']:f['feature_id']+'_@_'+f['state'] for f in patients[pid]['facts']}
        def cited(ledger):
            return {e['explanation_id']:(e,tuple(sorted({facts[f] for f in e['cited_fact_ids']}))) for e in ledger if e['patient_id']==pid and e['reviewed_available']}
        old=cited(full_ledger);new=cited(review_ledger)
        oldkeys=sorted({full,*minus.values(),*(ts for e,ts in old.values())})
        newkeys=sorted({full,*minus.values(),*(ts for e,ts in new.values())})
        layouts=[('current',newkeys),('repeat',newkeys),('reverse',list(reversed(newkeys))),('original',oldkeys)]
        caches={}
        for layout,keys in layouts:
            pred=bank.predict([list(k) for k in keys],{},SOURCES,pid)
            cache={m:dict(zip(keys,pred[m])) for m in SOURCES};caches[layout]=cache
            for m in SOURCES:
                for key,p in cache[m].items():
                    probabilities.append(dict(patient_id=pid,shard=a.shard,layout=layout,source=m,tokens=list(key),probabilities=p.tolist() if p is not None else None))
                for eid,(e,ts) in new.items():
                    if ts not in cache[m] or not e['target_supported']:continue
                    d=labels.index(e['diagnosis']);fp=cache[m][full];cp=cache[m][ts]
                    w={g:max(0.,float(fp[d]-cache[m][v][d])) for g,v in minus.items()}
                    cov=coverage(w,groups,set(ts));support=float(cp[d]);retained=float(cp[d]-fp[d])
                    origid,version=eid.rsplit('::human_return::',1)
                    baseline=original[origid,'encoder']['scores']
                    scores.append(dict(patient_id=pid,explanation_id=origid,version=version,shard=a.shard,layout=layout,source=m,
                        coverage=cov,absolute=support,retained=retained,full_support=float(fp[d]),positive_weight_mass=sum(w.values()),
                        positive_groups=sum(x>0 for x in w.values()),group_weights=w,
                        original_coverage=baseline[m+'_coverage'] if version=='automatic' else None,
                        original_absolute=baseline[m+'_absolute'] if version=='automatic' else None,
                        original_retained=baseline[m+'_retained'] if version=='automatic' else None))
        # For the worst current-vs-original-pool TabR subset (or first sorted if invariant), inspect actual neighbors twice.
        shared=sorted(set(caches['current']['tabr'])&set(caches['original']['tabr']))
        worst=max(shared,key=lambda k:float(np.max(np.abs(caches['current']['tabr'][k]-caches['original']['tabr'][k]))))
        for repeat in range(2):
            neighbors.append(dict(patient_id=pid,shard=a.shard,repeat=repeat,tokens=list(worst),
                cached_current=caches['current']['tabr'][worst].tolist(),**tabr_neighbors(bank,list(worst),pid)))
        print('GPU shard',a.shard,'patient',n,'/',len(ids),'finished',flush=True)
    emit(a.out/f'diagnostic_scores_{a.shard}_PRIVATE.jsonl',scores)
    emit(a.out/f'diagnostic_probabilities_{a.shard}_PRIVATE.jsonl',probabilities)
    emit(a.out/f'diagnostic_neighbors_{a.shard}_PRIVATE.jsonl',neighbors)

def summarize(a):
    import numpy as np
    score=[r for f in sorted(a.out.glob('diagnostic_scores_*_PRIVATE.jsonl')) for r in rows(f)]
    prob=[r for f in sorted(a.out.glob('diagnostic_probabilities_*_PRIVATE.jsonl')) for r in rows(f)]
    lookup={(r['patient_id'],r['shard'],r['source'],tuple(r['tokens']),r['layout']):r for r in prob}
    result=[]
    for m in SOURCES:
        for layout in ['repeat','reverse','original']:
            diffs=[]
            for r in prob:
                if r['source']!=m or r['layout']!=layout:continue
                other=lookup.get((r['patient_id'],r['shard'],m,tuple(r['tokens']),'current'))
                if other:diffs.append(float(np.max(np.abs(np.array(r['probabilities'])-other['probabilities']))))
            result.append(dict(source=m,comparison=layout+'_vs_current',probability_comparisons=len(diffs),max_probability_delta=max(diffs,default=None),above_2e_5=sum(x>2e-5 for x in diffs)))
    grouped=defaultdict(list)
    for r in prob:
        if r['layout']=='current':grouped[r['patient_id'],r['source'],tuple(r['tokens'])].append(r)
    for m in SOURCES:
        diffs=[]
        for (pid,src,tokens),rs in grouped.items():
            if src==m and len(rs)>1:
                diffs.append(float(np.max(np.ptp(np.array([r['probabilities'] for r in rs]),axis=0))))
        result.append(dict(source=m,comparison='cross_GPU_anchor_range',probability_comparisons=len(diffs),max_probability_delta=max(diffs,default=None),above_2e_5=sum(x>2e-5 for x in diffs)))
    table(a.out/'probability_replay_summary.csv',result)
    table(a.out/'coverage_diagnostic_PRIVATE.csv',[{k:v for k,v in r.items() if k!='group_weights'} for r in score])
    return dict(status='diagnostic_complete',model_fit=False,formal_tables_changed=False,score_rows=len(score),probability_rows=len(prob),summary=result)

def main(a):
    c=configs(a.config);plan=prepare(a,c)
    print('Affected patients:',len(plan['affected']),'controls:',len(plan['controls']),'cross-GPU anchors:',len(plan['anchors']),flush=True)
    ids=a.gpus.split(',');jobs=[]
    try:
        for i,g in enumerate(ids):
            log=(a.out/f'worker_{i}.log').open('w',encoding='utf-8')
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--shard',str(i),'--shards',str(len(ids)),'--threads',str(a.threads)]
            for key in ['source','out','review','config']:cmd+=['--'+key,str(getattr(a,key))]
            jobs.append((subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=g,CODE_TRACE_TABR_QUERY_MODE='single'),stdout=log,stderr=subprocess.STDOUT),log))
        while any(p.poll() is None for p,l in jobs):
            print('Diagnostic workers',[(i,p.poll()) for i,(p,l) in enumerate(jobs)],flush=True);time.sleep(10)
        if any(p.returncode for p,l in jobs):raise RuntimeError('Diagnostic worker failed; inspect logs')
    finally:
        for p,l in jobs:
            if p.poll() is None:p.terminate();p.wait()
            l.close()
    return summarize(a)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k,v in [('source','artifacts/human_return_v2'),('out','artifacts/human_return_trace_diagnostic_v1'),('review','inputs/human_review_20261016'),('config','configs/human_return_20261016.json')]:p.add_argument('--'+k,type=Path,default=ROOT/v)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--shard',type=int);p.add_argument('--shards',type=int);a=p.parse_args()
    for k in ['source','out','review','config']:setattr(a,k,getattr(a,k).resolve())
    os.environ['CODE_TRACE_TABR_QUERY_MODE']='single'
    if a.shard is not None:worker(a,configs(a.config))
    else:
        if len(set(a.gpus.split(',')))!=len(a.gpus.split(',')):raise ValueError('Duplicate GPU')
        a.out.mkdir(parents=True,exist_ok=False);state={};start=time.time()
        try:state=main(a)
        except Exception:state=dict(status='failed',error=traceback.format_exc());raise
        finally:
            state.update(seconds=time.time()-start,script_sha256=sha(Path(__file__)))
            dump(a.out/'run_manifest.json',state)
            dest=Path(str(a.out)+'_reports_PRIVATE.tar.gz')
            with tarfile.open(dest,'w:gz') as t:
                for f in a.out.rglob('*'):
                    if f.is_file():t.add(f,arcname=f.relative_to(a.out).as_posix())
            print('Return diagnostic report:',dest,flush=True)
