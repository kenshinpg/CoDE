"""Frozen-model frequency/IDF, forward greedy, SIS and group-mask Anchors.

No LLM calls, no refitting. Existing datasets and original scoring functions.
Run from tools/CoDE. Reports contain aggregates; row-level files stay on server.
"""
import argparse, csv, hashlib, importlib, json, os, subprocess, sys, tarfile, time, traceback
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
from extend_rq1_random_controls import setup, read_baseline
from run_trace2400_sources import read, rows, dump, sha
from evaluate_saved_subsets import matched_groups
from run_coverage_controls import coverage
from code_eval.round1_extra_baselines import weights_from_counts, order_weights, greedy_order, first_sis, anchor_mask
from code_eval.proxy_metrics import average_precision


def emit(f, r):
    f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n'); f.flush()


def vendor_parts(value):
    """Read relative manifest paths written on either Windows or Linux."""
    portable = value.replace('\\', '/')
    path = PurePosixPath(portable)
    if not path.parts or path.is_absolute() or '..' in path.parts or ':' in portable:
        raise ValueError('Invalid relative vendor path: '+value)
    return path.parts


def verify_vendor(root):
    entries = read(root/'vendor/round1_baseline_sources.json')
    for item in entries:
        path = root.joinpath('vendor', *vendor_parts(item['path']))
        if not path.is_file():
            raise FileNotFoundError('Missing vendored source: '+str(path))
        if sha(path) != item['sha256']:
            raise ValueError('Vendor content changed: '+str(path))
    print('Verified vendored source files:', len(entries), flush=True)


def args_for(a, ds):
    c = read(a.config)['datasets'][ds]
    out = dict(c, dataset=ds, threads=a.threads, partition='test')
    for k in ['source','models','protocol','sample','config','view','rq2_scores','rq2_ledger']:
        out[k] = Path(out[k]).resolve()
    return SimpleNamespace(**out)


def worker(a):
    x = args_for(a, a.dataset)
    bank, enc, labels, query = setup(x)
    if a.dataset == 'ddx':
        from run_code10k_rq1 import verify
        _, train, _ = verify(x)
    else:
        from run_trace2400_evaluation import inputs
        train,_,_,_,_,_ = inputs(x)
    count = Counter(t for r in train for t in set(r['tokens']))
    _, old = read_baseline(x.source, a.dataset)
    replay = {(r['patient_id'],r['probe']):r for r in old if r['selector']=='tree_local_abs_shap' and r['fraction']==.25}
    ledger = defaultdict(list)
    for e in rows(x.rq2_ledger): ledger[e['patient_id']].append(e)
    original_scores = defaultdict(list)
    for r in rows(x.rq2_scores):original_scores[r['explanation_id']].append(r)
    if set(original_scores)!={e['explanation_id'] for ee in ledger.values() for e in ee}:
        raise ValueError('Natural score/ledger identity mismatch')
    if any(len(v)!=2 or {r['probe'] for r in v}!={'encoder','lightgbm'} for v in original_scores.values()):
        raise ValueError('Missing/duplicate natural probe rows')
    selected = sorted(query)[a.shard::a.shards]
    folder = a.out/a.dataset/f'shard_{a.shard}'; folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with (folder/'selections_PRIVATE.jsonl').open('w',encoding='utf8') as dest, (folder/'natural_PRIVATE.jsonl').open('w',encoding='utf8') as natural:
        for i,pid in enumerate(selected):
            r=query[pid]; groups=enc.groups(r['tokens']); gs=sorted(groups); n=len(gs)
            full=tuple(sorted(r['tokens'])); context=r['context']; cluster=r.get('subject_group',pid)
            cache={}; search_count=0
            def predict(sets):
                nonlocal search_count
                keys=[tuple(sorted(ts)) for ts in sets]; missing=sorted(set(keys)-cache.keys())
                for start in range(0,len(missing),256):
                    part=missing[start:start+256]
                    pp=bank.predict([list(k) for k in part],context,sources=['lightgbm'])['lightgbm']
                    cache.update(zip(part,pp));search_count+=len(part)
                return np.stack([cache[k] for k in keys])
            def tokens(mask):return tuple(sorted(t for j,g in enumerate(gs) if mask[j] for t in groups[g]))
            def distributions(masks):return predict([tokens(mask) for mask in masks])
            fp=bank.predict([list(full)],context,sources=['lightgbm','encoder'])
            target=int(fp['lightgbm'][0].argmax());cache[full]=fp['lightgbm'][0]
            both=all(fp[p][0] is not None and labels[int(fp[p][0].argmax())]==r['label'] for p in ['encoder','lightgbm'])
            for probe in ['encoder','lightgbm']:
                rr=replay[pid,probe]
                if rr['both_full_correct']!=both:raise ValueError('Full-input replay mismatch')
                if rr['available']:
                    pr=bank.predict([rr['selected_tokens']],context,sources=[probe])[probe][0]
                    if pr is None or int(labels[int(pr.argmax())]==r['label'])!=rr['correct']:raise ValueError('Frozen RQ1 replay mismatch')
            w=weights_from_counts(count,len(train),groups)
            orders={name:[gs.index(g) for g in order_weights(wt)] for name,wt in w.items()}
            progress=dict(patient=i+1,total=len(selected),phase='greedy',seconds=round(time.monotonic()-started,1))
            dump(folder/'progress.json',progress)
            score=lambda masks:distributions(masks)[:,target]
            orders['greedy_tree']=greedy_order(score,n,math_ceil(.75*n))
            tasks=[]
            for name,order in orders.items():
                for fraction in [.25,.5,.75]:
                    ids=order[:math_ceil(fraction*n)]
                    ts=tuple(sorted(t for j in ids for t in groups[gs[j]]))
                    tasks.append(dict(method=name,fraction=fraction,tokens=ts,status='complete',details={}))
            dump(folder/'progress.json',dict(progress,phase='SIS'))
            sis=first_sis(score,n,threshold=.9)
            dump(folder/'progress.json',dict(progress,phase='Anchors'))
            seed=int(hashlib.sha256(('261023|'+a.dataset+'|'+pid).encode()).hexdigest()[:8],16)
            anchor=anchor_mask(lambda masks:distributions(masks).argmax(1),n,target,seed,limit=a.anchor_queries)
            for name,item in [('sis_tree',sis),('anchors_group_mask_tree',anchor)]:
                ids=item.get('indices');ts=None if ids is None else tuple(sorted(t for j in ids for t in groups[gs[j]]))
                tasks.append(dict(method=name,fraction=None,tokens=ts,status=item['status'],details=item))
            keys=set(); control_cache={}
            import random
            for t in tasks:
                ts=t['tokens']
                if ts is None:continue
                repeats=50 if t['fraction'] in [None,.25] else 10
                key=(len(ts),repeats)
                if key not in control_cache:
                    control_cache[key]=[tuple(matched_groups(groups,len(ts),random.Random(f'261023|{pid}|{len(ts)}|{j}'))) for j in range(repeats)]
                t['controls']=control_cache[key];keys.add(ts);keys.update(t['controls'])
            keys=sorted(keys);allp=bank.predict([list(k) for k in keys],context,sources=['encoder','lightgbm'])
            look={p:dict(zip(keys,v)) for p,v in allp.items()}
            for t in tasks:
                for probe in ['encoder','lightgbm']:
                    ts=t['tokens']; controls=t.get('controls',[])
                    ok=ts is not None and look[probe][ts] is not None and all(look[probe][c] is not None for c in controls)
                    pred=lambda key:int(labels[int(look[probe][key].argmax())]==r['label'])
                    emit(dest,dict(patient_id=pid,cluster=cluster,method=t['method'],probe=probe,fraction=t['fraction'],
                        both_full_correct=both,available=ok,search_status=t['status'],details=t['details'],
                        target=labels[target],fact_count=len(ts) if ts is not None else None,
                        group_count=sum(set(groups[g])<=set(ts) for g in gs) if ts is not None else None,
                        full_group_count=n,correct=pred(ts) if ok else None,
                        random_accuracy=float(np.mean([pred(c) for c in controls])) if ok else None,
                        random_repeats=len(controls),selected_tokens=ts,unique_tree_queries=search_count))
            for e in ledger[pid]:
                eid=e['explanation_id'];base=original_scores[eid]
                # Exact frozen score record owns mapping availability and cited ids.
                for b in base:
                    mapped=b.get('mapping_available',False)
                    cs=set(b['cited_tokens']) if mapped else None
                    if cs is not None and not cs<=set(full):raise ValueError('Nonpatient frozen cited token')
                    vals={name:coverage(wt,groups,cs) if cs is not None else None for name,wt in w.items()}
                    emit(natural,dict(b,cluster=cluster,scores=dict(b['scores'],**vals)))
            dump(folder/'progress.json',dict(progress,phase='complete_patient',completed=i+1,tree_queries=search_count,seconds=round(time.monotonic()-started,1)))
            print(a.dataset,a.shard,f'{i+1}/{len(selected)} patients',f'{search_count} tree queries',f'{time.monotonic()-started:.1f}s',flush=True)
    dump(folder/'run_manifest.json',dict(status='complete',patients=len(selected)))


def math_ceil(x):
    import math
    return math.ceil(x)


def summarize_panel(job):
    kind,name,rs,methods,seed=job;rng=np.random.default_rng(seed)
    clusters=sorted({r['cluster'] for r in rs});ix=[np.array([i for i,r in enumerate(rs) if r['cluster']==k]) for k in clusters]
    result=dict(kind=kind,name=name,records=len(rs),clusters=len(clusters))
    if not rs:return result
    if kind=='selection':
        y=np.asarray([[r['correct'],r['random_accuracy']] for r in rs],float)
        draws=[]
        for _ in range(2000):
            j=np.concatenate([ix[k] for k in rng.integers(0,len(ix),len(ix))]);draws.append((y[j,0]-y[j,1]).mean())
        return dict(result,accuracy=float(y[:,0].mean()),random_accuracy=float(y[:,1].mean()),gain=float((y[:,0]-y[:,1]).mean()),gain_ci95=np.quantile(draws,[.025,.975]).tolist())
    if kind=='direct_selection':
        y=np.asarray([r['delta'] for r in rs],float);draws=[]
        for _ in range(2000):
            j=np.concatenate([ix[k] for k in rng.integers(0,len(ix),len(ix))]);draws.append(y[j].mean())
        return dict(result,difference=float(y.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
    m,b=methods;y=np.asarray([r.get('reference_failure',r.get('failure')) for r in rs],int)
    vals=np.asarray([[-r['scores'][m],-r['scores'][b]] for r in rs]);point=[average_precision(y,vals[:,k]) for k in [0,1]];draws=[]
    for _ in range(2000):
        j=np.concatenate([ix[k] for k in rng.integers(0,len(ix),len(ix))]);ap=[average_precision(y[j],vals[j,k]) for k in [0,1]]
        if all(v is not None for v in ap):draws.append(ap[0]-ap[1])
    return dict(result,methods=methods,failures=int(y.sum()),ap=point,difference=point[0]-point[1] if all(v is not None for v in point) else None,
                valid_bootstrap=len(draws),requested_bootstrap=2000,ci95=np.quantile(draws,[.025,.975]).tolist() if draws else None)


def summarize(a,ds):
    rs=[];ns=[]
    for folder in sorted((a.out/ds).glob('shard_*')):
        if read(folder/'run_manifest.json')['status']!='complete':raise ValueError('Incomplete shard')
        rs+=rows(folder/'selections_PRIVATE.jsonl');ns+=rows(folder/'natural_PRIVATE.jsonl')
    if len({(r['patient_id'],r['method'],r['fraction'],r['probe']) for r in rs})!=len(rs):raise ValueError('Duplicate selections')
    x=args_for(a,ds);_,old=read_baseline(x.source,ds)
    expected=(1506 if ds=='ddx' else 359)*11*2
    if len(rs)!=expected or len(ns)!=(6024 if ds=='ddx' else 1436)*2:raise ValueError('Incomplete new record census')
    old_index={(r['patient_id'],r['selector'],r['fraction'],r['probe']):r for r in old}
    jobs=[];census=[]
    for method in sorted({r['method'] for r in rs}):
        for probe in ['encoder','lightgbm']:
            for fraction in sorted({r['fraction'] for r in rs if r['method']==method},key=lambda x:-1 if x is None else x):
                pool=[r for r in rs if (r['method'],r['probe'],r['fraction'])==(method,probe,fraction)]
                good=[r for r in pool if r['both_full_correct'] and r['available']]
                name=f'{ds}|{method}|{probe}|{fraction}'
                census.append(dict(panel=name,all_records=len(pool),eligible=sum(r['both_full_correct'] for r in pool),available=len(good),search_status=dict(Counter(r['search_status'] for r in pool))))
                jobs.append(('selection',name,[{k:r[k] for k in ['cluster','correct','random_accuracy']} for r in good],None,261023))
                if fraction is not None:
                    for baseline in ['tree_local_abs_shap','tree_class_abs_shap']:
                        common=[]
                        for r in good:
                            b=old_index[r['patient_id'],baseline,fraction,probe]
                            if b['both_full_correct'] and b['available']:
                                common.append(dict(cluster=r['cluster'],delta=r['correct']-b['correct']))
                        jobs.append(('direct_selection',name+' minus '+baseline,common,None,261023))
    for probe in ['encoder','lightgbm']:
        for stratum in ['correct','wrong','all']:
            eligible=[r for r in ns if r['probe']==probe and r['both_full_correct'] and r['probe_available'] and r['mapping_available'] and r['target_supported'] and (ds!='trace' or r['scan_status']=='COMPLETE') and (stratum=='all' or r['diagnosis_correct']==(stratum=='correct'))]
            for m in ['frequency','idf']:
                for b in ['group_coverage','fact_coverage','tree_positive_shap','cohd_coverage','lightgbm_absolute']:
                    common=[r for r in eligible if all(r['scores'].get(k) is not None for k in [m,b])]
                    compact=[dict(cluster=r['cluster'],reference_failure=r['reference_failure'],scores={k:r['scores'][k] for k in [m,b]}) for r in common]
                    jobs.append(('natural',f'{ds}|{probe}|{stratum}|{m} minus {b}',compact,[m,b],261023))
    result=[]
    with ProcessPoolExecutor(max_workers=a.summary_workers) as pool:
        pending={pool.submit(summarize_panel,j):j[1] for j in jobs}
        for future in as_completed(pending):
            result.append(future.result());print('SUMMARY',ds,len(result),'/',len(jobs),pending[future],flush=True)
    dump(a.out/ds/'summary.json',dict(status='complete',panels=result,census=census,
        inference='Review-requested supplementary analyses; pointwise 2000 cluster-bootstrap intervals, not multiplicity-adjusted. TRACE subject clusters; DDXPlus patient clusters. Tree optimized selectors use Tree full-input predicted diagnosis, never reference labels.',
        anchors='Original AnchorBaseBeam on Bernoulli(.5) intact-group retention masks; this is a specified perturbation-domain adaptation. Query-limited searches are censored, not failed clinical explanations.'))


def run(a):
    for name in ['numpy','torch','lightgbm']:importlib.import_module(name)
    verify_vendor(ROOT)
    lock=dict(config=sha(a.config),anchor_queries=a.anchor_queries,cpu_workers=a.cpu_workers,gpus=a.gpus,files={})
    for ds in ['ddx','trace']:
        x=args_for(a,ds)
        if sha(x.rq2_scores)!=read(a.config)['accepted_rq2_sha256'][ds]:raise ValueError('Accepted RQ2 snapshot changed: '+ds)
        for p in [x.source/'evaluation_lock.json',x.models/'fitted_sources.json',x.rq2_scores,x.rq2_ledger]:lock['files'][str(p)]=sha(p)
    for p in [Path(__file__),ROOT/'src/code_eval/round1_extra_baselines.py']:lock['files'][str(p)]=sha(p)
    if (a.out/'input_lock.json').exists():
        if read(a.out/'input_lock.json')!=lock:raise ValueError('Inputs changed on resume')
    else:dump(a.out/'input_lock.json',lock)
    for ds in ['ddx','trace']:
        folder=a.out/ds;folder.mkdir(exist_ok=True)
        n=a.cpu_workers if ds=='ddx' else len(a.gpus.split(','));procs=[]
        try:
            for shard in range(n):
                target=folder/f'shard_{shard}'
                if (target/'run_manifest.json').exists() and read(target/'run_manifest.json')['status']=='complete':continue
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(a.config),'--out',str(a.out),'--dataset',ds,'--shard',str(shard),'--shards',str(n),'--threads',str(a.threads),'--anchor-queries',str(a.anchor_queries)]
                log=(folder/f'worker_{shard}.log').open('w',encoding='utf8')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus.split(',')[shard%len(a.gpus.split(','))])
                procs.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT),log))
            while any(p.poll() is None for p,_ in procs):
                progress=[]
                for p in folder.glob('shard_*/progress.json'):
                    try:progress.append(dict(shard=p.parent.name,**read(p)))
                    except (json.JSONDecodeError,PermissionError):pass
                print('PROGRESS',ds,json.dumps(progress,ensure_ascii=False),flush=True)
                if any(p.poll() not in [None,0] for p,_ in procs):raise RuntimeError('Worker failed; inspect dataset worker logs')
                time.sleep(15)
            if any(p.returncode for p,_ in procs):raise RuntimeError('Worker failed')
        finally:
            for p,log in procs:
                if p.poll() is None:p.terminate();p.wait()
                log.close()
        summarize(a,ds)
    return dict(status='complete',models_trained=False,api_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/round1_extra_baselines_20261023.json')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/round1_extra_baselines_v1')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--cpu-workers',type=int,default=24)
    p.add_argument('--summary-workers',type=int,default=64);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--anchor-queries',type=int,default=200000);p.add_argument('--shard',type=int);p.add_argument('--shards',type=int)
    p.add_argument('--dataset',choices=['ddx','trace']);a=p.parse_args();a.config=a.config.resolve();a.out=a.out.resolve()
    if a.shard is not None:worker(a)
    else:
        a.out.mkdir(parents=True,exist_ok=True);state={}
        try:state=run(a)
        except Exception:state=dict(status='failed',error=traceback.format_exc());raise
        finally:
            dump(a.out/'run_manifest.json',state)
            archive=Path(str(a.out)+'_reports.tar.gz')
            with tarfile.open(archive,'w:gz') as tar:
                for f in a.out.rglob('*'):
                    if f.is_file() and (f.name in ['summary.json','run_manifest.json','input_lock.json','progress.json'] or f.suffix=='.log'):tar.add(f,arcname=f.relative_to(a.out))
            print('Return report:',archive,flush=True)
