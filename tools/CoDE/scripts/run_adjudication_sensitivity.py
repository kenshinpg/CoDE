"""Three frozen mapping contrasts; parallel scoring and resumable cluster bootstrap."""
import argparse, csv, hashlib, importlib.metadata, json, os, shutil, subprocess, sys, tarfile, time, traceback
from pathlib import Path
from types import SimpleNamespace

# Set before importing numpy/torch, including multiprocessing spawn children.
for _key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key]=os.environ.get(_key,'1') if '--shard' in sys.argv else '1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from code_eval.adjudication_sensitivity import (validate_reference,make_ledger,cook,make_panels,
    contrasts,bootstrap_block,finish_panel,CORE_SCORES)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def emit(p,rr):
    with Path(p).open('w',encoding='utf-8') as f:
        for r in rr:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rr):
    keys=list(dict.fromkeys(k for r in rr for k in r))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def path(p):return Path(p) if Path(p).is_absolute() else ROOT/p
def config(a):
    c=read(a.config)
    for values in c['datasets'].values():
        for k,v in values.items():values[k]=str(path(v))
    return c
def dataset_args(a,c,ds):
    keys=['models','protocol','sample','rq1','config'] if ds=='DDXPlus' else ['models','first','view']
    return SimpleNamespace(**{k:Path(c['datasets'][ds][k]) for k in keys},out=a.out/ds,
        threads=a.threads,shard=a.shard,shards=a.shards,partition='test')

def preflight(a,c,reviews):
    from prepare_adjudication_reference import validate
    lock=read(a.review/'input_lock.json')
    for name,h in lock['files'].items():
        if sha(a.review/name)!=h:raise ValueError('Frozen review input changed: '+name)
    protocol=read(a.review/'analysis_protocol.json')
    if a.seed!=protocol['seed'] or a.bootstrap!=protocol['bootstrap']:raise ValueError('Requested resampling differs from frozen protocol')
    validate(rows(a.review/'original_reference_PRIVATE.jsonl'),reviews,read(a.review/'policy_CONFIRMED_PRIVATE.json'))
    census=validate_reference(reviews)
    if census!={'DDXPlus':{'total':102,'complete':101,'unresolved':1},'TRACE':{'total':100,'complete':89,'unresolved':11}}:raise ValueError('Frozen review census changed')
    file_hash={}
    for name,h in c['worker_code_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen implementation changed: '+name)
        file_hash[name]=h
    for ds,v in c['datasets'].items():
        p=Path(v['rq2_scores']);h=sha(p)
        if h!=c['accepted_sha256'][ds+'_rq2_scores']:raise ValueError('Formal automatic score file changed: '+ds)
        file_hash[str(p)]=h
        file_hash[v['ledger']]=sha(v['ledger'])
        for name in ['models']:
            p=Path(v[name])/'fitted_sources.json';file_hash[str(p)]=sha(p)
        if ds=='TRACE':
            fm=read(Path(v['models'])/'fitted_sources.json')
            for name,h in c['accepted_TRACE_model_sha256'].items():
                if fm['models'][name]['sha256']!=h or sha(fm['models'][name]['path'])!=h:raise ValueError('TRACE checkpoint changed: '+name)
    for name in ['scripts/run_adjudication_sensitivity.py','src/code_eval/adjudication_sensitivity.py',
                 'src/code_eval/round1_followup.py','src/code_eval/proxy_metrics.py','scripts/prepare_adjudication_reference.py']:
        file_hash[name]=sha(ROOT/name)
    return dict(config_sha256=sha(a.config),review_lock=lock,files=file_hash,
        cpu_workers=a.cpu_workers,gpus=a.gpus,threads=a.threads,bootstrap=a.bootstrap,seed=a.seed,
        comparisons='three core AP contrasts; original F; complete adjudication; common paired records'),census

def prepare(a,c,ds,reviews):
    v=c['datasets'][ds];folder=a.out/ds;folder.mkdir(exist_ok=True)
    original={r['explanation_id']:r for r in rows(v['ledger'])}
    if ds=='DDXPlus':
        from run_code10k_rq1 import verify
        verify(dataset_args(a,c,ds))
        patients=rows(Path(v['sample'])/'data/patients_public.jsonl')
        facts={r['patient_id']:{f['fact_id'] for f in r['facts'] if f.get('visible',True) and f['coverage_scope']=='symptoms_antecedents'} for r in patients}
        cls=Path(v['rq1'])/'class_shap.json'
    else:
        from run_trace2400_evaluation import inputs
        inputs(dataset_args(a,c,ds))
        patients=rows(Path(v['view'])/'patients_model_input_PRIVATE.jsonl')
        facts={r['patient_id']:{f['fact_id'] for f in r['facts']} for r in patients}
        cls=Path(v['ledger']).parent/'class_shap.json'
    ledger,lookup=make_ledger([r for r in reviews if r['dataset']==ds],original,facts)
    target=folder/'explanation_ledger.jsonl'
    if target.exists():
        if rows(target)!=ledger or sha(folder/'class_shap.json')!=sha(cls):raise ValueError('Prepared inputs changed on resume')
    else:emit(target,ledger);shutil.copyfile(cls,folder/'class_shap.json')
    lock=dict(explanation_ledger_sha256=sha(target),class_shap_sha256=sha(folder/'class_shap.json'))
    if (folder/'evaluation_lock.json').exists() and read(folder/'evaluation_lock.json')!=lock:raise ValueError('Prepared input lock changed')
    dump(folder/'evaluation_lock.json',lock)
    return lookup

def score_worker(a):
    c=config(a);args=dataset_args(a,c,a.dataset)
    core=['cohd','encoder','lightgbm']
    if a.dataset=='DDXPlus':
        import run_code10k_rq2 as w
        from code_eval.frozen_bank import FrozenBank
        original_predict=FrozenBank.predict
        def predict(self,inputs,context,sources=None):return original_predict(self,inputs,context,core if sources is None else sources)
        FrozenBank.predict=predict
        class EmptyRetrieval:
            def __init__(self,*args,**kwargs):pass
            def predict(self,*args,**kwargs):return {}
        w.RetrievalSources=EmptyRetrieval
    else:
        import run_trace2400_rq2 as w
        from code_eval.trace2400.bank import TraceBank,RetrievalView
        original_predict=TraceBank.predict
        def predict(self,inputs,context,sources=None,patient_id=None,archive=False):
            return original_predict(self,inputs,context,core if sources is None else sources,patient_id,archive)
        TraceBank.predict=predict
        RetrievalView.predict=lambda self,*args,**kwargs:{}
    w.SOURCES=core
    old_emit=w.emit;count=0;started=time.monotonic()
    ledger=rows(args.out/'explanation_ledger.jsonl');selected=set(sorted({r['patient_id'] for r in ledger})[a.shard::a.shards])
    total=2*sum(r['patient_id'] in selected for r in ledger)
    progress=args.out/f'progress_{a.shard}.json'
    dump(progress,dict(status='loading',rows=0,total=total))
    def progress_emit(f,r):
        nonlocal count
        old_emit(f,r);f.flush();count+=1
        if count%4==0 or count==total:
            dump(progress,dict(status='scoring',rows=count,total=total,seconds=round(time.monotonic()-started,1)))
            print(a.dataset,'shard',a.shard,f'{count}/{total} rows',f'{time.monotonic()-started:.1f}s',flush=True)
    w.emit=progress_emit
    w.worker(args)
    if count!=total:raise ValueError('Worker row census mismatch')
    dump(progress,dict(status='complete',rows=count,total=total,seconds=round(time.monotonic()-started,1)))
    shard=args.out/f'shard_{a.shard}'
    dump(shard/'resume_lock.json',dict(scores_sha256=sha(shard/'scores.jsonl'),input_binding_sha256=sha(a.out/'input_binding.json'),rows=count))

def launch(a,ds):
    folder=a.out/ds;gpus=a.gpus.split(',');n=a.cpu_workers if ds=='DDXPlus' else len(gpus);jobs=[];reused=0
    try:
        for i in range(n):
            shard=folder/f'shard_{i}';rl=shard/'resume_lock.json'
            if shard.exists():
                if rl.exists() and read(shard/'run_manifest.json').get('status')=='complete':
                    x=read(rl)
                    if x['input_binding_sha256']!=sha(a.out/'input_binding.json') or x['scores_sha256']!=sha(shard/'scores.jsonl'):raise ValueError('Saved shard identity changed')
                    reused+=1;continue
                archive=folder/'incomplete';archive.mkdir(exist_ok=True)
                shard.rename(archive/(shard.name+'_'+str(time.time_ns())))
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--dataset',ds,'--shard',str(i),'--shards',str(n),
                '--review',str(a.review),'--config',str(a.config),'--out',str(a.out),'--threads',str(a.threads)]
            log=(folder/f'worker_{i}.log').open('w',encoding='utf-8')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if ds=='DDXPlus' else gpus[i],CODE_TRACE_TABR_QUERY_MODE='single')
            for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]=str(a.threads)
            jobs.append((i,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
        print(ds,'scoring: launched',len(jobs),'reused',reused,flush=True);started=time.monotonic()
        while True:
            failed=[i for i,p,l in jobs if p.poll() not in [None,0]]
            if failed:
                for i in failed:print((folder/f'worker_{i}.log').read_text(encoding='utf-8')[-6000:],flush=True)
                raise RuntimeError(ds+' scoring failed: '+str(failed))
            done=sum(p.poll()==0 for i,p,l in jobs)+reused;progress=[]
            for i in range(n):
                f=folder/f'progress_{i}.json'
                progress.append(read(f) if f.exists() else dict(rows=0,total=0,status='starting'))
            print(ds,f'shards {done}/{n}; rows {sum(x["rows"] for x in progress)}; elapsed {time.monotonic()-started:.0f}s',
                [(i,x['status'],x['rows']) for i,x in enumerate(progress)],flush=True)
            if done==n:break
            time.sleep(10)
    finally:
        for i,p,l in jobs:
            if p.poll() is None:p.terminate();p.wait()
            l.close()

def summarize(a,records):
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    import multiprocessing
    panels,flows=make_panels(records);table(a.out/'population_flow.csv',flows)
    common=[dict(panel_id=p['panel_id'],dataset=p['dataset'],audit_group=p['audit_group'],probe=p['probe'],proposal_stratum=p['proposal_stratum'],left=c['left'],right=c['right'],explanation_ids=[x['explanation_id'] for x,y in c['pairs']]) for p in panels for c in p['comparisons']]
    # Freeze intersections before calculating effects. Hash binds checkpoints to scores and policy.
    dump(a.out/'common_population_PRIVATE.json',common)
    binding=dict(records=sha(a.out/'scored_records_PRIVATE.jsonl'),common=digest(common),bootstrap=a.bootstrap,seed=a.seed,code=sha(ROOT/'src/code_eval/adjudication_sensitivity.py'))
    folder=a.out/'bootstrap';folder.mkdir(exist_ok=True)
    if (folder/'binding.json').exists() and read(folder/'binding.json')!=binding:raise ValueError('Bootstrap checkpoint binding changed')
    dump(folder/'binding.json',binding)
    tasks=[];arrays={};done_blocks=0
    for panel in panels:
        pid=panel['panel_id'];keys=sorted({k for c in contrasts(panel) for k in c.cluster});rng=np.random.default_rng(a.seed)
        arrays[pid]=np.full((a.bootstrap,9),np.nan)
        for start in range(0,a.bootstrap,250):
            n=min(250,a.bootstrap-start)
            counts=np.array([np.bincount(rng.integers(0,len(keys),len(keys)),minlength=len(keys)) for _ in range(n)],dtype=np.int32) if keys else np.zeros((n,0),dtype=np.int32)
            dest=folder/f'panel_{pid:02d}_block_{start:05d}.npz';lock=dest.with_suffix('.json')
            if dest.exists() and lock.exists():
                if sha(dest)!=read(lock)['sha256']:raise ValueError('Bootstrap block changed')
                block=np.load(dest)['draws']
                if block.shape!=(n,9):raise ValueError('Bootstrap block shape changed')
                arrays[pid][start:start+n]=block;done_blocks+=1
            elif not keys:
                done_blocks+=1
            else:tasks.append((panel,start,counts,keys))
    total=done_blocks+len(tasks);workers=min(a.summary_workers,len(tasks)) if tasks else 0
    print('Bootstrap:',len(panels),'panels x',a.bootstrap,'paired draws;',workers,'single-thread processes;',done_blocks,'/',total,'blocks reused/empty',flush=True)
    started=time.monotonic()
    if tasks:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(bootstrap_block,t) for t in tasks}
            while futures:
                completed,futures=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
                for future in completed:
                    pid,start,block=future.result();arrays[pid][start:start+len(block)]=block
                    dest=folder/f'panel_{pid:02d}_block_{start:05d}.npz'
                    with dest.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,draws=block)
                    dest.with_suffix('.tmp').replace(dest);dump(dest.with_suffix('.json'),dict(sha256=sha(dest)));done_blocks+=1
                if not completed or done_blocks%10==0 or not futures:print(f'Bootstrap blocks {done_blocks}/{total}; elapsed {time.monotonic()-started:.0f}s',flush=True)
    results=[];trans=[]
    for panel in panels:
        rr,tt=finish_panel(panel,arrays[panel['panel_id']],a.bootstrap,a.seed);results+=rr;trans+=tt
    table(a.out/'three_comparisons_all_strata.csv',results)
    table(a.out/'three_comparisons_PRIMARY.csv',[r for r in results if r['primary']])
    table(a.out/'endpoint_transitions.csv',trans)
    dump(a.out/'statistics.json',dict(results=results,transitions=trans,bootstrap_requested=a.bootstrap,summary_processes=workers))
    print('Summary complete:',len(results),'comparison rows;',sum(r['primary'] for r in results),'primary rows',flush=True)

def mapping_quality(a,reviews):
    output=[]
    for ds in ['DDXPlus','TRACE']:
      for group in ['random_sample','targeted_sample']:
        rr=[r for r in reviews if r['dataset']==ds and r[group]];complete=[r for r in rr if r['status']=='complete']
        tp=extra=miss=0
        for r in complete:
            x,y=set(r['automatic_fact_ids']),set(r['final_fact_ids']);tp+=len(x&y);extra+=len(x-y);miss+=len(y-x)
        output.append(dict(dataset=ds,audit_group=group,reviewed=len(rr),complete_reference=len(complete),unresolved=len(rr)-len(complete),patients=len({r['patient_id'] for r in complete}),true_positive=tp,extra_citations=extra,missed_citations=miss,precision=tp/(tp+extra) if tp+extra else None,recall=tp/(tp+miss) if tp+miss else None,f1=2*tp/(2*tp+extra+miss) if 2*tp+extra+miss else None))
    table(a.out/'mapping_quality_complete_reference.csv',output)

def run(a):
    if a.cpu_workers*a.threads>max(1,len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1):raise ValueError('CPU scoring processes x threads exceeds available cores')
    c=config(a);reviews=rows(a.review/'adjudication_reference_PRIVATE.jsonl')
    print('Stage 1: frozen input/model/protocol preflight',flush=True)
    binding,census=preflight(a,c,reviews)
    bp=a.out/'input_binding.json'
    if bp.exists():
        if not a.resume:raise ValueError('Output already exists; use --resume with unchanged inputs')
        if read(bp)!=binding:raise ValueError('Resume settings or input identity changed')
    elif any(f.name!='run_manifest.json' for f in a.out.iterdir()):raise ValueError('Output is not empty and has no binding')
    dump(bp,binding);dump(a.out/'review_census.json',census)
    import torch,lightgbm
    if not torch.cuda.is_available():raise RuntimeError('TRACE scoring requires the existing working torch GPU environment')
    gpuids=a.gpus.split(',')
    if len(set(gpuids))!=len(gpuids) or any(not x.isdigit() or int(x)>=torch.cuda.device_count() for x in gpuids):raise ValueError('GPU list is not available')
    dump(a.out/'environment.json',dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in ['numpy','torch','lightgbm','scipy']},cpu_count=os.cpu_count(),cpu_workers=a.cpu_workers,threads=a.threads,summary_workers=a.summary_workers,gpus=a.gpus))
    allrecords=[]
    for ds in ['DDXPlus','TRACE']:
        print('Stage 2:',ds,'prepare and score frozen automatic/adjudicated sets',flush=True)
        lookup=prepare(a,c,ds,reviews);launch(a,ds)
        scored=[]
        for f in sorted((a.out/ds).glob('shard_*')):
            if read(f/'run_manifest.json')['status']!='complete':raise ValueError('Incomplete score shard')
            scored+=rows(f/'scores.jsonl')
        old={(r['explanation_id'],r['probe']):r for r in rows(c['datasets'][ds]['rq2_scores'])}
        cooked,check=cook(scored,lookup,old,read(a.out/ds/'class_shap.json')['labels'])
        dump(a.out/ds/'automatic_replay_check.json',check);allrecords+=cooked
        print(ds,'automatic replay passed;',len(cooked),'rows',flush=True)
    emit(a.out/'scored_records_PRIVATE.jsonl',allrecords);mapping_quality(a,reviews)
    print('Stage 3: common-population lock and parallel paired bootstrap',flush=True)
    summarize(a,allrecords)
    (a.out/'README.txt').write_text('Finite adjudication reference sensitivity: original full patient facts and fitted sources are fixed.\n'
        'Random and failure-targeted audit subsets are separate; unresolved explanations are excluded from both mapping versions.\n'
        'PRIMARY reports the three contrasts for correct proposals with the Representation probe. Other strata/probe are secondary.\n'
        'Each comparison uses the intersection of available scores/probe endpoints under both mappings; IDs are in common_population_PRIVATE.json.\n'
        'Updated-endpoint results recompute probe failure after changing E. Fixed-endpoint results hold original failure labels constant.\n'
        'change_updated and change_fixed_endpoint are paired differences between adjudicated and automatic AP contrasts.\n'
        'Cluster units: DDXPlus pattern_id, TRACE subject_group. 5000 draws are retained; undefined AP draws are counted.\n'
        'Simultaneous intervals cover the three contrasts within each dataset/audit stratum/probe/proposal group and each endpoint version/change.\n'
        'No-failure AP is undefined; all-failure AP is 1 and its contrast 0, providing no discrimination evidence.\n'
        'Mapping precision/recall/F1 use complete adjudication references only. Failure-targeted estimates describe their selected audit sample.\n'
        'Private identifiers and reviewed evidence are retained for verification; this archive requires the existing access controls.\n',encoding='utf-8')
    return dict(status='complete',review_census=census,score_rows=len(allrecords),models_trained=False,api_calls=0)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,default=ROOT/'inputs/adjudication_20261022')
    p.add_argument('--config',type=Path,default=ROOT/'configs/adjudication_20261022.json')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/v3_3_adjudication_sensitivity_v1')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--cpu-workers',type=int,default=24)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--summary-workers',type=int,default=96)
    p.add_argument('--bootstrap',type=int,default=5000);p.add_argument('--seed',type=int,default=261022)
    p.add_argument('--resume',action='store_true');p.add_argument('--dataset',choices=['DDXPlus','TRACE'])
    p.add_argument('--shard',type=int);p.add_argument('--shards',type=int)
    a=p.parse_args();a.out=a.out.resolve();a.review=a.review.resolve();a.config=a.config.resolve()
    if a.shard is not None:score_worker(a);return
    if min(a.cpu_workers,a.threads,a.summary_workers,a.bootstrap)<1:p.error('Counts must be positive')
    if a.bootstrap!=5000:p.error('Frozen protocol requires 5000 bootstrap draws')
    ncpu=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    a.summary_workers=min(a.summary_workers,ncpu)
    a.out.mkdir(parents=True,exist_ok=True);state={}
    try:state=run(a)
    except Exception:state=dict(status='failed',error=traceback.format_exc());raise
    finally:
        dump(a.out/'run_manifest.json',state)
        report=a.out.with_name(a.out.name+'_reports.tar.gz')
        with tarfile.open(report,'w:gz') as archive:
            for f in sorted(a.out.rglob('*')):
                if f.is_file() and f.suffix!='.tmp':archive.add(f,arcname=f.relative_to(a.out))
        print('Return report:',report,flush=True)
if __name__=='__main__':main()
