"""Frozen role sensitivity: all, first-diagnosis relevant, first-diagnosis supporting citations."""
import argparse, csv, hashlib, importlib.metadata, json, os, shutil, subprocess, sys, tarfile, time, traceback
from pathlib import Path
from types import SimpleNamespace
for _key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_key]=os.environ.get(_key,'1') if '--shard' in sys.argv else '1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from run_adjudication_sensitivity import read, rows, sha, digest, dump, emit, table, path, config, dataset_args, score_worker
from code_eval.role_sensitivity import resolve,make_ledger,cook,make_panels,finish_panel,census
from code_eval.adjudication_sensitivity import contrasts,bootstrap_block


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
    common=[dict(panel_id=p['panel_id'],dataset=p['dataset'],audit_group=p['audit_group'],probe=p['probe'],proposal_stratum=p['proposal_stratum'],base_set=p['base_set'],restricted_set=p['restricted_set'],left=c['left'],right=c['right'],explanation_ids=[x['explanation_id'] for x,y in c['pairs']]) for p in panels for c in p['comparisons']]
    # Freeze intersections before calculating effects. Hash binds checkpoints to scores and policy.
    dump(a.out/'common_population_PRIVATE.json',common)
    binding=dict(records=sha(a.out/'scored_records_PRIVATE.jsonl'),common=digest(common),bootstrap=a.bootstrap,seed=a.seed,code={n:sha(ROOT/n) for n in ['src/code_eval/adjudication_sensitivity.py','src/code_eval/role_sensitivity.py','scripts/run_round1_role_sensitivity.py']})
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
            futures={pool.submit(bootstrap_block,t) for t in tasks}; last_log=0.
            while futures:
                completed,futures=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
                for future in completed:
                    pid,start,block=future.result();arrays[pid][start:start+len(block)]=block
                    dest=folder/f'panel_{pid:02d}_block_{start:05d}.npz'
                    with dest.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,draws=block)
                    dest.with_suffix('.tmp').replace(dest);dump(dest.with_suffix('.json'),dict(sha256=sha(dest)));done_blocks+=1
                if time.monotonic()-last_log>=10 or not futures:
                    print(f'Bootstrap blocks {done_blocks}/{total}; elapsed {time.monotonic()-started:.0f}s',flush=True);last_log=time.monotonic()
    results=[];trans=[]
    for panel in panels:
        rr,tt=finish_panel(panel,arrays[panel['panel_id']],a.bootstrap,a.seed);results+=rr;trans+=tt
    table(a.out/'three_comparisons_all_strata.csv',results)
    table(a.out/'three_comparisons_PRIMARY.csv',[r for r in results if r['primary']])
    table(a.out/'endpoint_transitions.csv',trans)
    dump(a.out/'statistics.json',dict(results=results,transitions=trans,bootstrap_requested=a.bootstrap,summary_processes=workers))
    print('Summary complete:',len(results),'comparison rows;',sum(r['primary'] for r in results),'primary rows',flush=True)

def preflight(a,c,reviews):
    lock=read(a.review/'input_lock.json')
    for name,h in lock['files'].items():
        if sha(a.review/name)!=h:raise ValueError('Frozen role input changed: '+name)
    protocol=read(a.review/'analysis_protocol.json')
    if a.seed!=protocol['seed'] or a.bootstrap!=protocol['bootstrap']:raise ValueError('Frozen resampling settings changed')
    counts=census(reviews)
    if [(r['dataset'],r['audit_group'],r['explanations']) for r in counts]!=[('DDXPlus','random',30),('DDXPlus','targeted',20),('TRACE','random',30),('TRACE','targeted',20)]:raise ValueError('Role cohort changed')
    file_hash={}
    for name,h in c['worker_code_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen implementation changed: '+name)
        file_hash[name]=h
    for ds,v in c['datasets'].items():
        print('Checking',ds,'accepted score cache and checkpoints',flush=True)
        h=sha(Path(v['rq2_scores']))
        if h!=c['accepted_sha256'][ds+'_rq2_scores']:raise ValueError('Formal automatic scores changed: '+ds)
        file_hash[v['rq2_scores']]=h;file_hash[v['ledger']]=sha(Path(v['ledger']))
        file_hash[str(Path(v['models'])/'fitted_sources.json')]=sha(Path(v['models'])/'fitted_sources.json')
        if ds=='TRACE':
            fm=read(Path(v['models'])/'fitted_sources.json')
            for name,h in c['accepted_TRACE_model_sha256'].items():
                if fm['models'][name]['sha256']!=h or sha(Path(fm['models'][name]['path']))!=h:raise ValueError('TRACE checkpoint changed: '+name)
    for name in ['scripts/run_round1_role_sensitivity.py','src/code_eval/role_sensitivity.py',
                 'scripts/run_adjudication_sensitivity.py','src/code_eval/adjudication_sensitivity.py']:
        file_hash[name]=sha(ROOT/name)
    return dict(config_sha256=sha(a.config),review_lock=lock,files=file_hash,
                cpu_workers=a.cpu_workers,gpus=a.gpus,threads=a.threads,bootstrap=a.bootstrap,seed=a.seed),counts


def descriptions(a,records,reviews):
    table(a.out/'role_census.csv',census(reviews))
    result=[]
    for ds in ['DDXPlus','TRACE']:
      for group in ['random_sample','targeted_sample']:
       for probe in ['encoder','lightgbm']:
        for stratum in ['correct','incorrect','all']:
         for version in ['all','first','support']:
            rr=[r for r in records if r['dataset']==ds and r[group] and r['probe']==probe and r['version']==version
                and (stratum=='all' or r['diagnosis_correct']==(stratum=='correct'))]
            eligible=[r for r in rr if r['both_full_correct'] and r['mapping_available'] and r['target_supported'] and r['probe_available']]
            result.append(dict(dataset=ds,audit_group=group,probe=probe,proposal_stratum=stratum,citation_set=version,
                reviewed=len(rr),eligible=len(eligible),patients=len({r['patient_id'] for r in eligible}),
                empty_sets=sum(not r['fact_ids'] for r in eligible),citation_count=sum(len(r['fact_ids']) for r in eligible),
                failures=sum(r['reference_failure'] for r in eligible),
                failure_patients=len({r['patient_id'] for r in eligible if r['reference_failure']}),
                score_available={m:sum(r['scores'].get(m) is not None for r in eligible) for m in ['cohd_coverage','tree_positive_shap','group_coverage','lightgbm_absolute']}))
    table(a.out/'set_endpoint_descriptives.csv',result)


def run(a):
    ncpu=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    if a.cpu_workers*a.threads>ncpu:raise ValueError('CPU scoring workers x threads exceeds available cores')
    c=config(a);reviews=resolve(rows(a.review/'role_sets_PRIVATE.jsonl'),read(a.review/'analysis_key_PRIVATE.json'))
    print('Stage 1: role-input and frozen model checks; CPU',ncpu,'scoring',a.cpu_workers,'x',a.threads,'summary',a.summary_workers,flush=True)
    binding,counts=preflight(a,c,reviews);bp=a.out/'input_binding.json'
    if bp.exists():
        if not a.resume:raise ValueError('Use --resume with unchanged inputs')
        if read(bp)!=binding:raise ValueError('Resume input identity/settings changed')
    elif any(f.name!='run_manifest.json' for f in a.out.iterdir()):raise ValueError('Output nonempty without input binding')
    dump(bp,binding);dump(a.out/'review_census.json',counts)
    import torch,lightgbm
    if not torch.cuda.is_available():raise RuntimeError('Use the existing vendor torch GPU environment; do not replace torch')
    gpuids=a.gpus.split(',')
    if len(set(gpuids))!=len(gpuids) or any(not x.isdigit() or int(x)>=torch.cuda.device_count() for x in gpuids):raise ValueError('Unavailable GPU list')
    dump(a.out/'environment.json',dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in ['numpy','torch','lightgbm','scipy']},cpu_count=ncpu,cpu_workers=a.cpu_workers,threads=a.threads,summary_workers=a.summary_workers,gpus=a.gpus))
    from code_eval.adjudication_sensitivity import automatic_gate
    accepted_all={(r['dataset'],r['explanation_id'],r['probe']):r for r in rows(a.review/'accepted_all_scores_PRIVATE.jsonl')}
    allrecords=[]
    for ds in ['DDXPlus','TRACE']:
        print('Stage 2:',ds,'all/first/support scoring + original automatic replay',flush=True)
        lookup=prepare(a,c,ds,reviews);launch(a,ds);scored=[]
        for f in sorted((a.out/ds).glob('shard_*')):
            if read(f/'run_manifest.json')['status']!='complete':raise ValueError('Incomplete score shard')
            scored+=rows(f/'scores.jsonl')
        old={(r['explanation_id'],r['probe']):r for r in rows(c['datasets'][ds]['rq2_scores'])}
        cooked,checks=cook(scored,lookup,old,read(a.out/ds/'class_shap.json')['labels'])
        all_delta=0.;all_count=0
        for r in cooked:
            if r['version']=='all':
                previous=accepted_all[ds,r['explanation_id'],r['probe']]
                if set(r['fact_ids'])!=set(previous['fact_ids']):raise ValueError('All-E reference citations changed')
                all_delta=max(all_delta,automatic_gate(r,previous));all_count+=1
        dump(a.out/ds/'all_reference_replay_check.json',dict(status='passed',rows=all_count,max_core_score_delta=all_delta,tolerance=2e-5))
        dump(a.out/ds/'automatic_replay_check.json',checks);allrecords+=cooked
        print(ds,'original replay passed;',len(cooked),'role/probe records',flush=True)
    emit(a.out/'scored_records_PRIVATE.jsonl',allrecords);descriptions(a,allrecords,reviews)
    print('Stage 3: three-set common populations; parallel cluster bootstrap',flush=True)
    summarize(a,allrecords)
    shutil.copyfile(a.review/'analysis_protocol.json',a.out/'analysis_protocol.json')
    (a.out/'README.txt').write_text(
        'Role sensitivity: all adjudicated citations, first-diagnosis relevant (support union oppose), and support-only.\n'
        'Full patient records, diagnoses, fitted models, importance denominators and evaluation probes remain fixed.\n'
        'Original automatic citations are replayed for parity only. all is the finalized adjudicated E, not the automatic E.\n'
        'PRIMARY: correct proposals with Representation probe. Random and failure-targeted audit strata are separate.\n'
        'Each AP contrast uses available observations common to all three citation sets. Sets with zero citations are scored.\n'
        'restricted_updated recomputes probe failure from the restricted E. restricted_fixed_all_endpoint keeps all-E failure labels.\n'
        'change_updated and change_fixed_all_endpoint compare each restricted-set contrast with the all-E contrast.\n'
        'Endpoint transitions show prediction flips and failure changes on these same observations.\n'
        '5000 paired draws, DDXPlus pattern_id / TRACE subject_group clusters; simultaneous intervals cover three contrasts within each panel/version.\n'
        'Undefined AP bootstrap draws are counted; no-failure AP is undefined; all-failure AP=1 has no discrimination.\n'
        'Role restrictions also change citation counts. This finite review tests role sensitivity, not a count-matched causal benefit.\n'
        'Review subsets provide limited sensitivity evidence, not population or clinical-quality estimates.\n'
        'Private record IDs and citation sets require the existing data access controls.\n',encoding='utf-8')
    return dict(status='complete',review_census=counts,role_score_rows=len(allrecords),scoring_rows_including_replay=800,models_trained=False,api_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,default=ROOT/'inputs/round1_roles_20261023')
    p.add_argument('--config',type=Path,default=ROOT/'configs/round1_roles_20261023.json')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/round1_role_sensitivity_v1')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--cpu-workers',type=int,default=24)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--summary-workers',type=int,default=96)
    p.add_argument('--bootstrap',type=int,default=5000);p.add_argument('--seed',type=int,default=261023)
    p.add_argument('--resume',action='store_true');p.add_argument('--dataset',choices=['DDXPlus','TRACE'])
    p.add_argument('--shard',type=int);p.add_argument('--shards',type=int)
    a=p.parse_args();a.out=a.out.resolve();a.review=a.review.resolve();a.config=a.config.resolve()
    if a.shard is not None:score_worker(a);return
    if min(a.cpu_workers,a.threads,a.summary_workers)<1 or a.bootstrap!=5000:p.error('Positive workers and frozen 5000 draws required')
    ncpu=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    a.summary_workers=min(a.summary_workers,ncpu)
    a.out.mkdir(parents=True,exist_ok=True);state={}
    try:state=run(a)
    except Exception:state=dict(status='failed',error=traceback.format_exc());raise
    finally:
        dump(a.out/'run_manifest.json',state)
        report=a.out.with_name(a.out.name+'_reports.tar.gz')
        print('Packaging private report:',report,flush=True)
        with tarfile.open(report,'w:gz') as archive:
            for f in sorted(a.out.rglob('*')):
                if f.is_file() and f.suffix!='.tmp' and 'bootstrap' not in f.relative_to(a.out).parts and 'incomplete' not in f.relative_to(a.out).parts:
                    archive.add(f,arcname=f.relative_to(a.out))
        print('Return report:',report,flush=True)
if __name__=='__main__':main()

