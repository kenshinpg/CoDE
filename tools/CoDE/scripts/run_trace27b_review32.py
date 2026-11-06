"""Frozen-model replay of the 27 resolved R02 spot-checks. No API or training."""
import os
for _k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[_k] = '1'
import argparse
import concurrent.futures as cf
import copy
import importlib.metadata
import multiprocessing
from pathlib import Path
import sys
import tarfile
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from trace_generator_rq2.common import read, rows, dump, emit, table, sha, digest, locked, index, cpu_count, METHODS, COMPARISONS, LABELS
from code_eval.adjudication_sensitivity import automatic_gate
from code_eval.round1_followup import APContrast, resample, intervals
from code_eval.proxy_metrics import average_precision


def validate_reviews(reviews, excluded):
    if len(reviews)!=27 or len(excluded)!=5:
        raise ValueError('Expected exactly 27 resolved and 5 excluded records')
    all_rows=reviews+excluded
    for key in ['case','task_id','patient_id']:
        index(all_rows,key)
    if any(not r['fully_resolved'] for r in reviews) or any(r['fully_resolved'] for r in excluded):
        raise ValueError('Unresolved record entered resolved set')
    for r in reviews:
        for key in ['auto_fact_ids','reviewed_fact_ids']:
            if len(r[key])!=len(set(r[key])):raise ValueError('Duplicate fact ID')


def make_jobs(reviews, ledger, patients, records, baseline):
    jobs=[]; lookup={}
    for r in reviews:
        pid=r['patient_id']; eid=r['task_id']; e=ledger[eid]
        if e['patient_id']!=pid or not e['reviewed_available'] or e['repair_status']!='COMPLETE':
            raise ValueError('Review identity or original mapping state differs')
        if set(e['cited_fact_ids'])!=set(r['auto_fact_ids']):raise ValueError('Automatic fact set changed')
        facts={f['fact_id'] for f in patients[pid]['facts']}
        if not set(r['reviewed_fact_ids'])<=facts:raise ValueError('Reviewed fact outside frozen F')
        es=[]
        for version,key in [('automatic','auto_fact_ids'),('reviewed','reviewed_fact_ids')]:
            new_id=eid+'::R02review32::'+version
            es.append(dict(e,explanation_id=new_id,cited_fact_ids=r[key],generator='Qwen3.8-27B'))
            lookup[new_id]=(r,version)
        for probe in ['encoder','lightgbm']:
            old=baseline[eid,probe]
            if old['patient_id']!=pid or old['generator']!='Qwen3.8-27B' or old['subject_group']!=e['subject_group']:
                raise ValueError('Accepted score cache identity mismatch')
        jobs.append((patients[pid],records[pid],es))
    return jobs,lookup


def prepare(a):
    print('Preflight: frozen files, review sets, original score cache and dependencies',flush=True)
    import torch, lightgbm
    manifest=read(a.review/'manifest.json')
    for name,expected in manifest['files'].items():
        if sha(a.review/name)!=expected:raise ValueError('Review input changed: '+name)
    for name,expected in manifest['parent_files'].items():
        if sha(a.parent/name)!=expected:raise ValueError('Accepted parent result changed: '+name)
    parent_binding=read(a.parent/'input_binding.json')
    for name,expected in manifest['code'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen scoring implementation changed: '+name)
    # Original data/model hashes came from the completed parent run, not newly fitted files.
    for name,expected in parent_binding['files'].items():
        path=Path(name)
        if not path.is_absolute():path=ROOT/path
        if sha(path)!=expected:raise ValueError('Original model/data changed: '+str(path))
    c=read(ROOT/'configs/trace_generator_rq2_20261027.json')
    if sha(ROOT/'configs/trace_generator_rq2_20261027.json')!=parent_binding['config']:
        raise ValueError('Parent configuration changed')
    view=ROOT/c['paths']['view']
    fm=read(ROOT/c['paths']['models']/'fitted_sources.json')
    model_paths={}
    for name,expected in c['model_sha256'].items():
        p=Path(fm['models'][name]['path'])
        if not p.is_absolute():p=ROOT/p
        if sha(p)!=expected:raise ValueError('Frozen model changed: '+name)
        model_paths[name]=str(p)
    spec=dict(dictionary=str(view/'evidence_dictionary.json'),models=model_paths)
    reviews=read(a.review/'resolved27_PRIVATE.json'); excluded=read(a.review/'excluded5_PRIVATE.json')
    validate_reviews(reviews,excluded)
    patients=index(rows(view/'patients_model_input_PRIVATE.jsonl'),'patient_id')
    records=index(rows(view/'test_records_PRIVATE.jsonl'),'patient_id')
    ledger=index(rows(a.parent/'explanation_ledger_PRIVATE.jsonl'),'explanation_id')
    original=rows(a.parent/'scores_PRIVATE.jsonl')
    baseline={(r['explanation_id'],r['probe']):r for r in original}
    if len(baseline)!=len(original):raise ValueError('Duplicate baseline row')
    jobs,lookup=make_jobs(reviews,ledger,patients,records,baseline)
    binding=dict(manifest=sha(a.review/'manifest.json'),review=digest(reviews),
        original_ledger=digest([ledger[r['task_id']] for r in reviews]),
        parent=sha(a.parent/'input_binding.json'),script=sha(Path(__file__)),
        bootstrap=a.bootstrap,seed=261016,block_size=250,score_tolerance=2e-5)
    locked(a.out/'binding.json',binding)
    dump(a.out/'protocol.json',read(a.review/'protocol.json'))
    dump(a.out/'excluded5_PRIVATE.json',excluded)
    dump(a.out/'environment.json',dict(python=sys.version,cpu_available=cpu_count(),
        score_processes=min(a.cpu_workers,len(jobs),cpu_count()),
        summary_process_limit=min(a.summary_workers,cpu_count()),threads_per_process=1,
        packages={n:importlib.metadata.version(n) for n in ['numpy','torch','lightgbm']}))
    emit(a.out/'replay_ledger_PRIVATE.jsonl',[e for _,_,es in jobs for e in es])
    dump(a.out/'selected_records_PRIVATE.json',reviews)
    return jobs,lookup,baseline,spec,binding


def cook(scored,lookup,baseline):
    out=[]; seen=set(); maximum=0.; prob_maximum=0.
    expected={(eid,p) for eid in lookup for p in ['encoder','lightgbm']}
    for r in scored:
        key=r['explanation_id'],r['probe']
        if key not in expected or key in seen:raise ValueError('Unexpected/duplicate scored row')
        seen.add(key); review,version=lookup[key[0]]; eid=review['task_id']
        if r['patient_id']!=review['patient_id']:raise ValueError('Patient changed')
        old=baseline[eid,r['probe']]
        if r['subject_group']!=old['subject_group']:raise ValueError('Subject cluster changed')
        if version=='automatic':
            maximum=max(maximum,automatic_gate(r,old,tolerance=2e-5))
            for name in ['full_probabilities','cited_probabilities']:
                if r[name] is not None:
                    prob_maximum=max(prob_maximum,float(np.max(np.abs(np.array(r[name])-old[name]))))
        pred=LABELS[int(np.argmax(r['cited_probabilities']))] if r['probe_available'] else None
        out.append(dict(r,explanation_id=eid,audit_id=review['case'],version=version,prediction=pred))
    if seen!=expected:raise ValueError('Incomplete scoring')
    by={(r['explanation_id'],r['probe'],r['version']):r for r in out}
    for r in out:
        if r['version']!='reviewed':continue
        auto=by[r['explanation_id'],r['probe'],'automatic']
        for key in ['diagnosis','reference_diagnosis','both_full_correct','full_correct','diagnosis_correct','target_supported','full_probabilities']:
            if r[key]!=auto[key]:raise ValueError('Frozen F or target changed: '+key)
    return out,dict(status='passed',automatic_rows=len(out)//2,endpoint_changes=0,
        maximum_score_difference=maximum,maximum_probability_difference=prob_maximum,tolerance=2e-5,
        note='Paired analyses use the replayed automatic and reviewed scores; automatic scores passed the original cache gate.')


def eligible(r,stratum):
    return (r['mapping_available'] and r['scan_status']=='COMPLETE' and r['target_supported']
        and r['probe_available'] and r['both_full_correct']
        and (stratum=='all' or r['diagnosis_correct']==(stratum=='correct'))
        and all(r['scores'].get(m) is not None and np.isfinite(r['scores'][m]) for m in METHODS))


def make_panels(scored):
    by={(r['explanation_id'],r['probe'],r['version']):r for r in scored}; panels=[]
    for probe in ['encoder','lightgbm']:
      for stratum in ['correct','incorrect','all']:
        autos=sorted([r for r in scored if r['probe']==probe and r['version']=='automatic'],key=lambda r:r['explanation_id'])
        pairs=[(r,by[r['explanation_id'],probe,'reviewed']) for r in autos]
        common=[(a,b) for a,b in pairs if eligible(a,stratum) and eligible(b,stratum)]
        arows=[a for a,b in common]; brows=[b for a,b in common]
        fixed=[dict(b,reference_failure=a['reference_failure']) for a,b in common]
        panels.append(dict(id=probe+'_'+stratum,probe=probe,stratum=stratum,
            primary=probe=='encoder' and stratum=='correct',
            reviewed_records=len(pairs),automatic_eligible=sum(eligible(a,stratum) for a,b in pairs),
            reviewed_eligible=sum(eligible(b,stratum) for a,b in pairs),
            common_records=len(common),subject_clusters=len({a['subject_group'] for a,b in common}),
            automatic_failures=sum(a['reference_failure'] for a,b in common),
            reviewed_failures=sum(b['reference_failure'] for a,b in common),
            failures_to_success=sum(a['reference_failure'] and not b['reference_failure'] for a,b in common),
            success_to_failure=sum(not a['reference_failure'] and b['reference_failure'] for a,b in common),
            ids=[a['explanation_id'] for a,b in common],
            versions=dict(automatic=arows,reviewed_fixed_endpoint=fixed,reviewed_updated_endpoint=brows)))
    return panels


def contrasts_for(panel):
    return [APContrast(rs,l,r,'subject_group') for rs in panel['versions'].values() for l,r in COMPARISONS]


def bootstrap_block(job):
    panel,start,count=job
    # Independent deterministic blocks; process count and resume order do not change samples.
    seed=np.random.SeedSequence([261016,panel['seed_index'],start])
    return panel['id'],start,resample(contrasts_for(panel),count,seed)


def summarize(scored,a):
    panels=make_panels(scored)
    for i,p in enumerate(panels):p['seed_index']=i
    dump(a.out/'common_populations_PRIVATE.json',[{k:v for k,v in p.items() if k!='versions'} for p in panels])
    folder=a.out/'bootstrap_blocks'; folder.mkdir(exist_ok=True)
    locked(folder/'binding.json',dict(scores=digest(scored),bootstrap=a.bootstrap,seed=261016,block_size=250))
    tasks=[]; completed={}
    for p in panels:
      for start in range(0,a.bootstrap,250):
        n=min(250,a.bootstrap-start); f=folder/(p['id']+'_'+str(start)+'.npz')
        if f.exists() and f.with_suffix('.lock').exists():
            if sha(f)!=read(f.with_suffix('.lock'))['sha256']:raise ValueError('Bootstrap checkpoint changed')
            with np.load(f,allow_pickle=False) as z:arr=z['draws']
            if arr.shape!=(n,9):raise ValueError('Bootstrap checkpoint shape mismatch')
            completed[p['id'],start]=arr
        else:tasks.append((p,start,n))
    total=len(completed)+len(tasks); workers=min(a.summary_workers,cpu_count(),len(tasks))
    print(f'Bootstrap: {len(completed)}/{total} blocks cached; {workers} processes, 250 draws/block',flush=True)
    if tasks:
      with cf.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={pool.submit(bootstrap_block,t) for t in tasks}; last=0.; started=time.monotonic()
        while pending:
            done,pending=cf.wait(pending,timeout=20,return_when=cf.FIRST_COMPLETED)
            for fut in done:
                pid,start,arr=fut.result(); f=folder/(pid+'_'+str(start)+'.npz')
                tmp=f.with_suffix('.tmp')
                with tmp.open('wb') as stream:np.savez_compressed(stream,draws=arr)
                os.replace(tmp,f); dump(f.with_suffix('.lock'),dict(sha256=sha(f)))
                completed[pid,start]=arr
            if time.monotonic()-last>=20 or not pending:
                state=dict(stage='bootstrap',blocks_done=len(completed),blocks_total=total,
                    processes=workers,elapsed_seconds=round(time.monotonic()-started,1))
                dump(a.out/'progress.json',state); print(state,flush=True); last=time.monotonic()
    result=[]; shifts=[]
    for p in panels:
        draws=np.concatenate([completed[p['id'],s] for s in range(0,a.bootstrap,250)])
        contrasts=contrasts_for(p)
        meta={k:p[k] for k in ['probe','stratum','primary','common_records','subject_clusters','automatic_failures','reviewed_failures','failures_to_success','success_to_failure']}
        for vi,version in enumerate(p['versions']):
            stats=intervals(contrasts[vi*3:vi*3+3],draws[:,vi*3:vi*3+3])
            for (left,right),v in zip(COMPARISONS,stats):
                rs=p['versions'][version]
                y=[r['reference_failure'] for r in rs]
                result.append(dict(meta,version=version,left=left,right=right,
                    ap_left=average_precision(y,[-r['scores'][left] for r in rs]),
                    ap_right=average_precision(y,[-r['scores'][right] for r in rs]),**v))
        for vi,version in [(1,'reviewed_fixed_endpoint'),(2,'reviewed_updated_endpoint')]:
          for j,(left,right) in enumerate(COMPARISONS):
            change=draws[:,vi*3+j]-draws[:,j]; finite=change[np.isfinite(change)]
            point=contrasts[vi*3+j].evaluate(np.ones(p['common_records']))-contrasts[j].evaluate(np.ones(p['common_records']))
            shifts.append(dict(meta,version=version,left=left,right=right,
                difference_change=float(point) if np.isfinite(point) else None,
                ci95_percentile=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,
                valid_replicates=len(finite),undefined_replicates=a.bootstrap-len(finite),
                interval_scope='descriptive paired mapping change; conditional on defined AP replicates'))
    table(a.out/'three_comparisons_ALL.csv',result)
    table(a.out/'three_comparisons_PRIMARY.csv',[r for r in result if r['primary']])
    table(a.out/'paired_contrast_changes.csv',shifts)
    dump(a.out/'statistics.json',dict(panels=[{k:v for k,v in p.items() if k!='versions'} for p in panels],comparisons=result,changes=shifts))
    return panels


def score_changes(scored):
    by={(r['explanation_id'],r['probe'],r['version']):r for r in scored}; out=[]
    for a in scored:
        if a['version']!='automatic':continue
        b=by[a['explanation_id'],a['probe'],'reviewed']
        common=dict(audit_id=a['audit_id'],explanation_id=a['explanation_id'],patient_id=a['patient_id'],
            probe=a['probe'],diagnosis_correct=a['diagnosis_correct'],both_full_correct=a['both_full_correct'],
            auto_prediction=a['prediction'],review_prediction=b['prediction'],
            prediction_changed=a['prediction']!=b['prediction'],auto_failure=a['reference_failure'],review_failure=b['reference_failure'],
            auto_cited_facts=a['cited_fact_count'],review_cited_facts=b['cited_fact_count'])
        for m in ['fact_coverage']+METHODS:
            x,y=a['scores'].get(m),b['scores'].get(m)
            out.append(dict(common,method=m,automatic=x,reviewed=y,change=y-x if x is not None and y is not None else None))
    return out


def run(a):
    from trace_generator_rq2.scoring import parallel_score
    jobs,lookup,baseline,spec,binding=prepare(a)
    dump(a.out/'progress.json',dict(stage='scoring',patients=len(jobs),timestamp=time.time()))
    scored=parallel_score(jobs,spec,a.out/'replay_cache',min(a.cpu_workers,cpu_count(),len(jobs)),binding,'R02 27-patient replay')
    cooked,gate=cook(scored,lookup,baseline)
    dump(a.out/'automatic_replay_gate.json',gate); emit(a.out/'paired_scores_PRIVATE.jsonl',cooked)
    table(a.out/'score_changes_PRIVATE.csv',score_changes(cooked))
    print('Automatic replay gate PASSED',gate,flush=True)
    panels=summarize(cooked,a)
    state=dict(status='complete',resolved_explanations=27,excluded_uncertain=5,score_rows=len(cooked),
        primary={k:v for k,v in panels[0].items() if k!='versions'},
        clinical_validation=False,single_reviewer_spotcheck=True,original_results_overwritten=False)
    dump(a.out/'summary.json',state); dump(a.out/'DONE.json',state)
    dump(a.out/'last_error.json',dict(status='none',timestamp=time.time()))
    print('COMPLETE. Primary common records:',panels[0]['common_records'],
        'failures auto/review:',panels[0]['automatic_failures'],panels[0]['reviewed_failures'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,default=ROOT/'artifacts/trace_qwen27b_rq2_shared_v2')
    p.add_argument('--review',type=Path,default=ROOT/'inputs/trace27b_review32_R02_20261027')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/trace27b_review32_replay_v1')
    p.add_argument('--cpu-workers',type=int,default=27)
    p.add_argument('--summary-workers',type=int,default=128)
    p.add_argument('--bootstrap',type=int,default=5000)
    a=p.parse_args()
    if min(a.cpu_workers,a.summary_workers)<1 or a.bootstrap!=5000:p.error('Positive process limits and bootstrap=5000 required')
    a.out=a.out.resolve(); a.parent=a.parent.resolve(); a.review=a.review.resolve()
    if a.out==a.parent or a.parent in a.out.parents or a.out in a.parent.parents:p.error('Use a separate sibling output directory')
    a.out.mkdir(parents=True,exist_ok=True)
    # Advisory OS lock releases automatically on process exit, including killed sessions.
    import fcntl
    lock=(a.out/'run.lock').open('a+')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('This output is already running; inspect its log instead of launching twice.')
    failed=False
    try:run(a)
    except Exception:
        failed=True; error=traceback.format_exc(); print(error,flush=True)
        dump(a.out/'last_error.json',dict(status='failed',error=error,timestamp=time.time()))
    finally:
        dump(a.out/'run_status.json',dict(status='failed' if failed else 'complete',timestamp=time.time()))
        report=a.out.with_name(a.out.name+'_reports.tar.gz')
        with tarfile.open(report,'w:gz') as tar:
            for f in sorted(a.out.iterdir()):
                if f.is_file() and f.suffix in ['.json','.jsonl','.csv','.txt','.md']:
                    tar.add(f,arcname=a.out.name+'/'+f.name)
        print('Return report:',report,flush=True)
    return 1 if failed else 0


if __name__=='__main__':sys.exit(main())
