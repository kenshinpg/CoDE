"""TRACE100 dual-reviewer sensitivity, using unchanged TRACE RQ2 scoring."""
import argparse,json,os,sys,time,shutil,subprocess,tarfile,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_trace2400_rq2 import worker
from run_trace2400_sources import read,rows,dump,sha
from run_reviewed_evidence_sensitivity import summarize
from code_eval.trace2400.review_contract import make_ledger,check_baseline

def prepare(a):
    if read(a.rq2/'run_manifest.json')['status']!='complete':raise ValueError('Incomplete RQ2')
    lock=read(a.rq2/'evaluation_lock.json')
    if sha(a.rq2/'explanation_ledger.jsonl')!=lock['explanation_ledger_sha256']:raise ValueError('RQ2 ledger drift')
    for path,h in lock['source_files'].items():
        if sha(Path(path))!=h:raise ValueError('Upstream drift: '+path)
    if sha(a.rq2/'class_shap.json')!=lock['class_shap_sha256']:raise ValueError('Class SHAP drift')
    rr=rows(a.review/'review_sets_PRIVATE.jsonl')
    if len(rr)!=100 or sum(r['random_sample'] for r in rr)!=60 or sum(r['targeted_sample'] for r in rr)!=40:raise ValueError('Review census')
    ledger,lookup=make_ledger(rr,{r['explanation_id']:r for r in rows(a.rq2/'explanation_ledger.jsonl')})
    (a.out/'explanation_ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledger),encoding='utf8')
    shutil.copy2(a.rq2/'class_shap.json',a.out/'class_shap.json')
    dump(a.out/'evaluation_lock.json',dict(class_shap_sha256=sha(a.out/'class_shap.json'),explanation_ledger_sha256=sha(a.out/'explanation_ledger.jsonl'),review_sha256=sha(a.review/'review_sets_PRIVATE.jsonl'),baseline_sha256=sha(a.rq2/'scores_PRIVATE.jsonl'),policy=read(a.review/'policy.json')))
    return lookup

def finish(a,lookup):
    records=[]
    for f in sorted(a.out.glob('shard_*')):
        if read(f/'run_manifest.json')['status']!='complete':raise ValueError('Incomplete shard')
        records+=rows(f/'scores.jsonl')
    expected={(eid,p) for eid in lookup for p in ['encoder','lightgbm']}
    if len(records)!=1000 or {(r['explanation_id'],r['probe']) for r in records}!=expected:raise ValueError('Scoring census')
    old={(r['explanation_id'],r['probe']):r for r in rows(a.rq2/'scores_PRIVATE.jsonl')}
    labels=read(a.out/'class_shap.json')['labels'];cooked=[];delta=0.;checks=0
    for r in records:
        review,v=lookup[r['explanation_id']];eid=review['explanation_id']
        if v=='automatic':delta=max(delta,check_baseline(r,old[eid,r['probe']]));checks+=1
        probs=r['cited_probabilities'];pred=None
        if r['probe_available']:
            if probs is None or len(probs)!=len(labels) or abs(sum(probs)-1)>1e-5:raise ValueError('Invalid class probabilities')
            pred=labels[max(range(len(probs)),key=probs.__getitem__)]
            if (pred!=r['reference_diagnosis'])!=r['reference_failure']:raise ValueError('Prediction/endpoint mismatch')
        cooked.append(dict(r,explanation_id=eid,audit_id=review['audit_id'],version=v,random_sample=review['random_sample'],targeted_primary_failure=review['targeted_sample'],failure=r['reference_failure'],prediction=pred,fact_ids=review['automatic_fact_ids'] if v=='automatic' else review[v]))
    output={}
    for who in ['R01','R02']:
        versions={'automatic':'automatic',who+'_explicit':'reviewed_strict',who+'_compatible':'reviewed_preserve_ambiguous'}
        selected=[dict(r,version=versions[r['version']]) for r in cooked if r['version'] in versions]
        # Legacy summarizer names are adapted here; compatible includes semantic matches, never uncertain cases.
        output[who]={k.replace('reviewed_strict','explicit').replace('reviewed_preserve_ambiguous','compatible'):v for k,v in summarize(selected).items()}
    dump(a.out/'review_sensitivity.json',output)
    dump(a.out/'baseline_check.json',dict(automatic_rows=checks,max_score_delta=delta,tolerance=2e-5))
    (a.out/'reviewed_scores_PRIVATE.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cooked),encoding='utf8')
    dump(a.out/'interpretation.json',dict(random_scope='60 COMPLETE explanations; not all generated explanations',targeted_scope='40 additional failures; descriptive transitions, no population AP',reviewers='reported separately; no automatic consensus',ap='descriptive subset estimates using common available scores; no significance claim',main_results='original full-test tables unchanged',policies=read(a.review/'policy.json')))
    return dict(status='complete',cases=100,versions=5,score_rows=len(records),models_trained=False,api_calls=0)

def main(a):
    if a.summarize_only:
        saved=read(a.out/'evaluation_lock.json')
        if saved['review_sha256']!=sha(a.review/'review_sets_PRIVATE.jsonl') or saved['baseline_sha256']!=sha(a.rq2/'scores_PRIVATE.jsonl') or saved['explanation_ledger_sha256']!=sha(a.out/'explanation_ledger.jsonl'):raise ValueError('Resume inputs drift')
    lookup=prepare(a);ids=a.gpus.split(',')
    if len(set(ids))!=len(ids):raise ValueError('Repeated GPU')
    if a.summarize_only:return finish(a,lookup)
    jobs=[]
    try:
        for i,g in enumerate(ids):
            cmd=[sys.executable,'-u',__file__,'--shard',str(i),'--shards',str(len(ids)),'--threads',str(a.threads)]
            for k in ['view','models','first','rq2','review','out']:cmd+=['--'+k,str(getattr(a,k))]
            log=(a.out/f'worker_{i}.log').open('w',encoding='utf8')
            jobs.append((subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=g),stdout=log,stderr=subprocess.STDOUT),log))
        while any(p.poll() is None for p,l in jobs):
            print('Scoring workers',[(i,p.poll()) for i,(p,l) in enumerate(jobs)],flush=True);time.sleep(10)
        if any(p.returncode for p,l in jobs):raise RuntimeError('Worker failed; inspect worker logs')
    finally:
        for p,l in jobs:
            if p.poll() is None:p.terminate();p.wait()
            l.close()
    return finish(a,lookup)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k,d in [('view','artifacts/trace2400_view_v2_names'),('models','artifacts/trace2400_extended_sources_v1'),('first','artifacts/trace2400_sources_v1'),('rq2','artifacts/trace2400_rq2_v1'),('review','inputs/trace2400_review100_20261015'),('out','artifacts/trace2400_review100_v1')]:p.add_argument('--'+k,type=Path,default=Path(d))
    p.add_argument('--partition',default='test',choices=['test']);p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--threads',type=int,default=4);p.add_argument('--shard',type=int);p.add_argument('--shards',type=int);p.add_argument('--summarize-only',action='store_true');a=p.parse_args()
    for k in ['view','models','first','rq2','review','out']:setattr(a,k,getattr(a,k).resolve())
    if a.shard is not None:worker(a)
    else:
        a.out.mkdir(exist_ok=a.summarize_only,parents=True);state={};start=time.time()
        try:state=main(a)
        except Exception:state=dict(status='failed',error=traceback.format_exc());raise
        finally:
            state['seconds']=time.time()-start;dump(a.out/'run_manifest.json',state)
            archive=a.out.with_name(a.out.name+'_reports.tar.gz')
            with tarfile.open(archive,'w:gz') as tar:
                for name in ['run_manifest.json','evaluation_lock.json','review_sensitivity.json','baseline_check.json','interpretation.json']:
                    f=a.out/name
                    if f.exists():tar.add(f,arcname=name)
                for f in a.out.glob('worker_*.log'):tar.add(f,arcname=f.name)
            print('Return report:',archive,flush=True)
