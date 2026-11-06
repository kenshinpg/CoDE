"""Gate fixed-query TabR inference, then rescore and summarize in new folders."""
import argparse,os,sys,json,time,subprocess,traceback,tarfile,shutil,copy
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from run_trace2400_sources import read,rows,sha,dump
from run_trace2400_evaluation import setup,inputs
from run_trace2400_rq2 import prepare,worker,summarize
from run_trace2400_review100 import finish
from code_eval.trace2400.review_contract import make_ledger,check_baseline
from code_eval.trace2400.forensic_contract import coverage
from code_eval.trace2400.tabr_gate import TOLERANCE,accept_report,discrepancy

def emit(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rs),encoding='utf-8')
def child_args(a,folder):
    b=copy.copy(a);b.out=folder;return b

def gate(a):
    import numpy as np
    b=child_args(a,a.out/'rq2');bank,retrieval,labels,enc,means,query=setup(b)
    patients={r['patient_id']:r for r in rows(a.view/'patients_model_input_PRIVATE.jsonl')}
    base=rows(b.out/'explanation_ledger.jsonl');reviewed=rows(a.out/'review/explanation_ledger.jsonl')
    selected=sorted(query)[a.shard::a.shards]
    report=dict(tested_patients=0,expected_patients=len(selected),comparisons=0,coverage_comparisons=0,probability_max_delta=0.,coverage_max_delta=0.,availability_changes=0,layouts=['repeat','reverse','split_7','merged_original_review'])
    for i,pid in enumerate(selected,1):
        r=query[pid];token={f['fact_id']:f['feature_id']+'_@_'+f['state'] for f in patients[pid]['facts']};full=tuple(sorted(r['tokens']));groups=enc.groups(r['tokens']);minus={g:tuple(sorted(set(full)-set(ts))) for g,ts in groups.items()}
        def extract(records):
            return {e['explanation_id']:(e,tuple(sorted({token[f] for f in e['cited_fact_ids']}))) for e in records if e['patient_id']==pid and e['reviewed_available']}
        oe=extract(base);re=extract(reviewed);old=sorted({full,*minus.values(),*(ts for e,ts in oe.values())});new=sorted({full,*minus.values(),*(ts for e,ts in re.values())});merged=sorted(set(old)|set(new))
        def predict(keys):return dict(zip(keys,bank.predict([list(k) for k in keys],{},['tabr'],pid)['tabr']))
        reference=predict(merged)
        configs=[predict(merged),predict(list(reversed(merged))),predict(old),predict(new)]
        split={}
        for offset in range(0,len(merged),7):split.update(predict(merged[offset:offset+7]))
        configs.append(split)
        for actual in configs:
            for k,v in actual.items():
                report['comparisons']+=1
                if (v is None)!=(reference[k] is None):report['availability_changes']+=1
                elif v is not None:report['probability_max_delta']=max(report['probability_max_delta'],float(np.max(np.abs(v-reference[k]))))
            for e,ts in [*oe.values(),*re.values()]:
                if ts not in actual or not e['target_supported']:continue
                d=labels.index(e['diagnosis'])
                def score(cache):
                    if cache[full] is None or any(cache[v] is None for v in minus.values()):return None
                    weights={g:max(0.,float(cache[full][d]-cache[v][d])) for g,v in minus.items()}
                    return coverage(weights,groups,set(ts))
                x,y=score(actual),score(reference);report['coverage_comparisons']+=1
                if (x is None)!=(y is None):report['availability_changes']+=1
                else:report['coverage_max_delta']=max(report['coverage_max_delta'],discrepancy(x,y))
        report['tested_patients']+=1
        if i%5==0:print('Gate shard',a.shard,i,'/',len(selected),'max',report['coverage_max_delta'],flush=True)
    dest=a.out/'gate';dest.mkdir(exist_ok=True)
    dump(dest/f'shard_{a.shard}.json',report)
    accept_report(report)

def launch(a,mode):
    ids=a.gpus.split(',');jobs=[]
    try:
        for i,g in enumerate(ids):
            cmd=[sys.executable,'-u',__file__,'--mode',mode,'--shard',str(i),'--shards',str(len(ids)),'--threads',str(a.threads)]
            for k in ['view','models','first','rq1','match','release','review','old_rq2','out']:cmd+=['--'+k.replace('_','-'),str(getattr(a,k))]
            log=(a.out/f'{mode}_worker_{i}.log').open('w',encoding='utf-8')
            jobs.append((subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=g,CODE_TRACE_TABR_QUERY_MODE='single'),stdout=log,stderr=subprocess.STDOUT),log))
        while any(p.poll() is None for p,l in jobs):
            print(mode,[(i,p.poll()) for i,(p,l) in enumerate(jobs)],flush=True);time.sleep(15)
        if any(p.returncode for p,l in jobs):raise RuntimeError(mode+' failed; summaries remain blocked')
    finally:
        for p,l in jobs:
            if p.poll() is None:p.terminate();p.wait()
            l.close()

def collect(folder,ledger):
    rs=[]
    for f in sorted(folder.glob('shard_*')):
        if read(f/'run_manifest.json')['status']!='complete':raise ValueError('Incomplete shard')
        rs+=rows(f/'scores.jsonl')
    expected={(r['explanation_id'],p) for r in ledger for p in ['encoder','lightgbm']}
    if len(rs)!=len(expected) or {(r['explanation_id'],r['probe']) for r in rs}!=expected:raise ValueError('Missing/duplicate scores')
    return rs

def run(a):
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_trace_tabr_gate.py','-v'],check=True,cwd=ROOT)
    if read(a.old_rq2/'run_manifest.json')['status']!='complete':raise ValueError('Old RQ2 incomplete')
    oldlock=read(a.old_rq2/'evaluation_lock.json')
    if sha(a.old_rq2/'explanation_ledger.jsonl')!=oldlock['explanation_ledger_sha256']:raise ValueError('Old ledger changed')
    if sha(a.old_rq2/'class_shap.json')!=oldlock['class_shap_sha256']:raise ValueError('Old class SHAP changed')
    for p,h in oldlock['source_files'].items():
        if sha(Path(p))!=h:raise ValueError('Old RQ2 provenance drift')
    q=a.out/'rq2';v=a.out/'review';q.mkdir();v.mkdir()
    qb=child_args(a,q);prepare(qb);original=rows(q/'explanation_ledger.jsonl')
    if original!=rows(a.old_rq2/'explanation_ledger.jsonl'):raise ValueError('Original population/ledger changed')
    reviews=rows(a.review/'review_sets_PRIVATE.jsonl')
    if len(reviews)!=100:raise ValueError('Review census')
    ledger,lookup=make_ledger(reviews,{r['explanation_id']:r for r in original});emit(v/'explanation_ledger.jsonl',ledger)
    shutil.copy2(q/'class_shap.json',v/'class_shap.json')
    dump(v/'evaluation_lock.json',dict(class_shap_sha256=sha(v/'class_shap.json'),explanation_ledger_sha256=sha(v/'explanation_ledger.jsonl'),review_sha256=sha(a.review/'review_sets_PRIVATE.jsonl'),query_policy='single'))
    dump(a.out/'repair_protocol.json',dict(policy='single TabR query per forward; unchanged weights/labels/exclusions/score definitions',tolerance=TOLERANCE,old_rq2=str(a.old_rq2),old_scores_sha256=sha(a.old_rq2/'scores_PRIVATE.jsonl'),models_sha256=sha(a.models/'fitted_sources.json'),review_sha256=sha(a.review/'review_sets_PRIVATE.jsonl'),paper_update=False))
    print('Stage 1: all test patients, fixed-query layout invariance',flush=True);launch(a,'gate')
    gates=[read(f) for f in sorted((a.out/'gate').glob('shard_*.json'))]
    expected=len(inputs(a)[-1]);report=dict(tested_patients=sum(r['tested_patients'] for r in gates),expected_patients=expected,comparisons=sum(r['comparisons'] for r in gates),coverage_comparisons=sum(r['coverage_comparisons'] for r in gates),probability_max_delta=max(r['probability_max_delta'] for r in gates),coverage_max_delta=max(r['coverage_max_delta'] for r in gates),availability_changes=sum(r['availability_changes'] for r in gates))
    dump(a.out/'invariance_gate.json',accept_report(report))
    print('Stage 2: rescore full RQ2 and all five review sets; no summaries yet',flush=True)
    launch(a,'rq2');launch(a,'review')
    base=collect(q,original);corrected=collect(v,ledger);old={(r['explanation_id'],r['probe']):r for r in base};delta=0.;count=0
    for r in corrected:
        rev,version=lookup[r['explanation_id']]
        if version=='automatic':delta=max(delta,check_baseline(r,old[rev['explanation_id'],r['probe']]));count+=1
    if count!=200:raise ValueError('Baseline census')
    dump(a.out/'review_baseline_gate.json',dict(status='passed',automatic_rows=count,max_score_delta=delta,tolerance=TOLERANCE,all_methods_checked=True,excluded_methods=[]))
    # Audit numerical repair against original results, without requiring legacy drift to disappear.
    archived={(r['explanation_id'],r['probe']):r for r in rows(a.old_rq2/'scores_PRIVATE.jsonl')};changes=defaultdict(lambda:dict(max_delta=0.,above_original_tolerance=0,availability_changes=0));endpoints=defaultdict(int)
    for r in base:
        o=archived[r['explanation_id'],r['probe']]
        for k in ['reference_failure','proposal_failure','both_full_correct','probe_available']:
            if r[k]!=o[k]:endpoints[k]+=1
        for k,x in r['scores'].items():
            y=o['scores'][k];rec=changes[k]
            if (x is None)!=(y is None):rec['availability_changes']+=1
            elif x is not None:
                d=discrepancy(x,y);rec['max_delta']=max(rec['max_delta'],d);rec['above_original_tolerance']+=int(d>TOLERANCE)
    dump(a.out/'legacy_comparison.json',dict(endpoint_changes=dict(endpoints),methods=dict(changes),interpretation='new uniform inference vs legacy; differences retained for review, not silently substituted'))
    print('Stage 3: gates passed; restore RQ2 and review summaries including TabR coverage',flush=True)
    status=summarize(qb);dump(q/'run_manifest.json',status)
    vb=child_args(a,v);vb.rq2=q
    state=finish(vb,lookup)
    dump(v/'run_manifest.json',state)
    return dict(status='complete_gates_passed',query_policy='single',models_trained=False,api_calls=0,paper_updated=False,rq2=str(q),review=str(v))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);base='inputs/trace_production/'
    for k,d in [('view','artifacts/trace2400_view_v2_names'),('models','artifacts/trace2400_extended_sources_v1'),('first','artifacts/trace2400_sources_v1'),('rq1','artifacts/trace2400_evaluation_v3'),('match',base+'trace_match_full_v1'),('release',base+'trace_release_v1/run_20261015_005327_147'),('review','inputs/trace2400_review100_20261015'),('old-rq2','artifacts/trace2400_rq2_v1'),('out','artifacts/trace2400_tabr_repair_v1')]:p.add_argument('--'+k,type=Path,default=Path(d))
    p.add_argument('--partition',default='test',choices=['test']);p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--threads',type=int,default=4);p.add_argument('--summary-workers',type=int,default=8);p.add_argument('--shard',type=int);p.add_argument('--shards',type=int);p.add_argument('--mode',choices=['gate','rq2','review']);a=p.parse_args()
    for k in ['view','models','first','rq1','match','release','review','old_rq2','out']:setattr(a,k,getattr(a,k).resolve())
    os.environ['CODE_TRACE_TABR_QUERY_MODE']='single'
    if a.mode:
        if a.mode=='gate':gate(a)
        else:worker(child_args(a,a.out/a.mode))
    else:
        if len(set(a.gpus.split(',')))!=len(a.gpus.split(',')):raise ValueError('Duplicate GPUs')
        a.out.mkdir(parents=True,exist_ok=False);state={};started=time.time()
        try:state=run(a)
        except Exception:state=dict(status='failed_no_acceptance',error=traceback.format_exc());raise
        finally:
            state.update(seconds=time.time()-started,code_sha256={str(f.relative_to(ROOT)):sha(f) for folder in ['scripts','src'] for f in (ROOT/folder).rglob('*.py')});dump(a.out/'run_manifest.json',state)
            archive=a.out.with_name(a.out.name+'_reports.tar.gz')
            with tarfile.open(archive,'w:gz') as t:
                for f in a.out.rglob('*'):
                    if f.is_file() and (f.name.endswith('.log') or f.name in ['run_manifest.json','repair_protocol.json','invariance_gate.json','review_baseline_gate.json','legacy_comparison.json','input_audit.json','population_flow.json','rq2_summary.json','rq2_information_audit.json','review_sensitivity.json','baseline_check.json','interpretation.json'] or f.parent.name=='gate'):
                        t.add(f,arcname=f.relative_to(a.out).as_posix())
            print('Return report:',archive,flush=True)
