"""TRACE RQ1/RQ3 adapters. Scoring loops copied from code10k RQ1/RQ3.
Changes: frozen TRACE input/model adapters, six sources (no simple voting),
subject-cluster resampling, private record isolation. No predictor fitting.
"""
import argparse,itertools,json,math,os,random,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_trace2400_sources import read,rows,dump,sha,load_inputs
from evaluate_saved_subsets import matched_groups
from run_controlled_sets import sampled_sets
from run_coverage_controls import coverage,permute
from code_eval.tree_subset import checked_contributions
from code_eval.trace2400.rq1_summary import BUDGETS,FAMILIES as RQ1_FAMILIES,summarize
from code_eval.trace2400.rq2_summary import SOURCES,FAMILIES,panel,add_pairwise_families

def emit(f,r):f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def support_scores(a,b):return float(a),float(a)-float(b)

def inputs(a):
    data,labels,enc,contract=load_inputs(a.view)
    fm=read(a.models/'fitted_sources.json')
    if fm['status']!='fitted_development_checked' or fm['dataset']!='TRACE2400-v1' or fm['input_sha256']!=contract['input_sha256']:raise ValueError('Source/view mismatch')
    paths={n:Path(v['path']) for n,v in fm['models'].items()}
    for n,p in paths.items():
        if sha(p)!=fm['models'][n]['sha256']:raise ValueError('Source checkpoint drift: '+n)
    split={r['admission_id']:r for r in rows(a.view/'split_PRIVATE.jsonl')}
    if a.partition=='test':
        p=a.view/'test_records_PRIVATE.jsonl'
        if sha(p)!=read(a.view/'manifest.json')['output_sha256'][p.name]:raise ValueError('Test input drift')
        data['test']=rows(p)
    query={}
    for r in data[a.partition]:
        s=split[r['patient_id']]
        if s['partition']!=a.partition or s['reference_class']!=r['label'] or r['context'] or r['patient_id'] in query:raise ValueError('Query contract failed')
        enc.encode(r['tokens'],{})
        query[r['patient_id']]=dict(r,subject_group=s['subject_group'],component='TRACE2400')
    expected={p for p,s in split.items() if s['partition']==a.partition}
    if set(query)!=expected:raise ValueError('Incomplete query census')
    return data['train'],labels,enc,paths,split,query

def setup(a):
    import torch
    from code_eval.trace2400.bank import TraceBank,RetrievalView
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train,labels,enc,paths,split,query=inputs(a)
    cls=read(a.out/'class_shap.json');lock=read(a.out/'evaluation_lock.json')
    if sha(a.out/'class_shap.json')!=lock['class_shap_sha256'] or cls['labels']!=labels or cls['columns']!=enc.columns:raise ValueError('Class SHAP drift')
    bank=TraceBank(paths,read(a.view/'evidence_dictionary.json'),labels,train,list(query.values()),{p:s['subject_group'] for p,s in split.items()},a.threads)
    return bank,RetrievalView(bank),labels,enc,np.asarray(cls['mean_absolute']),query

def class_shap(a):
    import lightgbm as lgb
    train,labels,enc,paths,split,query=inputs(a)
    model=lgb.Booster(model_file=str(paths['lightgbm']))
    sums=np.zeros((len(labels),len(enc.columns)));counts=np.zeros(len(labels),dtype=int)
    for start in range(0,len(train),16):
        batch=train[start:start+16];values=checked_contributions(model,np.stack([enc.encode(r['tokens'],{}) for r in batch]),len(labels))
        for i,r in enumerate(batch):
            d=labels.index(r['label']);sums[d]+=np.abs(values[i,d,:-1]);counts[d]+=1
        if start%160==0:print('Class SHAP',start+len(batch),'/',len(train),flush=True)
    if (counts==0).any() or counts.sum()!=len(train):raise ValueError('Incomplete class attribution')
    dump(a.out/'class_shap.json',dict(labels=labels,columns=enc.columns,counts=counts.tolist(),mean_absolute=(sums/counts[:,None]).tolist(),train_sha256=sha(a.view/'train_records_PRIVATE.jsonl'),tree_sha256=sha(paths['lightgbm'])))

def worker_rq1(a):
    bank,retrieval,labels,enc,means,query=setup(a)
    gain=bank.tree.feature_importance(importance_type='gain')
    selected=sorted(query)[a.shard::a.shards]
    out=a.out/f'shard_{a.shard}';out.mkdir();started=time.time();state={}
    try:
        with (out/'selections.jsonl').open('w',encoding='utf-8') as dest,(out/'patient_audit.jsonl').open('w',encoding='utf-8') as audit:
            for i,pid in enumerate(selected):
                r=query[pid];groups=enc.groups(r['tokens']);full=set(r['tokens']);gs=sorted(groups)
                minus=[sorted(full-set(groups[g])) for g in gs];inputs=[sorted(full)]+minus
                ps=bank.predict(inputs,r['context']);ps.update(retrieval.predict(inputs,r));orders={};weight_audit={}
                for name,values in ps.items():
                    if all(v is not None for v in values):
                        d=int(values[0].argmax());w={g:abs(float(values[0][d]-values[j+1][d])) for j,g in enumerate(gs)}
                        orders[name]=sorted(gs,key=lambda g:(-w[g],g));weight_audit[name]=dict(target=labels[d],absolute_mass=sum(w.values()),nonzero_groups=sum(v>1e-12 for v in w.values()))
                shap=checked_contributions(bank.tree,enc.encode(r['tokens'],r['context'])[None],len(labels))[0]
                target=int(ps['lightgbm'][0].argmax());cols={g:sorted({enc.token(t)[0] for t in ts}) for g,ts in groups.items()}
                for name,w in {'tree_gain':{g:float(gain[c].sum()) for g,c in cols.items()},'tree_local_abs_shap':{g:float(np.abs(shap[target,c]).sum()) for g,c in cols.items()},'tree_class_abs_shap':{g:float(means[target,c].sum()) for g,c in cols.items()}}.items():
                    orders[name]=sorted(gs,key=lambda g:(-w[g],g))
                both=all(ps[n][0] is not None and labels[int(ps[n][0].argmax())]==r['label'] for n in ['encoder','lightgbm'])
                emit(audit,dict(subject_group=r["subject_group"],patient_id=pid,pattern_id=r['pattern_id'],component=r['component'],reference=r['label'],both_full_correct=both,
                    full_probabilities={n:v[0].tolist() if v[0] is not None else None for n,v in ps.items()},full_predictions={n:labels[int(v[0].argmax())] if v[0] is not None else None for n,v in ps.items()},importance=weight_audit,available_selectors=sorted(orders)))
                tasks=[];queries=set();control_cache={}
                for name in sorted(set(itertools.chain.from_iterable(RQ1_FAMILIES.values()))):
                    for frac in BUDGETS:
                        if name not in orders:
                            tasks.append((name,frac,None,None));continue
                        ts=tuple(sorted(t for g in orders[name][:math.ceil(frac*len(gs))] for t in groups[g]))
                        key=(frac,len(ts))
                        if key not in control_cache:control_cache[key]=[tuple(matched_groups(groups,len(ts),random.Random(f'261013|{pid}|{frac}|{len(ts)}|{j}'))) for j in range(10)]
                        controls=control_cache[key]
                        if any(len(c)!=len(ts) for c in controls):raise ValueError('Random count matching failed')
                        queries.add(ts);queries.update(controls);tasks.append((name,frac,ts,controls))
                keys=sorted(queries);pred=bank.predict([list(q) for q in keys],r['context'],sources=['encoder','lightgbm'])
                lookup={n:dict(zip(keys,v)) for n,v in pred.items()}
                for name,frac,ts,controls in tasks:
                    for probe in ['encoder','lightgbm']:
                        valid=ts is not None and lookup[probe][ts] is not None and all(lookup[probe][c] is not None for c in controls)
                        emit(dest,dict(subject_group=r["subject_group"],patient_id=pid,selector=name,probe=probe,fraction=frac,both_full_correct=both,available=valid,
                            selected_tokens=list(ts) if ts is not None else None,fact_count=len(ts) if ts is not None else None,
                            correct=int(labels[int(lookup[probe][ts].argmax())]==r['label']) if valid else None,
                            random_accuracy=float(np.mean([labels[int(lookup[probe][c].argmax())]==r['label'] for c in controls])) if valid else None,
                            random_correct=[int(labels[int(lookup[probe][c].argmax())]==r['label']) for c in controls] if valid else None,
                            controls=[list(c) for c in controls] if controls is not None else None,
                            unavailable_reason=None if valid else ('source_importance_unavailable' if ts is None else 'probe_unavailable')))
                if (i+1)%10==0:print('Shard',a.shard,'patients',i+1,'/',len(selected),'seconds',round(time.time()-started,1),flush=True)
        state=dict(status='complete',patients=len(selected),patient_ids=selected)
    except Exception:state=dict(status='failed',error=traceback.format_exc());raise
    finally:dump(out/'run_manifest.json',state)

def worker_rq3(a):
    bank,retrieval,labels,enc,means,query=setup(a)
    gain=bank.tree.feature_importance(importance_type='gain')
    chosen=sorted(query)[a.shard::a.shards];folder=a.out/f'shard_{a.shard}';folder.mkdir();state={};started=time.time()
    try:
        with (folder/'scores.jsonl').open('w',encoding='utf-8') as dest:
            for i,pid in enumerate(chosen):
                r=query[pid]
                groups=enc.groups(r['tokens']);full=set(r['tokens']);keys={tuple(sorted(full))};minus={g:tuple(sorted(full-set(ts))) for g,ts in groups.items()};keys.update(minus.values())
                cited={};tasks=[]
                for fraction in [.25,.5,.75]:
                    count,sets=sampled_sets(groups,261013,pid,fraction,10)
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
                            for j in range(10):scores[f'permuted_{j:02d}']=coverage(permute(weights['tree_positive_shap'],f"261013|{pid}|{e['diagnosis']}|{j}"),groups,cs)
                    for probe in ['encoder','lightgbm']:
                        pp=cache[probe][ts] if mapped else None;available=pp is not None and fp[probe] is not None
                        emit(dest,dict(subject_group=r["subject_group"],explanation_id=e['explanation_id'],patient_id=pid,pattern_id=r['pattern_id'],component=r['component'],fraction=e['fraction'],requested_sets=e['requested_sets'],unique_sets=e['unique_sets'],anchor_fact_count=e['anchor_fact_count'],reference_diagnosis=r['label'],diagnosis=e['diagnosis'],diagnosis_correct=e['diagnosis_correct'],target_supported=e['target_supported'],
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
 'paired_cohd_coverage_vs_cohd_absolute','paired_tabr_coverage_vs_group_coverage','paired_tabr_absolute_vs_no_retrieval_absolute']

def register_families():
    add_pairwise_families()
    for baseline in ['no_retrieval_absolute']:
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


def preflight(a):
    """First reproduce archives; independently validate production math against CPU."""
    import copy,torch
    from code_eval.trace2400.bank import TraceBank
    from code_eval.trace2400.math_policy import POLICIES,ARCHIVE_POLICIES
    from diagnose_trace_encoder_parity import compare
    train,labels,enc,paths,split,query=inputs(a)
    dev=rows(a.view/'development_records_PRIVATE.jsonl');ids=[r['patient_id'] for r in dev]
    bank=TraceBank(paths,read(a.view/'evidence_dictionary.json'),labels,train,dev,{p:s['subject_group'] for p,s in split.items()},a.threads)
    report={'status':'running','archive_reproduction':{},'production_vs_cpu':{},'production_batch_vs_single':{},'archive_vs_production_descriptive':{}}
    dump(a.out/'inference_math_policy.json',dict(tf32_by_source=POLICIES,archive_tf32_by_source=ARCHIVE_POLICIES,parameter_precision='float32',parity_atol=2e-5,parity_rtol=1e-5,reason='Archived Encoder uses TF32; formal subset scoring uses strict math verified against CPU; same frozen weights'))
    def array(values):return np.array([p if p is not None else [np.nan]*len(labels) for p in values])
    for name in paths:
        cpu=copy.deepcopy(bank.neural[name][0]).to('cpu').eval() if name in bank.neural else None
        for fraction in [0,.25,.5,.75,1.]:
            key=name+'|'+str(fraction);oldname={'cohd':'association','lightgbm':'tree_mask_augmented'}.get(name,name)
            file=a.first/'development_predictions_PRIVATE'/f'{oldname}_keep{fraction}.npz' if name in ['cohd','lightgbm','encoder'] else a.models/name/'predictions_PRIVATE'/f'keep{fraction}.npz'
            with np.load(file,allow_pickle=False) as z:
                if z['ids'].tolist()!=ids:raise ValueError('Archived admission axis mismatch')
                expected=z['probabilities'].copy()
            sets=[enc.masked(r['tokens'],fraction,f"261013|development|{r['input_group_id']}") for r in dev]
            archive=array(bank.predict(sets,{},[name],ids,archive=True)[name])
            prod=array(bank.predict(sets,{},[name],ids)[name])
            report['archive_reproduction'][key]=compare(archive,expected)
            report['archive_vs_production_descriptive'][key]=compare(prod,expected)
            # Exactly this adapter, its encoding and zero context, compared with a CPU copy.
            if cpu is not None:
                vocab=bank.neural[name][1];valid=[i for i,t in enumerate(sets) if all(v in vocab for v in t)];reference=np.full(expected.shape,np.nan)
                with torch.no_grad():
                    for start in range(0,len(valid),128):
                        ix=valid[start:start+128];x=torch.zeros((len(ix),max(1,max(len(sets[i]) for i in ix))),dtype=torch.long)
                        for j,i in enumerate(ix):
                            if sets[i]:x[j,:len(sets[i])]=torch.tensor([vocab[v] for v in sets[i]],dtype=torch.long)
                        reference[ix]=cpu(x,torch.zeros((len(ix),2))).softmax(1).numpy()
                report['production_vs_cpu'][key]=compare(prod,reference)
            single=array([bank.predict([ts],{},[name],pid)[name][0] for pid,ts in zip(ids[:12],sets[:12])])
            report['production_batch_vs_single'][key]=compare(single,prod[:12])
            dump(a.out/'adapter_parity.json',report)
            print('Parity',key,'archive',report['archive_reproduction'][key],'batch',report['production_batch_vs_single'][key],flush=True)
    gates=['archive_reproduction','production_vs_cpu','production_batch_vs_single']
    failures=[group+':'+key for group in gates for key,value in report[group].items() if not value['passed']]
    report.update(status='failed' if failures else 'passed',failed_checks=failures)
    dump(a.out/'adapter_parity.json',report)
    if failures:raise ValueError('Parity failed; aggregate errors saved in adapter_parity.json: '+', '.join(failures))


def launch(a,phase):
    ids=a.gpus.split(',');folder=a.out/phase;folder.mkdir()
    import shutil
    for name in ['class_shap.json','evaluation_lock.json']:shutil.copyfile(a.out/name,folder/name)
    jobs=[]
    try:
        for i,gpu in enumerate(ids):
            cmd=[sys.executable,'-u',__file__,'--worker',phase,'--shard',str(i),'--shards',str(len(ids)),'--partition',a.partition,'--threads',str(a.threads),'--out',str(folder)]
            for k in ['view','models','first']:cmd+=['--'+k,str(getattr(a,k))]
            log=(folder/f'worker_{i}.log').open('w',encoding='utf8')
            jobs.append((subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT),log))
        while any(p.poll() is None for p,l in jobs):
            print(phase,'workers',[(i,p.poll()) for i,(p,l) in enumerate(jobs)],flush=True);time.sleep(20)
        if any(p.returncode for p,l in jobs):raise RuntimeError(phase+' failed; inspect private worker logs')
    finally:
        for p,l in jobs:
            if p.poll() is None:p.terminate();p.wait()
            l.close()
    return folder

def main(a):
    import torch,lightgbm,shutil
    if not torch.cuda.is_available():raise RuntimeError('Use C550 vendor torch environment')
    if len(set(a.gpus.split(',')))!=len(a.gpus.split(',')):raise ValueError('Duplicate GPU')
    train,labels,enc,paths,split,query=inputs(a)
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_trace_evaluation.py','-v'],cwd=ROOT,check=True)
    cmd=[sys.executable,'-u',__file__,'--worker','parity','--partition',a.partition,'--threads',str(a.threads)]
    for k in ['view','models','first','out']:cmd+=['--'+k,str(getattr(a,k))]
    subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus.split(',')[0]),check=True)
    print('Stage 1: training-only class SHAP',flush=True);class_shap(a)
    dump(a.out/'evaluation_lock.json',dict(dataset='TRACE2400-v1',partition=a.partition,admissions=len(query),subjects=len({r['subject_group'] for r in query.values()}),fitted_manifest_sha256=sha(a.models/'fitted_sources.json'),class_shap_sha256=sha(a.out/'class_shap.json'),query_sha256=sha(a.view/(a.partition+'_records_PRIVATE.jsonl')),budgets=BUDGETS,random_repeats=10,bootstrap=1000,random_set_seed=261013,primary_population='both Encoder and LightGBM full-input predictions match reference; common availability per family',resampling='inferred-subject clusters; admission-weighted point estimates; pairs only within admission',rq1_families=RQ1_FAMILIES,no_training=True,api_calls=0))
    folder=launch(a,'rq1');records=[];audits=[]
    for i in range(len(a.gpus.split(','))):
        records+=rows(folder/f'shard_{i}/selections.jsonl');audits+=rows(folder/f'shard_{i}/patient_audit.jsonl')
    if len(audits)!=len(query) or {r['patient_id'] for r in audits}!=set(query):raise ValueError('RQ1 census mismatch')
    selectors=set(itertools.chain.from_iterable(RQ1_FAMILIES.values()))
    if len(records)!=len(query)*len(selectors)*3*2:raise ValueError('RQ1 incomplete records')
    summary=summarize(records);summary.update(dataset='TRACE2400-v1',partition=a.partition,analysis_unit='admission',resampling_unit='inferred subject')
    dump(a.out/'rq1_summary.json',summary)
    from code_eval.tree_subset import metrics
    all_sources=['cohd','encoder','grasp','lightgbm','tabr','no_retrieval'];performance={}
    common=[r for r in audits if all(r['full_probabilities'][n] is not None for n in all_sources)]
    for name in all_sources:
        ok=[r for r in audits if r['full_probabilities'][name] is not None]
        def metric(rs):return metrics(np.array([r['full_probabilities'][name] for r in rs]),np.array([labels.index(r['reference']) for r in rs])) if rs else None
        performance[name]=dict(available=len(ok),unavailable=len(audits)-len(ok),all_available=metric(ok),common_population=metric(common))
    dump(a.out/'full_input_sources.json',dict(dataset='TRACE2400-v1',partition=a.partition,admissions=len(audits),common_admissions=len(common),results=performance))
    dump(a.out/'population_flow.json',dict(admissions=len(query),subjects=len({r['subject_group'] for r in query.values()}),both_full_correct=sum(r['both_full_correct'] for r in audits),full_probe_unavailable={n:sum(r['full_predictions'][n] is None for r in audits) for n in ['encoder','lightgbm']}))
    folder=launch(a,'rq3');records=[r for i in range(len(a.gpus.split(','))) for r in rows(folder/f'shard_{i}/scores.jsonl')]
    audit=validate_records(records,set(query))
    dump(folder/'sampling_audit_PRIVATE.json',audit)
    dump(a.out/'sampling_audit.json',dict(admissions=audit['patients'],unique_sets=audit['unique_sets'],ar_constant_shift_checked=True,blocks=len(audit['blocks'])))
    path=folder/'scores_PRIVATE.jsonl'
    with path.open('w',encoding='utf8') as f:
        for r in records:emit(f,r)
    tasks=[(str(path),probe,budget,family) for probe in ['encoder','lightgbm'] for budget in BUDGETS for family in CONTROL_FAMILIES+['permutations']]
    panels=[]
    with ProcessPoolExecutor(max_workers=a.summary_workers) as pool:
        for i,r in enumerate(pool.map(calculate,tasks),1):
            r.update(dataset='TRACE2400-v1',partition=a.partition,analysis_unit='admission',resampling_unit='inferred subject');panels.append(r)
            dump(a.out/'rq3_summary.json',panels);print('Summary',i,'/',len(tasks),flush=True)
    (a.out/'RESULT_SCOPE.md').write_text('# TRACE2400-v1 RQ1 and RQ3\n\nFrozen-model evidence selection and controlled subsets. Results belong to '+a.partition+'. Inferred-subject cluster intervals retain admission weighting. Natural-rationale RQ2 and citation interventions require actual rollouts and mapped references and are not included. Private admission-level records stay on the server.\n',encoding='utf8')
    return dict(status='complete',dataset='TRACE2400-v1',partition=a.partition,admissions=len(query),rq3_rows=len(records),rq3_panels=len(panels),models_trained=False,api_calls=0)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name,default in [('view','artifacts/trace2400_view_v2_names'),('models','artifacts/trace2400_extended_sources_v1'),('first','artifacts/trace2400_sources_v1'),('out','artifacts/trace2400_evaluation_v1')]:p.add_argument('--'+name,type=Path,default=Path(default))
    p.add_argument('--partition',choices=['development','test'],default='test');p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--threads',type=int,default=4);p.add_argument('--summary-workers',type=int,default=8)
    p.add_argument('--worker',choices=['rq1','rq3','parity']);p.add_argument('--shard',type=int);p.add_argument('--shards',type=int)
    a=p.parse_args()
    for name in ['view','models','first','out']:setattr(a,name,getattr(a,name).resolve())
    if a.worker:
        {'rq1':worker_rq1,'rq3':worker_rq3,'parity':preflight}[a.worker](a)
    else:
        a.out.mkdir(parents=True,exist_ok=False);state={}
        try:state=main(a)
        except Exception as e:
            state={'status':'failed','exception_type':type(e).__name__,'locations':[{'module':Path(t.filename).name,'line':t.lineno} for t in traceback.extract_tb(e.__traceback__)]};raise
        finally:
            state['code_sha256']={str(f.relative_to(ROOT)):sha(f) for parent in ['scripts','src'] for f in (ROOT/parent).rglob('*.py')}
            dump(a.out/'run_manifest.json',state)
            with tarfile.open(str(a.out)+'_reports.tar.gz','w:gz') as tar:
                for name in ['adapter_parity.json','inference_math_policy.json','class_shap.json','evaluation_lock.json','rq1_summary.json','rq3_summary.json','full_input_sources.json','population_flow.json','sampling_audit.json','run_manifest.json','RESULT_SCOPE.md']:
                    f=a.out/name
                    if f.exists():tar.add(f,arcname=name)
            print('Return report:',str(a.out)+'_reports.tar.gz',flush=True)
