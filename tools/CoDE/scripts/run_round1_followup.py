"""A1-A3/B1-B2 on frozen DDXPlus/TRACE records. No fitting, LLM calls or paper edits."""
import argparse, csv, hashlib, json, os, sys, tarfile, traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
import numpy as np
from code_eval.round1_followup import availability, absolute_endpoints, family_job, AP_COMPARISONS

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:return [json.loads(s) for s in f if s.strip()]
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def table(p,rs):
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rs:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def resolved(cfg):
    return {k:(str((ROOT/v).resolve()) if not Path(v).is_absolute() else v) for k,v in cfg.items()}

def check_inputs(config,background):
    report={};missing=[]
    for ds,cfg in config['datasets'].items():
        for key,value in cfg.items():
            if key=='context':continue
            if key in ['models','protocol','sample','view']:continue
            if not Path(value).exists():missing.append(ds+': '+key+' = '+value)
        report[ds]={'paths':cfg}
    if missing:raise FileNotFoundError('Missing inputs. Adjust ONLY paths in config; do not substitute older results:\n'+'\n'.join(missing))
    if background:
        import torch,lightgbm
        report['dependencies']={'torch':torch.__version__,'lightgbm':lightgbm.__version__}
    report['numpy']=np.__version__
    return report

def load_scores(path,ds,rq):
    rr=rows(path);seen=set()
    for r in rr:
        key=(r['explanation_id'],r['probe'])
        if key in seen:raise ValueError('Duplicate score key: '+str(key))
        seen.add(key)
        for field in ['patient_id','pattern_id','reference_diagnosis','mapping_available','target_supported',
                      'probe_available','both_full_correct','full_probabilities','cited_probabilities','scores']:
            if field not in r:raise ValueError('Missing score field '+field)
        if ds=='TRACE' and 'subject_group' not in r:raise ValueError('Missing inferred subject group')
        if ds=='TRACE' and rq=='RQ2':r['mapping_available']=r['mapping_available'] and r['scan_status']=='COMPLETE'
        for x in r['scores'].values():
            if x is not None and not np.isfinite(x):raise ValueError('Nonfinite score')
    expected=1506 if ds=='DDX' else 359
    for probe in ['encoder','lightgbm']:
        if len({r['patient_id'] for r in rr if r['probe']==probe})!=expected:raise ValueError('Test census mismatch')
    if rq=='RQ2' and len(rr)!=expected*4*2:raise ValueError('Four-rollout census mismatch')
    return rr

def context_predictions(ds,cfg,score_rows,threads):
    """Use the same frozen Encoder/Tree forward operations; CPU float32, no retraining."""
    import torch,lightgbm as lgb
    from code_eval.context_encoder import ContextEncoder
    torch.set_num_threads(threads)
    if ds=='DDX':
        from run_code10k_rq1 import verify
        from run_code10k_sources import structural_record
        from code_eval.tree_subset import Encoding
        args=SimpleNamespace(models=Path(cfg['models']),protocol=Path(cfg['protocol']),sample=Path(cfg['sample']),config=ROOT/'configs/code10k_rq2_20261007.json')
        fm,_,dictionary=verify(args);enc=Encoding(dictionary);labels=fm['labels']
        split={r['patient_id']:r for r in rows(Path(cfg['protocol'])/'patient_split_PRIVATE.jsonl') if r['partition']=='test'}
        query={r['patient_id']:structural_record(r,split[r['patient_id']]) for r in rows(Path(cfg['sample'])/'data/patients_public.jsonl') if r['patient_id'] in split}
        tree_path=Path(cfg['models'])/'source/tree/mask_augmented.txt';neural_path=Path(cfg['models'])/'encoder/encoder.pt'
    else:
        from run_trace2400_evaluation import inputs
        args=SimpleNamespace(view=Path(cfg['view']),models=Path(cfg['models']),partition='test')
        _,labels,enc,paths,_,query=inputs(args);tree_path=paths['lightgbm'];neural_path=paths['encoder']
    ck=torch.load(neural_path,map_location='cpu',weights_only=True)
    if ck['labels']!=labels:raise ValueError('Checkpoint labels differ')
    model=ContextEncoder(len(ck['vocabulary']),len(labels),ck['hidden']);model.load_state_dict(ck['state']);model.eval()
    tree=lgb.Booster(model_file=str(tree_path));vocab=ck['vocabulary']
    def predict(tokens,context):
        x=torch.zeros((1,max(1,len(tokens))),dtype=torch.long)
        p=None
        if all(t in vocab for t in tokens):
            if tokens:x[0,:len(tokens)]=torch.tensor([vocab[t] for t in tokens])
            h=torch.tensor([[(float(context['age'])-ck['age_mean'])/ck['age_std'],float(context['sex']=='M')]] if ds=='DDX' else [[0.,0.]],dtype=torch.float32)
            with torch.no_grad():p=model(x,h).softmax(1)[0].numpy()
        q=tree.predict(enc.encode(tokens,context)[None],num_threads=1)[0]
        return {'encoder':p,'lightgbm':q}
    archived={(r['patient_id'],r['probe']):r['full_probabilities'] for r in score_rows}
    audit={'cpu_float32_anchor_tolerance':2e-4,'anchor_patients':0,'maximum_anchor_delta':0.,'argmax_changes':0,
           'models':{'encoder':sha(neural_path),'lightgbm':sha(tree_path)}}
    # All full-input predictions are verified; this also validates label axes and input handling.
    result={}
    for i,(pid,r) in enumerate(sorted(query.items()),1):
        full=predict(r['tokens'],r['context']);bg=predict([],r['context'])
        for probe in ['encoder','lightgbm']:
            a=archived[pid,probe];b=full[probe]
            if (a is None)!=(b is None):raise ValueError('Full-input availability differs in background adapter')
            if a is not None:
                delta=float(np.max(np.abs(np.asarray(a)-b)));audit['maximum_anchor_delta']=max(audit['maximum_anchor_delta'],delta)
                if np.argmax(a)!=np.argmax(b):audit['argmax_changes']+=1
                if delta>2e-4 or np.argmax(a)!=np.argmax(b):raise ValueError('Frozen background adapter disagrees with archived full prediction: '+probe)
            value=bg[probe]
            if value is not None and (not np.isfinite(value).all() or not np.isclose(np.sum(value),1,atol=1e-5)):raise ValueError('Invalid background probabilities')
            result[pid,probe]=value.tolist() if value is not None else None
        audit['anchor_patients']+=1
        if i%100==0:print(ds,'context/full anchor',i,'/',len(query),flush=True)
    return result,labels,audit

def trace_profile(cfg,rs):
    test={r['patient_id']:r for r in rows(Path(cfg['view'])/'test_records_PRIVATE.jsonl')}
    train=rows(Path(cfg['view'])/'train_records_PRIVATE.jsonl');vocabulary={t for r in train for t in r['tokens']}
    patients={r['patient_id']:r for r in rows(Path(cfg['view'])/'patients_model_input_PRIVATE.jsonl') if r['patient_id'] in test}
    provenance={r['fact_id']:r for r in rows(cfg['fact_provenance'])}
    gen=rows(cfg['generations']);gen=[r for r in gen if r['patient_id'] in test]
    ledger={r['explanation_id']:r for r in rows(cfg['ledger'])}
    if len(gen)!=4*len(test) or len(ledger)!=len(gen):raise ValueError('TRACE profile ledger census mismatch')
    record_index={(r['explanation_id'],r['probe']):r for r in rs};out=[];diseases=[];sources=[]
    for group in ['retained_322','excluded_37']:
        ids={p for p,r in test.items() if bool(set(r['tokens'])-vocabulary)==(group=='excluded_37')}
        if len(ids)!=(322 if group=='retained_322' else 37):raise ValueError('Frozen OOV group changed')
        gg=[r for r in gen if r['patient_id'] in ids];valid=[r for r in gg if r['status']=='complete']
        complete=[r for r in gg if ledger[r['task_id']]['repair_status']=='COMPLETE']
        counts=[len(test[p]['tokens']) for p in ids]
        top1=sum(r['parsed_diagnoses'][0]==test[r['patient_id']]['label'] for r in valid)
        top3=sum(test[r['patient_id']]['label'] in r['parsed_diagnoses'] for r in valid)
        out.append(dict(group=group,admissions=len(ids),observed_facts_mean=float(np.mean(counts)),
             observed_facts_median=float(np.median(counts)),observed_facts_q25=float(np.quantile(counts,.25)),observed_facts_q75=float(np.quantile(counts,.75)),
             generation_attempts=len(gg),generation_valid=len(valid),generation_valid_rate=len(valid)/len(gg),
             mapping_complete=len(complete),mapping_complete_rate_all=len(complete)/len(gg),
             mapping_complete_rate_valid=len(complete)/len(valid) if valid else None,
             llm_top1_all=top1/len(gg),llm_top3_all=top3/len(gg),llm_top1_valid=top1/len(valid) if valid else None,
             component='TRACE2400',unknown_state_occurrences=sum(len(set(test[p]['tokens'])-vocabulary) for p in ids)))
        for d,n in sorted(Counter(test[p]['label'] for p in ids).items()):diseases.append(dict(group=group,disease=d,admissions=n,proportion=n/len(ids)))
        bysource=defaultdict(Counter)
        for pid in ids:
            expected=set(test[pid]['tokens']);facts=patients[pid]['facts']
            if {f['feature_id']+'_@_'+f['state'] for f in facts}!=expected:raise ValueError('Fact/token mismatch')
            for f in facts:
                if f['fact_id'] not in provenance:raise ValueError('Missing fact provenance')
                for src in provenance[f['fact_id']]['sources']:bysource[src][pid]+=1
        for src,c in sorted(bysource.items()):sources.append(dict(group=group,source=src,admissions_with_source=len(c),
            source_fact_memberships=sum(c.values()),mean_facts_per_admission=sum(c.values())/len(ids),
            note='One fact can have multiple source types; source counts overlap'))
        for probe in ['encoder','lightgbm']:
            sample={r['patient_id']:r for r in rs if r['patient_id'] in ids and r['probe']==probe}
            values=[r for r in sample.values() if r['full_probabilities'] is not None]
            out.append(dict(group=group,probe=probe,admissions=len(ids),full_evaluable=len(values),
                full_correct=sum(r['full_correct'] for r in values),full_accuracy=sum(r['full_correct'] for r in values)/len(values) if values else None))
    return out,diseases,sources

def run(a):
    config=read(a.config)
    for ds in config['datasets']:config['datasets'][ds]=resolved(config['datasets'][ds])
    config['references']=resolved(config['references'])
    preflight=check_inputs(config,not a.skip_background)
    for p in config['references'].values():
        if not Path(p).exists():raise FileNotFoundError(p)
    if a.preflight:
        print(json.dumps(preflight,ensure_ascii=False,indent=2));return
    a.out.mkdir(parents=True,exist_ok=False)
    dump(a.out/'run_manifest.json',{'status':'running'})
    protocol=dict(config=config,scope=['A1','A2','A3','B1','B2'],C='awaiting adjudication; not executed',
        inference='post-review specified families; percentile marginal CIs and centered maximum absolute deviation simultaneous CIs',
        family_resampling='shared draws from the union of clusters; each contrast retains its own common sample',
        family_repetitions=5000,family_seed=261016,original_seed=261006,original_repetitions=1000,
        stability_seed=261016,stability_repetitions=5000,empty_input='frozen CPU float32 Encoder/Tree; full-input anchors checked',
        paper_modified=False,training=False,api_calls=0)
    protocol['families']=[dict(dataset=ds,rq=rq,probe='encoder',endpoint='reference',
        comparisons=AP_COMPARISONS if rq=='RQ2' else [('lightgbm_absolute','tree_positive_shap',b) for b in [.25,.5,.75]],
        metric='failure_ap' if rq=='RQ2' else 'patient_mean_pair_concordance',
        stratum='correct' if rq=='RQ2' else 'all',
        cluster='subject_group' if ds=='TRACE' else ('pattern_id' if rq=='RQ2' else 'patient_id'))
        for ds in ['DDX','TRACE'] for rq in ['RQ2','RQ3']]
    dump(a.out/'contrast_family.json',protocol)
    state={'status':'failed'}
    try:
        datasets={};panels={};absolute=[];avail=[];jobs=[];context_audits={}
        for ds,cfg in config['datasets'].items():
            print('Load',ds,'frozen score records',flush=True)
            datasets[ds]={rq:load_scores(cfg[rq.lower()],ds,rq) for rq in ['RQ2','RQ3']}
            for rq in ['RQ2','RQ3']:
                expected=config.get('accepted_score_sha256',{}).get(ds+'_'+rq)
                if expected and sha(cfg[rq.lower()])!=expected:raise ValueError('Accepted frozen score file differs: '+ds+' '+rq)
            panels[ds]={rq:read(config['references'][ds+'_'+rq]) for rq in ['RQ2','RQ3']}
            if a.skip_background:
                bg={};labels=read(Path(cfg['models'])/'fitted_sources.json').get('labels')
                if ds=='TRACE':labels=sorted({r['reference_diagnosis'] for r in datasets[ds]['RQ2']})
                context_audits[ds]={'status':'not_run_by_explicit_flag'}
            else:
                print('A2',ds,'frozen context-only prediction and full-input anchor verification',flush=True)
                bg,labels,context_audits[ds]=context_predictions(ds,cfg,datasets[ds]['RQ2'],a.threads)
                # Internal cache remains on the server; aggregate report excludes patient identifiers.
                dump(a.out/(ds+'_context_PRIVATE.json'),[dict(patient_id=p,probe=m,probabilities=v) for (p,m),v in bg.items()])
            for rq in ['RQ2','RQ3']:
                rr=datasets[ds][rq];pp=panels[ds][rq]
                avail+=availability(rr,pp,ds,rq)
                absolute+=absolute_endpoints(rr,labels,ds,rq,bg)
                jobs.append((ds,rq,rr,pp,5000,261016))
        print('A3 TRACE excluded/retained profiles',flush=True)
        pr,di,so=trace_profile(config['datasets']['TRACE'],datasets['TRACE']['RQ2'])
        table(a.out/'TRACE_excluded_profile.csv',pr);table(a.out/'TRACE_excluded_diseases.csv',di);table(a.out/'TRACE_excluded_sources.csv',so)
        table(a.out/'availability_by_family.csv',avail);table(a.out/'absolute_probe_endpoints.csv',absolute)
        dump(a.out/'context_inference_audit.json',context_audits)
        print('B1/B2 four fixed families; 5000 paired clustered replicates',flush=True)
        paired=[];diagnostics=[]
        with ProcessPoolExecutor(max_workers=min(4,a.workers)) as pool:
            futures={pool.submit(family_job,j):j[:2] for j in jobs}
            for f in as_completed(futures):
                rr,dd=f.result();paired+=rr;diagnostics+=dd;print('DONE',futures[f],flush=True)
        paired.sort(key=lambda r:(r['dataset'],r['rq'],r['budget'] or 0,r['left'],r['right']))
        table(a.out/'paired_robustness.csv',paired);table(a.out/'bootstrap_diagnostics.csv',diagnostics)
        dump(a.out/'results.json',dict(paired=paired,bootstrap_diagnostics=diagnostics))
        warnings=[]
        if a.skip_background:warnings.append('A2 context-only predictions omitted by command flag; rerun in a new output directory for full A2.')
        if any(r['simultaneous_status']!='available' for r in paired):warnings.append('Some family intervals condition on defined bootstrap draws; inspect valid and zero-event counts before inference.')
        state=dict(status='complete' if not a.skip_background else 'partial_background_pending',completed=['A1','A3','B1','B2']+([] if a.skip_background else ['A2']),
            A2='complete' if not a.skip_background else 'background_pending',warnings=warnings,
            reference_gates='common denominators, point estimates and DDX original bootstrap intervals reproduced',
            source_identity={ds:{rq:sha(config['datasets'][ds][rq.lower()]) for rq in ['RQ2','RQ3']} for ds in datasets},
            no_paper_changes=True,api_calls=0,model_fits=0)
        summary=['# Round1 A/B supplement','', 'Status: '+state['status'],
                 'Frozen original mappings and endpoints; C adjudication is pending.',
                 'All paired point estimates reproduce the supplied accepted summaries.',
                 'RQ2: pooled failure AP. RQ3: within-admission/patient pair concordance, then mean across admissions/patients.',
                 'A1 independent reason counts overlap. Sequential funnel rows provide additive exclusions.',
                 'Missing reason metadata remains reason_not_recorded; it is never inferred as zero positive weight.',
                 'A2 record-weighted and patient-mean accuracies are separate columns; H contains age/sex for DDXPlus and is empty for TRACE.',
                 'Bootstrap intervals use proportions; multiply differences by 100 for percentage points.',
                 'B1 families were specified after review. B2 measures Monte Carlo stability.', '']
        summary+=warnings
        (a.out/'README_results.md').write_text('\n'.join(summary),encoding='utf-8')
    except Exception:
        state.update(error=traceback.format_exc());raise
    finally:
        dump(a.out/'run_manifest.json',state)
        report=Path(str(a.out)+'_reports.tar.gz')
        with tarfile.open(report,'w:gz') as tar:
            for f in sorted(a.out.iterdir()):
                if f.is_file() and 'PRIVATE' not in f.name:tar.add(f,arcname=f.name)
        print('Return report:',report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/round1_followup_20261016.json')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/round1_followup_v1')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--preflight',action='store_true');p.add_argument('--skip-background',action='store_true')
    run(p.parse_args())
