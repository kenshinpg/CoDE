"""Read-only Round 1 statistics from accepted v3_3 caches. No model/API calls."""
import argparse, csv, hashlib, itertools, json, multiprocessing, os, sys, tarfile, time, traceback
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA={}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_csv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def ap(y,s):
    y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float)
    if not len(y) or not y.sum():return None
    if not np.isfinite(s).all():raise ValueError('Nonfinite score')
    ix=np.argsort(-s,kind='stable');z=y[ix];v=s[ix]
    ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1]
    tp=np.cumsum(z)[ends];recall=tp/z.sum()
    return float(np.sum(np.diff(np.r_[0.,recall])*tp/(ends+1)))
def close(a,b):return a is None and b is None or a is not None and b is not None and abs(a-b)<1e-10
def meta(ds,p):return dict(dataset=ds,mapping_policy=p.get('mapping_policy','complete_only'),probe=p['probe'],stratum=p['stratum'],endpoint=p['endpoint'],family=p['family'])
def derived(value,prevalence):
    return dict(ap=value,failure_prevalence=prevalence,ap_minus_prevalence=value-prevalence if value is not None and prevalence is not None else None,ap_lift=value/prevalence if value is not None and prevalence else None)
def self_eval(m,probe):
    return m.startswith(probe+'_') or probe=='lightgbm' and (m.startswith('tree_') or m.startswith('permuted_'))
def summary_tables(ds,panels):
    aps=[];pairs=[];lodo=[];align=[]
    for p in panels:
        m=meta(ds,p);prev=p['failures']/p['common'] if p['common'] else None
        assert close(prev,p['failure_prevalence'])
        for method,v in p['methods'].items():
            aps.append(dict(m,method=method,records=p['common'],patients_or_admissions=p['patients'],failures=p['failures'],**derived(v['ap'],prev),self_evaluation=self_eval(method,p['probe']),existing_ap_ci_low=(v['ci95'] or [None,None])[0],existing_ap_ci_high=(v['ci95'] or [None,None])[1]))
            pairs.append(dict(m,method=method,contributing_patients_or_admissions=p['pair_patients'],discordant_pairs=p['discordant_pairs'],concordance=v.get('pair_concordance'),reporting='descriptive_only' if p['pair_patients']<=11 else 'point_estimate; existing inference unchanged'))
        for row in p.get('information_audit',{}).get('leave_one_disease',[]):
            prev2=row['failure']['explanations']/row['explanations'] if row['explanations'] else None
            for method,value in row['ap'].items():lodo.append(dict(m,removed_disease=row['removed_disease'],records=row['explanations'],patients_or_admissions=row['patients'],failures=row['failure']['explanations'],method=method,**derived(value,prev2),origin='previously_saved_descriptive_deletion'))
    by={tuple(p.get(k,'complete_only') for k in ['mapping_policy','probe','stratum','endpoint','family']):p for p in panels}
    for p in panels:
        if p['endpoint']!='proposal':continue
        key=(p.get('mapping_policy','complete_only'),p['probe'],p['stratum'],'reference',p['family']);r=by[key]
        assert (p['common'],p['patients'])==(r['common'],r['patients'])
        for method,v in p['methods'].items():align.append(dict(meta(ds,p),method=method,records=p['common'],patients_or_admissions=p['patients'],reference_failures=r['failures'],proposal_disagreements=p['failures'],reference_failure_ap=r['methods'][method]['ap'],proposal_disagreement_ap=v['ap'],interpretation='proposal disagreement is cross-model agreement endpoint, not clinical explanation quality'))
    return aps,pairs,lodo,align

def init_worker(data):
    global DATA
    DATA=data
def common_rows(ds,p):
    methods=list(p['methods']);out=[]
    for r in DATA[ds]:
        if r['probe']!=p['probe'] or not r['mapping_available']:continue
        if ds=='TRACE' and p.get('mapping_policy')=='complete_only' and r['scan_status']!='COMPLETE':continue
        if not r['probe_available'] or not r['both_full_correct']:continue
        if p['family']!='quantity_only' and not r['target_supported']:continue
        if p['stratum']!='all' and r['diagnosis_correct']!=(p['stratum']=='correct'):continue
        if not all(r['scores'].get(m) is not None for m in methods):continue
        out.append(r)
    return out
def panel_job(task):
    ds,p=task;rs=common_rows(ds,p);methods=list(p['methods']);key='reference_failure' if p['endpoint']=='reference' else 'proposal_failure'
    y=np.array([r[key] for r in rs],dtype=int);scores={m:-np.array([r['scores'][m] for r in rs]) for m in methods}
    assert len(rs)==p['common'] and int(y.sum())==p['failures'],(ds,p['family'],'population drift')
    assert len({r['patient_id'] for r in rs})==p['patients']
    for m in methods:assert close(ap(y,scores[m]),p['methods'][m]['ap']),(ds,p['family'],m,'AP replay mismatch')
    rows=[];contrasts=[];existing={x['removed_disease']:x for x in p.get('information_audit',{}).get('leave_one_disease',[])}
    labels=np.array([r['reference_diagnosis'] for r in rs]);ids=np.array([r['patient_id'] for r in rs]);clusters=np.array([r.get('subject_group',r.get('pattern_id',r['patient_id'])) for r in rs])
    for disease in sorted(set(labels)):
        mask=labels!=disease;n=int(mask.sum());fail=int(y[mask].sum());prev=fail/n if n else None
        # Match the original LODO convention: a single outcome class is not used for ranking comparisons.
        vals={m:ap(y[mask],scores[m][mask]) if 0<fail<n else None for m in methods}
        info=dict(meta(ds,p),removed_disease=str(disease),records=n,patients_or_admissions=len(set(ids[mask])),clusters=len(set(clusters[mask])),failures=fail)
        for m in methods:
            if disease in existing:assert close(vals[m],existing[disease]['ap'][m]),(ds,disease,m,'saved LODO mismatch')
            rows.append(dict(info,method=m,**derived(vals[m],prev),origin='recomputed_from_same_fixed_scores',status='defined' if 0<fail<n else 'no_outcome_contrast'))
        for a,b in itertools.combinations(methods,2):
            delta=vals[a]-vals[b] if vals[a] is not None and vals[b] is not None else None
            original=p['methods'][a]['ap']-p['methods'][b]['ap'] if p['methods'][a]['ap'] is not None and p['methods'][b]['ap'] is not None else None
            contrasts.append(dict(info,left=a,right=b,ap_difference=delta,original_ap_difference=original,sign_reversal=delta*original<0 if delta is not None and original is not None else None,inference='descriptive; no significance test'))
    return rows,contrasts,dict(meta(ds,p),replayed_records=len(rs),replayed_failures=int(y.sum()),methods=len(methods),deleted_diseases=len(set(labels)),saved_deletions_checked=len(existing))

def intervention_tables(path,reference):
    raw=read(path);expected=read(reference);by=defaultdict(list);seen=set()
    for r in raw:
        k=(r['panel'],r['patient_id']);assert k not in seen,'Duplicate patient aggregate';seen.add(k);by[r['panel']].append(r)
    out=[]
    for panel,p in expected.items():
        rs=by.get(panel,[]);assert len(rs)==p['patients'],panel
        if not rs:continue
        sel,direction,op,stratum,probe=panel.split('|')
        for target in ['d','y']:
            metric='high_minus_random_'+target;x=np.array([r['metrics'][metric] for r in rs],dtype=float)
            assert np.isfinite(x).all() and close(float(x.mean()),p['metrics'][metric]['mean']),panel
            lo,med,hi=np.quantile(x,[.25,.5,.75]);eps=1e-10
            positive=int((x>eps).sum());negative=int((x< -eps).sum());zero=int((np.abs(x)<=eps).sum());assert positive+zero+negative==len(x)
            out.append(dict(selector=sel,direction=direction,operation=op,proposal_stratum=stratum,probe=probe,target=target,patients=len(x),analyzed_explanations=p['analyzed_explanations'],mean_pp=100*float(x.mean()),q25_pp=100*float(lo),median_pp=100*float(med),q75_pp=100*float(hi),positive_patients=positive,zero_patients=zero,negative_patients=negative,positive_fraction=positive/len(x),zero_fraction=zero/len(x),negative_fraction=negative/len(x),numerical_zero_tolerance=eps,interpretation='patient-average effects; positive is not per-patient statistical or clinical significance'))
    return out

def load_trimmed(path,ds):
    keep=['explanation_id','patient_id','pattern_id','subject_group','reference_diagnosis','probe','mapping_available','scan_status','target_supported','probe_available','both_full_correct','diagnosis_correct','reference_failure','proposal_failure','scores']
    out=[];seen=set()
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line);k=(r['explanation_id'],r['probe']);assert k not in seen,'Duplicate score key';seen.add(k)
            out.append({k:r[k] for k in keep if k in r})
    assert len(out)==(12048 if ds=='DDX' else 2872),(ds,len(out))
    return out
def run(a):
    cfg=read(a.config);datasets=a.datasets.split(',');tasks=[];audit={};allap=[];allpairs=[];allsaved=[];allalign=[]
    for ds in datasets:
        c=cfg['datasets'][ds];ref=ROOT/c['reference'];h=sha(ref)
        if h!=c['reference_sha256']:raise ValueError('Reference summary changed: '+ds)
        panels=read(ref);aa,pp,ll,ee=summary_tables(ds,panels)
        allap+=aa;allpairs+=pp;allsaved+=ll;allalign+=ee;tasks.extend((ds,p) for p in panels)
        audit[ds]={'reference_sha256':h,'panels':len(panels)}
        if not a.summary_only:
            path=ROOT/c['scores'];print('LOAD',ds,path,flush=True)
            h=sha(path)
            if h!=c['scores_sha256']:raise ValueError('Accepted score cache changed: '+ds)
            DATA[ds]=load_trimmed(path,ds);audit[ds]['score_sha256']=h
        print('SUMMARY',ds,len(panels),'panels',flush=True)
    for name,rs in [('ap_prevalence_lift',allap),('pair_descriptive',allpairs),('saved_leave_one_disease',allsaved),('endpoint_alignment',allalign)]:write_csv(a.out/(name+'.csv'),rs)
    ip=Path(a.interventions) if a.interventions else ROOT/cfg['interventions']['patient_aggregates']
    ir=Path(a.intervention_reference) if a.intervention_reference else ROOT/cfg['interventions']['reference']
    if sha(ip)!=cfg['interventions']['patient_sha256'] or sha(ir)!=cfg['interventions']['reference_sha256']:raise ValueError('Intervention aggregates changed')
    intervention=intervention_tables(ip,ir);write_csv(a.out/'intervention_patient_distribution.csv',intervention)
    print('INTERVENTIONS',len(intervention),'patient-level panels verified',flush=True)
    if not a.summary_only:
        workers=min(a.workers,len(tasks),os.cpu_count() or 1);ctx=multiprocessing.get_context('fork' if sys.platform!='win32' else 'spawn')
        results=[];deletions=[];contrasts=[];started=time.time()
        print('PARALLEL',workers,'processes; one numeric thread each;',len(tasks),'tasks',flush=True)
        with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_worker,initargs=(DATA,)) as pool:
            pending={pool.submit(panel_job,t) for t in tasks}
            while pending:
                done,pending=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
                for future in done:
                    rows,ds,log=future.result();deletions+=rows;contrasts+=ds;results.append(log)
                if not done or len(results)%10==0 or not pending:print('PROGRESS',len(results),'/',len(tasks),'elapsed',round(time.time()-started,1),'seconds',flush=True)
        sort=lambda r:tuple(str(r.get(k,'')) for k in ['dataset','mapping_policy','probe','stratum','endpoint','family','removed_disease','method','left','right'])
        write_csv(a.out/'leave_one_disease_all_methods.csv',sorted(deletions,key=sort))
        write_csv(a.out/'leave_one_disease_all_contrasts.csv',sorted(contrasts,key=sort))
        dump(a.out/'cache_replay_checks.json',sorted(results,key=sort))
        print('REPLAY VERIFIED',len(results),'panels;',len(contrasts),'descriptive LODO contrasts',flush=True)
    dump(a.out/'input_audit.json',audit)
    return dict(status='complete',mode='summary_only' if a.summary_only else 'full_cache_replay',datasets=datasets,panels=len(tasks),ap_rows=len(allap),intervention_rows=len(intervention),models_trained=False,model_predictions_added=False,api_calls=0,new_bootstrap_intervals=False,reviewer_responses_confirmed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/round1_cached_statistics_20261023.json');p.add_argument('--out',type=Path,default=ROOT/'artifacts/round1_cached_statistics_v1');p.add_argument('--workers',type=int,default=64);p.add_argument('--datasets',default='DDX,TRACE');p.add_argument('--summary-only',action='store_true');p.add_argument('--interventions',type=Path);p.add_argument('--intervention-reference',type=Path)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);state={};start=time.time()
    try:state=run(a)
    except Exception:state={'status':'failed','error':traceback.format_exc()};raise
    finally:
        state['seconds']=time.time()-start;dump(a.out/'run_manifest.json',state)
        (a.out/'README.txt').write_text('Read-only summaries of accepted automatic-mapping results. AP lift and AP-minus-prevalence are point estimates using each panel\'s own denominator; prevalence is the no-information reference, not an exact finite-sample expected random AP. Existing AP intervals are unchanged. LODO is descriptive and excludes every represented disease within each frozen common panel. Undefined/single-outcome deletions are retained. No recalibration, prediction, training or LLM calls. Patient-level effects average explanations within each patient before equal patient weighting. Positive effects are numerical directions, not per-patient significance. Mapping policies, proposal strata, probes and endpoints remain separate. TRACE identifiers and raw notes are excluded from this return package. Reviewer replies remain unconfirmed.\n',encoding='utf-8')
        archive=Path(str(a.out)+'_reports.tar.gz')
        with tarfile.open(archive,'w:gz') as tar:
            for f in sorted(a.out.glob('*')):
                if f.is_file() and f.suffix in ['.csv','.json','.txt']:tar.add(f,arcname=f.name)
        print('Return report:',archive,flush=True)
