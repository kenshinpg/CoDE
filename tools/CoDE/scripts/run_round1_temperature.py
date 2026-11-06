"""Development-only temperature fit; fixed-test RQ2/RQ3 ranking sensitivity.

No model training or LLM calls. Coverage/attribution scores, outcomes, eligibility
and cluster units remain frozen. Only Tree direct support is transformed in
ranking comparisons. Confidence intervals condition on the fitted temperatures.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
import argparse,csv,json,sys,time,tarfile,traceback,hashlib,multiprocessing
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from round1_temperature_data import read,rows,sha,checked,load_development
from run_round1_budget_calibration import calibration
from code_eval.round1_followup import APContrast,PairContrast,intervals,primary,accepted_contrast,AP_COMPARISONS
FAMILIES=[]

def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(p,rs):
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rs: w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def transform(p,T):
    p=np.asarray(p,float)
    if not np.isfinite(T) or T<=0: raise ValueError('Temperature must be positive')
    if p.ndim!=2 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5): raise ValueError('Invalid probability matrix')
    if T==1.: return p.copy()
    z=np.log(np.maximum(p,1e-12))/T; z-=z.max(1,keepdims=True); q=np.exp(z); q/=q.sum(1,keepdims=True)
    if not np.array_equal(q.argmax(1),p.argmax(1)): raise ValueError('Temperature changed a hard prediction')
    return q

def fit_temperature(budgets):
    """Equal budget weights, equal available records within budget; bounded scalar NLL."""
    arrays=[]
    for p,y in budgets:
        transform(p,1.); y=np.asarray(y,int)
        if len(y)!=len(p) or not len(y) or (y<0).any() or (y>=p.shape[1]).any(): raise ValueError('Invalid development targets')
        arrays.append((np.log(np.maximum(p,1e-12)),y))
    def loss(logt):
        ans=[]
        for logp,y in arrays:
            z=logp/np.exp(logt);m=z.max(1);lse=m+np.log(np.exp(z-m[:,None]).sum(1))
            ans.append(float(np.mean(lse-z[np.arange(len(y)),y])))
        return float(np.mean(ans))
    grid=np.linspace(np.log(.05),np.log(20.),81);ls=[loss(x) for x in grid];i=int(np.argmin(ls));lo=grid[max(0,i-1)];hi=grid[min(80,i+1)]
    ratio=(5**.5-1)/2
    for _ in range(70):
        a=hi-ratio*(hi-lo);b=lo+ratio*(hi-lo)
        if loss(a)<loss(b):hi=b
        else:lo=a
    candidates=[0.,float(grid[0]),float(grid[-1]),float((lo+hi)/2)]
    best=min(candidates,key=loss);T=float(np.exp(best));original=loss(0.);fitted=loss(best)
    if fitted>original+1e-12:raise ValueError('Development optimizer worsened NLL')
    return dict(temperature=T,dev_nll_before=original,dev_nll_after=fitted,boundary=bool(T<=.05001 or T>=19.999),bounds=[.05,20.],available_by_budget=[len(y) for _,y in budgets])

def fit_all(data,profiles):
    fits={};out=[]
    for ds,info in data.items():
        for probe in ['encoder','lightgbm']:
            for budget in ['pooled',0,.25,.5,.75,1]:
                fs=[.25,.5,.75,1] if budget=='pooled' else [budget]
                r=fit_temperature([info['probabilities'][probe,f] for f in fs]);fits[ds,probe,str(budget)]=r['temperature']
                out.append(dict(dataset=ds,probe=probe,fit_budget=budget,**profiles[ds,probe],**r,policy='equal weights across preset .25/.5/.75/1 budgets; zero budget excluded from pooled fit' if budget=='pooled' else 'single development budget; calibration diagnostic only'))
                print('FIT',ds,probe,budget,'T=',round(r['temperature'],5),'development NLL',round(r['dev_nll_before'],6),'->',round(r['dev_nll_after'],6),flush=True)
    return fits,out

def calibrate_scores(rs,labels,T):
    """Join Tree posterior to any probe row; never infer posterior from scalar A."""
    by={r['explanation_id']:r for r in rs if r['probe']=='lightgbm'}
    if len(by)!=sum(r['probe']=='lightgbm' for r in rs):raise ValueError('Duplicate Tree score records')
    calibrated={};checked_n=0
    for eid,r in by.items():
        if r['scores'].get('lightgbm_absolute') is None:continue
        if r['diagnosis'] not in labels or r['cited_probabilities'] is None:raise ValueError('Scored A without supported posterior')
        p=np.array(r['cited_probabilities'])[None];d=labels.index(r['diagnosis'])
        if abs(float(p[0,d])-r['scores']['lightgbm_absolute'])>1e-10:raise ValueError('Tree A posterior replay mismatch')
        calibrated[eid]=float(transform(p,T)[0,d]);checked_n+=1
    out=[]
    for r in rs:
        s=dict(r['scores']);eid=r['explanation_id']
        if s.get('lightgbm_absolute') is not None:
            tree=by[eid]
            if any(r[k]!=tree[k] for k in ['patient_id','diagnosis','reference_diagnosis']):raise ValueError('Cross-probe join mismatch')
            if abs(s['lightgbm_absolute']-tree['scores']['lightgbm_absolute'])>1e-10:raise ValueError('Cross-probe A mismatch')
            s['lightgbm_absolute']=calibrated[eid]
        out.append(dict(r,scores=s))
    return out,checked_n

def make_family(ds,rq,rs,after,panels):
    pairs=AP_COMPARISONS if rq=='rq2' else [('lightgbm_absolute','tree_positive_shap')]*3
    before_cs=[];after_cs=[];metas=[]
    cluster='subject_group' if ds=='TRACE' else ('pattern_id' if rq=='rq2' else 'patient_id')
    for i,(left,right) in enumerate(pairs):
        fraction=None if rq=='rq2' else [.25,.5,.75][i];stratum='correct' if rq=='rq2' else 'all'
        fam='paired_'+('tree_positive_shap_vs_lightgbm_absolute' if rq=='rq3' else left+'_vs_'+right)
        ps=[p for p in panels if p['probe']=='encoder' and p['endpoint']=='reference' and p['stratum']==stratum and p['family']==fam and p.get('fraction')==fraction and p.get('mapping_policy','complete_only')=='complete_only']
        if len(ps)!=1:raise ValueError('Missing frozen core panel '+str((ds,rq,fam)))
        p=ps[0];a=primary(rs,list(p['methods']),stratum,fraction);b=primary(after,list(p['methods']),stratum,fraction)
        if [(r['explanation_id'],r['reference_failure']) for r in a]!=[(r['explanation_id'],r['reference_failure']) for r in b]:raise ValueError('Core population/outcome changed')
        cls=APContrast if rq=='rq2' else PairContrast;c=cls(a,left,right,cluster);v=cls(b,left,right,cluster)
        actual=c.evaluate(np.ones(c.n));old=accepted_contrast(p,left,right,'ap' if rq=='rq2' else 'pair_concordance')['difference']
        if old is None or abs(actual-old)>1e-10 or len(a)!=p['common'] or len({r['patient_id'] for r in a})!=p['patients'] or sum(r['reference_failure'] for r in a)!=p['failures']:raise ValueError('Core baseline replay failed')
        if rq=='rq3' and (c.n!=p['pair_patients'] or c.pairs!=p['discordant_pairs']):raise ValueError('Core pair count mismatch')
        before_cs.append(c);after_cs.append(v)
        metas.append(dict(dataset=ds,analysis=rq,probe='encoder',stratum=stratum,left=left,right=right,budget=fraction,records=len(a),patients=p['patients'],failures=p['failures'],clusters=len(set(c.cluster)),pair_patients=c.n if rq=='rq3' else None,pairs=c.pairs if rq=='rq3' else None,cluster=cluster,baseline_replayed=True))
    return {'before':before_cs,'after':after_cs,'meta':metas}

def init_workers(families):
    global FAMILIES
    FAMILIES=families

def bootstrap_block(task):
    index,start,count=task;fam=FAMILIES[index];cs=fam['before'];ds=fam['after']
    keys=sorted(set(k for c in cs for k in c.cluster));look={k:i for i,k in enumerate(keys)};ix=[np.array([look[k] for k in c.cluster]) for c in cs]
    before=np.full((count,len(cs)),np.nan);after=before.copy()
    for b in range(count):
        rng=np.random.default_rng(np.random.SeedSequence([261023,index,start+b]))
        weights=np.bincount(rng.integers(len(keys),size=len(keys)),minlength=len(keys))
        for j,(c,d,ind) in enumerate(zip(cs,ds,ix)):
            before[b,j]=c.evaluate(weights[ind]);after[b,j]=d.evaluate(weights[ind])
    return index,start,before,after

class Difference:
    def __init__(self,a,b): self.a=a;self.b=b;self.n=a.n
    def evaluate(self,w):return self.b.evaluate(w)-self.a.evaluate(w)

def core_summary(families,draws):
    result=[]
    for i,f in enumerate(families):
        for version,cs,dd in [('original',f['before'],draws[i][0]),('pooled_temperature',f['after'],draws[i][1]),('calibrated_minus_original',[Difference(a,b) for a,b in zip(f['before'],f['after'])],draws[i][1]-draws[i][0])]:
            for meta,stats in zip(f['meta'],intervals(cs,dd)):
                result.append(dict(meta,variant=version,**stats,inference='post-review sensitivity; fixed development temperatures/fitted models; paired cluster bootstrap; simultaneous intervals within three contrasts for this dataset/RQ/variant'))
    return result

def descriptive_panels(ds,rs,after,panels):
    out=[]
    for p in panels:
        if p['endpoint']!='reference' or p.get('mapping_policy','complete_only')!='complete_only':continue
        for left,right in AP_COMPARISONS:
            if p['family']!='paired_'+left+'_vs_'+right:continue
            a=primary(rs,list(p['methods']),p['stratum'],probe=p['probe']);b=primary(after,list(p['methods']),p['stratum'],probe=p['probe'])
            if len(a)!=p['common']:raise ValueError('Descriptive population drift')
            cluster='subject_group' if ds=='TRACE' else 'pattern_id'
            vals=[]
            for group in [a,b]:
                c=APContrast(group,left,right,cluster);x=c.evaluate(np.ones(c.n));vals.append(float(x) if np.isfinite(x) else None)
            old=accepted_contrast(p,left,right,'ap')['difference']
            if old is not None and (vals[0] is None or abs(old-vals[0])>1e-10):raise ValueError('Descriptive original AP mismatch')
            out.append(dict(dataset=ds,probe=p['probe'],stratum=p['stratum'],left=left,right=right,records=len(a),failures=sum(r['reference_failure'] for r in a),original=vals[0],calibrated=vals[1],change=vals[1]-vals[0] if None not in vals else None,sign_reversal=vals[0]*vals[1]<0 if None not in vals else None,inference='descriptive; same original common records, outcomes and coverage weights'))
    return out

def calibration_panels(ds,rq,rs,labels,fits):
    groups=defaultdict(dict);out=[];bins=[]
    for r in rs:
        if not r.get('probe_available') or not r.get('mapping_available',True):continue
        if ds=='TRACE' and rq=='rq2' and r.get('scan_status')!='COMPLETE':continue
        y=labels.index(r['reference_diagnosis']);budget=r.get('fraction','natural')
        for pop in ['all_cached_available']+(['both_full_correct'] if r['both_full_correct'] else []):
            for view,field,key in [('full','full_probabilities',r['patient_id']),('cited','cited_probabilities',r['explanation_id'])]:
                p=r.get(field)
                if p is None:continue
                g=(r['probe'],str(budget) if view=='cited' else 'full',view,pop)
                old=groups[g].get(key)
                if old and (old[1]!=y or not np.allclose(old[0],p,atol=1e-7)):raise ValueError('Repeated probability drift')
                groups[g][key]=(p,y)
    for g,records in groups.items():
        probe,budget,view,pop=g;vals=list(records.values());p=np.array([v[0] for v in vals]);y=np.array([v[1] for v in vals]);temp=fits[ds,probe,'pooled']
        variants=[('original',1.),('pooled_temperature',temp)]
        if budget!='natural':variants.append(('budget_temperature',fits[ds,probe,'1' if budget=='full' else budget]))
        for variant,T in variants:
            q=transform(p,T);m,bb=calibration(q,y);meta=dict(dataset=ds,analysis=rq,probe=probe,budget=budget,view=view,population=pop,variant=variant,temperature=T)
            out.append(dict(meta,**m,prediction_changes=int((q.argmax(1)!=p.argmax(1)).sum())))
            bins.extend(dict(meta,**b) for b in bb)
    return out,bins

def run(a):
    cfg=read(ROOT/'configs/round1_followup_20261016.json');refcfg=read(ROOT/'configs/round1_cached_statistics_20261023.json');lock=read(ROOT/'configs/round1_temperature_20261023.json')
    print('STAGE 1 verify frozen development inputs / reuse probability caches',flush=True)
    data,profiles,audit=load_development(ROOT,cfg,a.out,a.predict_threads)
    print('STAGE 2 DEVELOPMENT-only temperature fitting',flush=True)
    fits,fitrows=fit_all(data,profiles);table(a.out/'development_temperatures.csv',fitrows)
    dump(a.out/'frozen_temperature_policy.json',dict(test_consulted_for_fit=False,pooled_budgets=[.25,.5,.75,1],weights='equal budget; equal available records within budget',bounds=[.05,20],probability_floor=1e-12,fit_objective='development multiclass NLL',ranking_temperature='pooled for all natural/controlled sets',coverage_recomputed=False,outcomes_recomputed=False,intervals_condition_on_temperature=True))
    print('STAGE 3 replay accepted TEST comparisons; fixed records/outcomes',flush=True)
    families=[];diagnostics=[];binrows=[];desc=[];replay=[]
    for ds in ['DDX','TRACE']:
        labels=data[ds]['labels']
        for rq in ['rq2','rq3']:
            path=ROOT/cfg['datasets'][ds][rq];h=sha(path);expected=lock['score_sha256'][ds+'_'+rq.upper()]
            if h!=expected:raise ValueError('Accepted TEST score file changed')
            audit[str(path)]=h;rs=rows(path);seen=set()
            for r in rs:
                key=(r['explanation_id'],r['probe'])
                if key in seen or r['patient_id'] not in data[ds]['test_ids'] or r['patient_id'] in data[ds]['dev_ids']:raise ValueError('Invalid TEST identity')
                seen.add(key)
                if ds=='TRACE' and rq=='rq2':r['mapping_available']=r['mapping_available'] and r['scan_status']=='COMPLETE'
                p=r.get('cited_probabilities')
                if r.get('probe_available') and p is not None:
                    if bool(labels[int(np.argmax(p))]!=r['reference_diagnosis'])!=r['reference_failure']:raise ValueError('Frozen endpoint mismatch')
            refpath=ROOT/cfg['references'][ds+'_'+rq.upper()]
            checked(refpath,lock['reference_sha256'][ds+'_'+rq.upper()],audit)
            panels=read(refpath);after,n=calibrate_scores(rs,labels,fits[ds,'lightgbm','pooled'])
            families.append(make_family(ds,rq,rs,after,panels))
            if rq=='rq2':desc+=descriptive_panels(ds,rs,after,panels)
            cc,bb=calibration_panels(ds,rq,rs,labels,fits);diagnostics+=cc;binrows+=bb
            replay.append(dict(dataset=ds,analysis=rq,records=len(rs),tree_A_posterior_replayed=n,hard_predictions_and_populations_fixed=True))
            print('REPLAY PASS',ds,rq,len(rs),'records;',n,'Tree posteriors;',len(cc),'calibration panels',flush=True)
    table(a.out/'test_calibration.csv',diagnostics);table(a.out/'test_reliability_bins.csv',binrows);table(a.out/'rq2_descriptive_rankings.csv',desc);dump(a.out/'cache_replay.json',replay);dump(a.out/'input_audit.json',audit)
    print('STAGE 4 paired cluster bootstrap, 5000 draws/family; scheduling-independent seeds',flush=True)
    tasks=[(i,start,min(100,5000-start)) for i in range(len(families)) for start in range(0,5000,100)]
    workers=min(a.workers,len(tasks),os.cpu_count() or 1);results=[(np.full((5000,3),np.nan),np.full((5000,3),np.nan)) for _ in families];completed=0
    ctx=multiprocessing.get_context('fork' if sys.platform!='win32' else 'spawn')
    print('BOOTSTRAP',workers,'CPU processes;',len(tasks),'blocks; numerical threads=1',flush=True);start=time.time()
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_workers,initargs=(families,)) as pool:
        pending={pool.submit(bootstrap_block,t) for t in tasks}
        while pending:
            done,pending=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for f in done:
                i,offset,b,c=f.result();results[i][0][offset:offset+len(b)]=b;results[i][1][offset:offset+len(c)]=c;completed+=1
            if not done or completed%10==0 or not pending:print('BOOTSTRAP PROGRESS',completed,'/',len(tasks),'elapsed',round(time.time()-start,1),'seconds',flush=True)
    table(a.out/'core_rankings_paired_intervals.csv',core_summary(families,results))
    np.savez_compressed(a.out/'bootstrap_draws.npz',**{f'{i}_{v}':rr[j] for i,rr in enumerate(results) for j,v in enumerate(['original','calibrated'])})
    return dict(status='complete',development_temperature_fit=True,model_training=False,llm_calls=0,dev_encoder_replayed=True,test_predictions_added=False,core_families=len(families),bootstrap_per_family=5000,calibration_panels=len(diagnostics),workers=workers,coverage_fixed=True,probe_hard_predictions_fixed=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'artifacts/round1_temperature_v1');p.add_argument('--workers',type=int,default=64);p.add_argument('--predict-threads',type=int,default=8);a=p.parse_args()
    if not 1<=a.workers<=128 or not 1<=a.predict_threads<=32:p.error('workers 1..128; predict-threads 1..32')
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False);state={};started=time.time()
    try:state=run(a)
    except Exception:state={'status':'failed','error':traceback.format_exc()};raise
    finally:
        state['seconds']=time.time()-started;dump(a.out/'run_manifest.json',state)
        (a.out/'README.txt').write_text('Development-only positive scalar temperature scaling. Original fits, test cohorts, hard endpoints and coverage weights fixed. Primary RQ2 and RQ3 comparisons use pooled development temperatures; per-budget fits are calibration diagnostics. Same cluster draws pair original/calibrated estimates. Simultaneous intervals cover three contrasts per dataset/RQ/variant. Intervals condition on fitted temperatures; development-estimation uncertainty is not bootstrapped. Undefined draws reported. TRACE cache availability and historical inference-math differences retained. Private development probabilities stay on server and are excluded from this report.\n',encoding='utf-8')
        with tarfile.open(str(a.out)+'_reports.tar.gz','w:gz') as z:
            for f in a.out.iterdir():
                if f.is_file():z.add(f,arcname=f.name)
        print('Return report:',str(a.out)+'_reports.tar.gz',flush=True)
