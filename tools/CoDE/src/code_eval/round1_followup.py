"""Frozen-score follow-up: denominators, absolute endpoints and paired resampling."""
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np

BASE = ['fact_coverage', 'group_coverage']
AP_COMPARISONS = [('cohd_coverage','group_coverage'),
                  ('tree_positive_shap','group_coverage'),
                  ('tree_positive_shap','lightgbm_absolute')]

def census(rs):
    failed = [r for r in rs if r.get('reference_failure') is True]
    return dict(records=len(rs), patients=len({r['patient_id'] for r in rs}),
                failures=len(failed), failure_patients=len({r['patient_id'] for r in failed}))

def primary(rs, methods, stratum='correct', fraction=None, probe='encoder'):
    return [r for r in rs if r['probe']==probe and r['mapping_available']
            and r['both_full_correct'] and r['probe_available'] and r['target_supported']
            and (stratum=='all' or r['diagnosis_correct']==(stratum=='correct'))
            and (fraction is None or r.get('fraction')==fraction)
            and all(r['scores'].get(m) is not None for m in methods)]

def availability(rs, panels, dataset, rq):
    out=[]
    for p in panels:
        if p.get('mapping_policy','complete_only')!='complete_only': continue
        methods=list(p['methods']); frac=p.get('fraction'); stratum=p['stratum']
        base=[r for r in rs if r['probe']==p['probe'] and (frac is None or r.get('fraction')==frac)]
        meta=dict(dataset=dataset,rq=rq,probe=p['probe'],family=p['family'],
                  stratum=stratum,endpoint=p['endpoint'],budget=frac)
        conditions=[('mapping_available',lambda r:r['mapping_available']),
                    ('proposal_stratum',lambda r:stratum=='all' or r['diagnosis_correct']==(stratum=='correct')),
                    ('target_supported',lambda r:p['family']=='quantity_only' or r['target_supported']),
                    ('probe_available',lambda r:r['probe_available']),
                    ('both_full_correct',lambda r:r['both_full_correct']),
                    ('common_scores',lambda r:all(r['scores'].get(m) is not None for m in methods))]
        current=base
        out.append(dict(meta,kind='funnel',stage='start',method='',**census(current)))
        for stage,fn in conditions:
            out.append(dict(meta,kind='independent_exclusion',stage=stage,method='',**census([r for r in base if not fn(r)])))
            current=[r for r in current if fn(r)]
            out.append(dict(meta,kind='funnel',stage=stage,method='',**census(current)))
        if len(current)!=p['common'] or len({r['patient_id'] for r in current})!=p['patients']:
            raise ValueError('Accepted common population differs: '+str(meta))
        scope=[r for r in base if r['mapping_available'] and
               (stratum=='all' or r['diagnosis_correct']==(stratum=='correct'))]
        for m in methods:
            missing=[r for r in scope if r['scores'].get(m) is None]
            reason=lambda r:r.get('coverage_unavailability',{}).get(m) or 'reason_not_recorded'
            for why in sorted({reason(r) for r in missing}):
                out.append(dict(meta,kind='score_missing_reason',stage=why,method=m,
                                **census([r for r in missing if reason(r)==why])))
            out.append(dict(meta,kind='score_available_before_common',stage='mapped_stratum',method=m,
                            **census([r for r in scope if r['scores'].get(m) is not None])))
    return out

class APContrast:
    """Tie-aware weighted AP; bootstrap counts reproduce duplicated-cluster AP."""
    def __init__(self, rs, left, right, cluster):
        self.cluster=[r[cluster] for r in rs]; self.n=len(rs)
        self.y=np.asarray([r['reference_failure'] for r in rs],float)
        self.layouts=[]
        for m in [left,right]:
            s=-np.asarray([r['scores'][m] for r in rs],float)
            ix=np.argsort(-s,kind='stable'); ss=s[ix]
            ends=np.r_[np.where(ss[:-1]!=ss[1:])[0],len(ss)-1] if len(ss) else np.array([],int)
            self.layouts.append((ix,ends))
    def evaluate(self,w):
        positive=float(w@self.y)
        if positive<=0:return np.nan
        ans=[]
        for ix,ends in self.layouts:
            ww=w[ix];tp=np.cumsum(ww*self.y[ix])[ends];total=np.cumsum(ww)[ends]
            precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
            ans.append(float(np.sum(np.diff(np.r_[0.,tp/positive])*precision)))
        return ans[0]-ans[1]

class PairContrast:
    def __init__(self,rs,left,right,cluster):
        by=defaultdict(list)
        for r in rs:by[r['patient_id']].append(r)
        self.cluster=[];self.values=[];self.pairs=0
        for pid,rows in sorted(by.items()):
            ds=[]
            for a,b in combinations(rows,2):
                if a['diagnosis']!=b['diagnosis'] or a['cited_fact_count']!=b['cited_fact_count'] or a['reference_failure']==b['reference_failure']:continue
                good,bad=(b,a) if a['reference_failure'] else (a,b)
                def concord(m):
                    d=good['scores'][m]-bad['scores'][m]
                    return .5 if abs(d)<=1e-12 else float(d>0)
                ds.append(concord(left)-concord(right))
            if ds:
                self.cluster.append(rows[0][cluster]);self.values.append(float(np.mean(ds)));self.pairs+=len(ds)
        self.values=np.asarray(self.values);self.n=len(self.values)
    def evaluate(self,w):return float(w@self.values/w.sum()) if w.sum() else np.nan

def resample(contrasts, repetitions, seed):
    keys=sorted(set(k for c in contrasts for k in c.cluster)); lookup={k:i for i,k in enumerate(keys)}
    indices=[np.asarray([lookup[k] for k in c.cluster],int) for c in contrasts]
    rng=np.random.default_rng(seed);result=np.full((repetitions,len(contrasts)),np.nan)
    if not keys:return result
    for b in range(repetitions):
        counts=np.bincount(rng.integers(0,len(keys),len(keys)),minlength=len(keys))
        for j,(c,ix) in enumerate(zip(contrasts,indices)):result[b,j]=c.evaluate(counts[ix])
    return result

def intervals(contrasts,draws):
    points=np.array([c.evaluate(np.ones(c.n)) for c in contrasts])
    joint=np.isfinite(draws).all(1); ok=np.isfinite(points).all() and int(joint.sum())>=2
    radius=float(np.quantile(np.max(np.abs(draws[joint]-points),axis=1),.95)) if ok else None
    result=[]
    for j,p in enumerate(points):
        values=draws[np.isfinite(draws[:,j]),j]
        result.append(dict(difference=float(p) if np.isfinite(p) else None,
            ci95_percentile=np.quantile(values,[.025,.975]).tolist() if len(values) else None,
            simultaneous_ci95=[float(p-radius),float(p+radius)] if ok else None,
            bootstrap_requested=len(draws),valid_replicates=len(values),undefined_replicates=len(draws)-len(values),
            jointly_valid_replicates=int(joint.sum()),simultaneous_status='available' if ok and joint.all() else
            ('conditional_on_defined_replicates' if ok else 'unavailable')))
    return result

def accepted_contrast(panel,left,right,metric):
    values=panel['paired_ap_differences'] if metric=='ap' else panel['information_audit']['paired_concordance_differences']
    key=left+' minus '+right;reverse=key not in values
    r=values[right+' minus '+left] if reverse else values[key]
    return dict(difference=(-r['difference'] if reverse else r['difference']) if r['difference'] is not None else None,
                ci95=([-r['ci95'][1],-r['ci95'][0]] if reverse else r['ci95']) if r['ci95'] else None)

def family_job(task):
    dataset,rq,rs,panels,reps,seed=task
    configs=AP_COMPARISONS if rq=='RQ2' else [('lightgbm_absolute','tree_positive_shap')]*3
    cs=[];meta=[]
    for i,(left,right) in enumerate(configs):
        fraction=None if rq=='RQ2' else [.25,.5,.75][i]
        family='paired_'+('tree_positive_shap_vs_lightgbm_absolute' if rq=='RQ3' else left+'_vs_'+right)
        pp=[p for p in panels if p['probe']=='encoder' and p['endpoint']=='reference' and
            p['stratum']==('correct' if rq=='RQ2' else 'all') and p['family']==family and
            p.get('fraction')==fraction and p.get('mapping_policy','complete_only')=='complete_only']
        if len(pp)!=1:raise ValueError('Missing/ambiguous accepted panel: '+family)
        p=pp[0];common=primary(rs,list(p['methods']),p['stratum'],fraction)
        metric='ap' if rq=='RQ2' else 'pair_concordance'
        cluster='subject_group' if dataset=='TRACE' else ('pattern_id' if rq=='RQ2' else 'patient_id')
        c=(APContrast if rq=='RQ2' else PairContrast)(common,left,right,cluster)
        actual=c.evaluate(np.ones(c.n));old=accepted_contrast(p,left,right,metric)
        if len(common)!=p['common'] or len({r['patient_id'] for r in common})!=p['patients'] or sum(r['reference_failure'] for r in common)!=p['failures']:
            raise ValueError('Reference denominator gate failed')
        if old['difference'] is None or not np.isfinite(actual) or abs(actual-old['difference'])>1e-10:raise ValueError('Reference point estimate gate failed')
        if rq=='RQ3' and (c.n!=p['pair_patients'] or c.pairs!=p['discordant_pairs']):raise ValueError('Pair denominator gate failed')
        cs.append(c);meta.append(dict(dataset=dataset,rq=rq,family=dataset+'_'+rq,metric=metric,
            left=left,right=right,budget=fraction,cluster=cluster,**census(common),
            clusters=len(set(c.cluster)),pair_patients=c.n if rq=='RQ3' else None,
            discordant_pairs=c.pairs if rq=='RQ3' else None,original_1000_ci95=old['ci95'],
            original_difference=old['difference'],seed=seed))
    draws=resample(cs,reps,seed)
    rows=[dict(m,**s) for m,s in zip(meta,intervals(cs,draws))];diagnostics=[]
    if dataset=='DDX' and rq=='RQ2':
        for c,m in zip(cs,meta):
            old=resample([c],1000,261006)[:,0];new=resample([c],5000,261016)[:,0]
            valid=old[np.isfinite(old)];ci=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None
            if ci is None or not np.allclose(ci,m['original_1000_ci95'],atol=1e-10,rtol=0):raise ValueError('Original bootstrap reproduction gate failed')
            for label,d,sd in [('original_reproduced',old,261006),('new_seed_5000',new,261016)]:
                v=d[np.isfinite(d)];q=np.quantile(v,[.025,.975]).tolist() if len(v) else None
                diagnostics.append(dict(m,analysis=label,seed=sd,requested=len(d),valid=len(v),
                    zero_failure_draws=int(np.isnan(d).sum()),ci95=q,
                    ci_lower_change=q[0]-ci[0] if q else None,ci_upper_change=q[1]-ci[1] if q else None))
    return rows,diagnostics

def endpoint_row(rs,field,labels,meta):
    valid=[r for r in rs if r.get(field) is not None];by=defaultdict(list)
    outcomes=[]
    for r in valid:
        correct=int(labels[int(np.argmax(r[field]))]==r['reference_diagnosis'])
        outcomes.append(correct);by[r['patient_id']].append(correct)
    return dict(meta,records=len(valid),patients=len(by),unavailable=len(rs)-len(valid),
                failures=len(valid)-sum(outcomes),failure_patients=sum(any(x==0 for x in v) for v in by.values()),
                accuracy_record_weighted=float(np.mean(outcomes)) if outcomes else None,
                accuracy_patient_mean=float(np.mean([np.mean(v) for v in by.values()])) if by else None)

def absolute_endpoints(rs,labels,dataset,rq,background):
    out=[]
    for probe in ['encoder','lightgbm']:
        allrows=[r for r in rs if r['probe']==probe]
        if rq=='RQ2':
            specs=[('all_mapped',lambda r:r['mapping_available']),
                   ('both_full_correct_all_proposals',lambda r:r['mapping_available'] and r['both_full_correct']),
                   ('both_full_correct_supported_correct',lambda r:r['mapping_available'] and r['both_full_correct'] and r['target_supported'] and r['diagnosis_correct'])]
        else:specs=[('both_full_correct_budget_'+str(b),lambda r,b=b:r['fraction']==b and r['both_full_correct']) for b in [.25,.5,.75]]
        for name,fn in specs:
            rr=[r for r in allrows if fn(r)]
            out.append(endpoint_row(rr,'cited_probabilities',labels,dict(dataset=dataset,rq=rq,probe=probe,population=name,input='H+E')))
            ids={r['patient_id'] for r in rr};by={r['patient_id']:r for r in allrows if r['patient_id'] in ids}
            for field,label in [('full_probabilities','H+F'),('context_probabilities','H')]:
                pp=[dict(r,context_probabilities=background.get((pid,probe))) for pid,r in by.items()]
                out.append(endpoint_row(pp,field,labels,dict(dataset=dataset,rq=rq,probe=probe,population=name+'_same_patients',input=label)))
        if rq=='RQ2':
            by={}
            for r in allrows:
                if r['patient_id'] in by:
                    previous=by[r['patient_id']]['full_probabilities'];current=r['full_probabilities']
                    if (previous is None)!=(current is None) or (current is not None and not np.allclose(previous,current,atol=1e-6,rtol=0)):raise ValueError('Repeated full probabilities disagree')
                by[r['patient_id']]=r
            for field,label in [('full_probabilities','H+F'),('context_probabilities','H')]:
                out.append(endpoint_row([dict(r,context_probabilities=background.get((pid,probe))) for pid,r in by.items()],field,labels,
                                        dict(dataset=dataset,rq=rq,probe=probe,population='all_test_before_full_correct_filter',input=label)))
    return out
