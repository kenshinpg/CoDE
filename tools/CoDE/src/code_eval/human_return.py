"""Human explicit-set sensitivity on frozen F; no adjudication or fitting."""
import csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:return [json.loads(s) for s in f if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def emit(p,rs):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rs),encoding='utf-8')
def table(p,rs):
    fields=list(dict.fromkeys(k for r in rs for k in r))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rs:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def complete(r):
    return r['reported_completeness']=='COMPLETE' and not r['has_unresolved_or_incomplete'] and not r['unresolved_fact_ids']

def validate_reviews(rs):
    keys=set()
    for r in rs:
        key=r['dataset'],r['explanation_id']
        if key in keys:raise ValueError('Duplicate human review ID')
        keys.add(key)
        if r['adjudicated_E'] is not None:raise ValueError('Unexpected adjudication; this run uses human_return_exact_E only')
        for field in ['automatic_E','human_return_exact_E']:
            if not isinstance(r[field],list) or len(r[field])!=len(set(r[field])):raise ValueError('Unavailable or duplicate explicit set')
        if not any(r['sample_membership'].get(k) for k in ['random','targeted']):raise ValueError('Missing review membership')
        if set(r['human_return_exact_E'])&set(r['unresolved_fact_ids']):raise ValueError('EXACT overlaps unresolved')
    return {ds:dict(explanations=sum(r['dataset']==ds for r in rs),complete=sum(r['dataset']==ds and complete(r) for r in rs),
                   random=sum(r['dataset']==ds and r['sample_membership']['random'] for r in rs),
                   targeted=sum(r['dataset']==ds and r['sample_membership']['targeted'] for r in rs),
                   overlap=sum(r['dataset']==ds and all(r['sample_membership'].values()) for r in rs)) for ds in sorted({r['dataset'] for r in rs})}

def make_ledger(rs,original,facts):
    ledger=[];lookup={}
    for r in rs:
        e=original[r['explanation_id']];pid=r['patient_id']
        if e['patient_id']!=pid or e['partition']!='test':raise ValueError('Review patient/split mismatch')
        if set(e['cited_fact_ids'])!=set(r['automatic_E']) or not e['reviewed_available']:raise ValueError('Automatic E drift/unavailable')
        for version,key in [('automatic','automatic_E'),('human_exact','human_return_exact_E')]:
            if not set(r[key])<=facts[pid]:raise ValueError('Human or automatic set outside frozen F')
            eid=e['explanation_id']+'::human_return::'+version
            # The worker needs an available operational set; human completeness is kept separately.
            ledger.append(dict(e,explanation_id=eid,cited_fact_ids=r[key],reviewed_available=True))
            lookup[eid]=(r,version)
    return ledger,lookup

def score_drift_audit(records,lookup,old,tolerance=2e-5):
    """Inspect every automatic score before deciding whether results can be used."""
    details=[];methods={}
    for r in records:
        review,version=lookup[r['explanation_id']]
        if version!='automatic':continue
        prev=old[review['explanation_id'],r['probe']]
        for m in sorted(set(r['scores'])|set(prev['scores'])):
            x=r['scores'].get(m);y=prev['scores'].get(m)
            kind=('schema' if m not in r['scores'] or m not in prev['scores'] else
                  'availability' if (x is None)!=(y is None) else
                  'nonfinite' if x is not None and not (np.isfinite(x) and np.isfinite(y)) else
                  'numeric' if x is not None and abs(x-y)>tolerance else None)
            d=abs(x-y) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y) else None
            s=methods.setdefault(m,dict(rows=0,mismatch_rows=0,max_absolute_delta=0.))
            s['rows']+=1;s['mismatch_rows']+=int(kind is not None)
            if d is not None:s['max_absolute_delta']=max(s['max_absolute_delta'],d)
            if kind:details.append(dict(explanation_id=review['explanation_id'],patient_id=r['patient_id'],probe=r['probe'],method=m,
                kind=kind,original=y,recomputed=x,absolute_delta=d))
    return dict(tolerance=tolerance,methods=methods,mismatching_methods=sorted({r['method'] for r in details}),details=details)

def cook(records,lookup,old,labels,source_flags=None,quarantined_methods=(),numerical_reference_methods=()):
    quarantined_methods=set(quarantined_methods)
    numerical_reference_methods=set(numerical_reference_methods)
    allowed_reference={m+'_'+s for m in ['tabr','no_retrieval','grasp'] for s in ['coverage','absolute','retained']}
    if not numerical_reference_methods<=allowed_reference:raise ValueError('Unsupported numerical reference method')
    if not quarantined_methods<={'voting_coverage','voting_absolute','voting_retained'}:
        raise ValueError('Only the explicit Voting sensitivity quarantine is supported')
    out=[];seen=set();delta=0.;source_flags=source_flags or {}
    for r in records:
        if numerical_reference_methods:
            policy=r.get('numerical_reference',{})
            if policy.get('policy')!='CPU_float64_single_thread' or set(policy.get('methods',[]))!=numerical_reference_methods:
                raise ValueError('Numerical reference provenance missing')
        review,v=lookup[r['explanation_id']];eid=review['explanation_id'];k=(r['explanation_id'],r['probe'])
        if k in seen:raise ValueError('Duplicate score record')
        seen.add(k);prev=old[eid,r['probe']]
        if v=='automatic':
            for field in ['patient_id','reference_diagnosis','diagnosis','probe_available','both_full_correct','reference_failure','proposal_failure','cited_fact_count']:
                if r[field]!=prev[field]:raise ValueError('Automatic endpoint drift: '+field)
            if set(r['scores'])!=set(prev['scores']):raise ValueError('Score schema drift')
            for m,x in r['scores'].items():
                y=prev['scores'][m]
                if m in quarantined_methods:continue
                if m in numerical_reference_methods:
                    if x is not None and not np.isfinite(x):raise ValueError('Nonfinite numerical reference score')
                    continue
                if (x is None)!=(y is None):raise ValueError('Automatic score availability drift: '+m)
                if x is not None:
                    if not (np.isfinite(x) and np.isfinite(y)):raise ValueError('Nonfinite automatic score: '+m)
                    d=abs(x-y);delta=max(delta,d)
                    if d>2e-5:raise ValueError('Automatic score drift: '+m)
        pred=None
        if r['probe_available']:
            p=np.asarray(r['cited_probabilities'],float)
            if p.shape!=(len(labels),) or not np.isfinite(p).all() or not np.isclose(p.sum(),1,atol=1e-5):raise ValueError('Invalid prediction probabilities')
            pred=labels[int(p.argmax())]
            if (pred!=r['reference_diagnosis'])!=r['reference_failure']:raise ValueError('Prediction mismatch')
        factids=review['automatic_E'] if v=='automatic' else review['human_return_exact_E']
        flags=source_flags.get(review['patient_id'],{})
        retained_scores=dict(r['scores'])
        quarantined_scores={m:retained_scores[m] for m in quarantined_methods if m in retained_scores}
        for m in quarantined_scores:retained_scores[m]=None
        out.append(dict(r,scores=retained_scores,quarantined_raw_scores=quarantined_scores,
            score_quarantine_reason='automatic baseline replay mismatch; entire Voting family withheld for both mappings' if quarantined_scores else None,
            explanation_id=eid,dataset=review['dataset'],audit_id=review['audit_id'],version=v,
            sample_membership=review['sample_membership'],human_complete=complete(review),
            human_reported_completeness=review['reported_completeness'],unresolved_fact_ids=review['unresolved_fact_ids'],
            fact_ids=factids,added_fact_ids=sorted(set(review['human_return_exact_E'])-set(review['automatic_E'])),
            removed_fact_ids=sorted(set(review['automatic_E'])-set(review['human_return_exact_E'])),prediction=pred,
            source_F_error_fact_ids=sorted(flags.get('ERROR_CONFIRMED',[])),source_F_unresolved_fact_ids=sorted(flags.get('UNRESOLVED',[])),
            cited_source_error_fact_ids=sorted(set(factids)&set(flags.get('ERROR_CONFIRMED',[]))),
            cited_source_unresolved_fact_ids=sorted(set(factids)&set(flags.get('UNRESOLVED',[]))),
            interpretation='human explicit-set sensitivity conditional on frozen F; not final adjudication'))
    expected={(eid,p) for eid in lookup for p in ['encoder','lightgbm']}
    if seen!=expected:raise ValueError('Missing scoring rows')
    return out,dict(automatic_rows=sum(v=='automatic' for _,v in lookup.values())*2,max_score_delta=delta,tolerance=2e-5,
        quarantined_methods=sorted(quarantined_methods),numerical_reference_methods=sorted(numerical_reference_methods),
        interpretation='separate numerical-reference sensitivity; not a reproduction of formal scores' if numerical_reference_methods else 'automatic replay check')

def ap_draws(y,s,w):
    """Tie-aware weighted AP; one-class draws are excluded from discrimination inference."""
    y=np.asarray(y,float);s=np.asarray(s,float)
    ix=np.argsort(-s,kind='stable');ends=np.r_[np.where(s[ix][:-1]!=s[ix][1:])[0],len(s)-1]
    if not len(y):return np.full(len(w),np.nan)
    ww=w[:,ix];tp=np.cumsum(ww*y[ix],axis=1)[:,ends];total=np.cumsum(ww,axis=1)[:,ends]
    pos=w@y;neg=w.sum(1)-pos
    precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
    recall=np.divide(tp,pos[:,None],out=np.zeros_like(tp),where=pos[:,None]>0)
    v=(np.diff(np.c_[np.zeros(len(w)),recall],axis=1)*precision).sum(1)
    v[(pos==0)|(neg==0)]=np.nan
    return v

def weights(pairs,reps,seed,dataset):
    field='subject_group' if dataset=='TRACE' else 'pattern_id'
    keys=sorted({a[field] for a,b in pairs});ids={k:i for i,k in enumerate(keys)}
    ix=np.array([ids[a[field]] for a,b in pairs],int)
    if not keys:return np.empty((reps+1,0)),0
    rng=np.random.default_rng(seed);draw=rng.integers(0,len(keys),size=(reps,len(keys)))
    counts=np.zeros((reps,len(keys)))
    for j in range(reps):counts[j]=np.bincount(draw[j],minlength=len(keys))
    return np.vstack([np.ones(len(pairs)),counts[:,ix]]),len(keys)

def interval(v):
    good=v[1:][np.isfinite(v[1:])];point=float(v[0]) if np.isfinite(v[0]) else None
    ci=np.quantile(good,[.025,.975]).tolist() if len(good)>=2 and point is not None else [None,None]
    return dict(estimate=point,ci_low=ci[0],ci_high=ci[1],valid_replicates=len(good),undefined_replicates=len(v)-1-len(good),
                status='available' if point is not None and len(good)==len(v)-1 else ('conditional_on_defined_resamples' if point is not None and len(good)>=2 else 'not_estimable'))

def select_pairs(records):
    by=defaultdict(dict)
    for r in records:by[r['explanation_id'],r['probe']][r['version']]=r
    if any(set(v)!={'automatic','human_exact'} for v in by.values()):raise ValueError('Unpaired mapping versions')
    return [(v['automatic'],v['human_exact']) for v in by.values()]

def summarize(records,reps=1000,seed=261006):
    pairs=select_pairs(records);stats=[];transitions=[]
    for dataset in sorted({a['dataset'] for a,b in pairs}):
      for group in ['random','targeted']:
       for subset in ['all_reviewed','COMPLETE']:
        for scope in ['review_scope','formal_RQ2_scope']:
         for stratum in ['all','correct','incorrect']:
          for probe in ['encoder','lightgbm']:
            pp=[(a,b) for a,b in pairs if a['dataset']==dataset and a['probe']==probe and a['sample_membership'][group]
                and (subset=='all_reviewed' or b['human_complete']) and (scope=='review_scope' or a['both_full_correct'])
                and (stratum=='all' or a['diagnosis_correct']==(stratum=='correct'))]
            meta=dict(dataset=dataset,audit_group=group,human_subset=subset,analysis_scope=scope,proposal_stratum=stratum,probe=probe,
                interpretation='fixed existing F',paper_target='RQ2 natural explanations / mapping-sensitivity appendix')
            common=[(a,b) for a,b in pp if a['probe_available'] and b['probe_available'] and a['mapping_available'] and b['mapping_available']]
            transitions.append(dict(meta,row_type='probe',method='',reviewed_explanations=len(pp),reviewed_patients=len({a['patient_id'] for a,b in pp}),
                common_explanations=len(common),common_patients=len({a['patient_id'] for a,b in common}),
                changed_sets=sum(set(a['fact_ids'])!=set(b['fact_ids']) for a,b in pp),
                added_facts=sum(len(b['added_fact_ids']) for a,b in pp),removed_facts=sum(len(b['removed_fact_ids']) for a,b in pp),
                automatic_unscorable=sum(not a['probe_available'] for a,b in pp),human_unscorable=sum(not b['probe_available'] for a,b in pp),
                available_to_unavailable=sum(a['probe_available'] and not b['probe_available'] for a,b in pp),
                unavailable_to_available=sum(not a['probe_available'] and b['probe_available'] for a,b in pp),
                both_unavailable=sum(not a['probe_available'] and not b['probe_available'] for a,b in pp),
                prediction_changed=sum(a['prediction']!=b['prediction'] for a,b in common),
                success_to_failure=sum(not a['reference_failure'] and b['reference_failure'] for a,b in common),
                failure_to_success=sum(a['reference_failure'] and not b['reference_failure'] for a,b in common),
                automatic_failures=sum(a['reference_failure'] for a,b in common),human_failures=sum(b['reference_failure'] for a,b in common),
                automatic_failure_patients=len({a['patient_id'] for a,b in common if a['reference_failure']}),
                human_failure_patients=len({b['patient_id'] for a,b in common if b['reference_failure']}),
                source_F_error_patients=len({a['patient_id'] for a,b in pp if a['source_F_error_fact_ids']})))
            methods=sorted({m for a,b in pairs if a['dataset']==dataset for m in a['scores'] if m!='words' and not m.startswith('permuted_')})
            comparisons=[(m,None) for m in methods]
            comparisons += [(m,ref) for m in ['cohd_coverage','tree_positive_shap','tree_class_shap','tabr_coverage'] for ref in ['group_coverage','lightgbm_absolute','cohd_absolute'] if m in methods and ref in methods]
            comparisons += [(m,n) for m,n in [('cohd_coverage','tree_positive_shap'),('tree_positive_shap','tree_class_shap')] if m in methods and n in methods]
            for m,ref in comparisons:
                needed=[m]+([ref] if ref else [])
                qq=[(a,b) for a,b in common if (all(x in ['fact_coverage','group_coverage'] for x in needed) or a['target_supported'])
                    and all(a['scores'].get(x) is not None and b['scores'].get(x) is not None for x in needed)]
                if ref is None:
                    transitions.append(dict(meta,row_type='score',method=m,reviewed_explanations=len(pp),
                        automatic_unscorable=sum(a['scores'].get(m) is None for a,b in pp),human_unscorable=sum(b['scores'].get(m) is None for a,b in pp),
                        common_explanations=sum(a['scores'].get(m) is not None and b['scores'].get(m) is not None for a,b in pp)))
                w,ncluster=weights(qq,reps,seed,dataset)
                y0=[a['reference_failure'] for a,b in qq];y1=[b['reference_failure'] for a,b in qq]
                s0=-np.array([a['scores'][m] for a,b in qq]);s1=-np.array([b['scores'][m] for a,b in qq])
                for endpoint,y in [('fixed_original',y0),('updated_human',y1)]:
                    auto=ap_draws(y0,s0,w);human=ap_draws(y,s1,w)
                    if ref:
                        auto-=ap_draws(y0,[-a['scores'][ref] for a,b in qq],w)
                        human-=ap_draws(y,[-b['scores'][ref] for a,b in qq],w)
                    diff=interval(human-auto);aa=interval(auto);hh=interval(human)
                    rec=dict(meta,method=m,reference_method=ref or '',metric='failure_AP_difference' if ref else 'failure_AP',endpoint_policy=endpoint,
                        common_explanations=len(qq),common_patients=len({a['patient_id'] for a,b in qq}),clusters=ncluster,
                        automatic_failures=sum(y0),human_failures=sum(y1),
                        automatic_failure_patients=len({a['patient_id'] for a,b in qq if a['reference_failure']}),
                        human_failure_patients=len({b['patient_id'] for a,b in qq if b['reference_failure']}),
                        automatic_unestimable_reason=('no_common_records' if not qq else ('single_outcome_class' if len(set(y0))<2 else '')),
                        human_unestimable_reason=('no_common_records' if not qq else ('single_outcome_class' if len(set(y))<2 else '')),
                        source_F_error_patients=len({a['patient_id'] for a,b in qq if a['source_F_error_fact_ids']}),
                        bootstrap_repetitions=reps,bootstrap_seed=seed,cluster_unit='subject_group' if dataset=='TRACE' else 'pattern_id')
                    for prefix,val in [('automatic',aa),('human',hh),('human_minus_automatic',diff)]:rec.update({prefix+'_'+k:v for k,v in val.items()})
                    if not ref and qq:
                        ds=-s1+s0;mean=w@ds/w.sum(1);rec.update({'mean_score_change_'+k:v for k,v in interval(mean).items()})
                    stats.append(rec)
    return stats,transitions
