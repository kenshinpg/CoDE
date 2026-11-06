"""Frozen finite-audit adjudication contrasts. No training or semantic decisions."""
from collections import Counter
import copy
import numpy as np
from .round1_followup import APContrast, intervals
from .proxy_metrics import average_precision

COMPARISONS=[('cohd_coverage','group_coverage'),('tree_positive_shap','group_coverage'),('tree_positive_shap','lightgbm_absolute')]
CORE_SCORES=sorted({m for pair in COMPARISONS for m in pair}|{'fact_coverage'})
PROBES=['encoder','lightgbm']

def validate_reference(records):
    keys=[(r['dataset'],r['explanation_id']) for r in records]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate adjudication record')
    for r in records:
        if r['status'] not in ['complete','unresolved'] or not r['common_omissions_checked'] or set(r['completed_by'])!={'R01','R02'}:raise ValueError('Unclosed adjudication')
        f=r['final_fact_ids'];u=r['unresolved_fact_ids']
        if len(f)!=len(set(f)) or set(f)&set(u) or (r['status']=='complete')!= (not u):raise ValueError('Unresolved reference contract')
        if not r['random_sample'] and not r['targeted_sample']:raise ValueError('Missing stratum')
    return {ds:dict(total=sum(r['dataset']==ds for r in records),complete=sum(r['dataset']==ds and r['status']=='complete' for r in records),unresolved=sum(r['dataset']==ds and r['status']=='unresolved' for r in records)) for ds in sorted({r['dataset'] for r in records})}

def make_ledger(reviews,original,facts):
    ledger=[];lookup={}
    for r in reviews:
        e=original[r['explanation_id']]
        if e['patient_id']!=r['patient_id'] or e['partition']!='test' or not e['reviewed_available']:raise ValueError('Review/ledger identity or availability changed')
        if set(e['cited_fact_ids'])!=set(r['automatic_fact_ids']):raise ValueError('Automatic E changed')
        if r['dataset']=='TRACE' and e['subject_group']!=r['subject_group']:raise ValueError('TRACE cluster changed')
        for v,field in [('automatic','automatic_fact_ids'),('adjudicated','final_fact_ids')]:
            if v=='adjudicated' and r['status']!='complete':continue
            fs=r[field]
            if not set(fs)<=facts[r['patient_id']]:raise ValueError('Citation outside frozen F')
            eid=r['explanation_id']+'::adjudication20261022::'+v
            ledger.append(dict(e,explanation_id=eid,cited_fact_ids=fs))
            lookup[eid]=(r,v)
    return ledger,lookup

def automatic_gate(row,old,tolerance=2e-5):
    for k in ['patient_id','diagnosis','reference_diagnosis','diagnosis_correct','probe_available','both_full_correct','reference_failure','proposal_failure','cited_fact_count','mapping_available','target_supported']:
        if row[k]!=old[k]:raise ValueError('Automatic replay endpoint drift: '+k)
    maximum=0.
    for m in CORE_SCORES:
        x,y=row['scores'].get(m),old['scores'].get(m)
        if (x is None)!=(y is None):raise ValueError('Automatic score availability drift: '+m)
        if x is not None:
            if not np.isfinite([x,y]).all():raise ValueError('Nonfinite score')
            maximum=max(maximum,abs(x-y))
            if abs(x-y)>tolerance:raise ValueError('Automatic core score drift: '+m)
    for key in ['full_probabilities','cited_probabilities']:
        x,y=row[key],old[key]
        if (x is None)!=(y is None):raise ValueError('Probability availability drift')
        if x is not None and (np.shape(x)!=np.shape(y) or not np.allclose(x,y,atol=tolerance,rtol=0)):raise ValueError('Automatic probability drift: '+key)
    return maximum

def cook(scored,lookup,old,labels):
    expected={(eid,p) for eid in lookup for p in PROBES};seen=set();out=[];delta=0.;checks=0
    for r in scored:
        k=(r['explanation_id'],r['probe'])
        if k not in expected or k in seen:raise ValueError('Unexpected or duplicate score row')
        seen.add(k);review,v=lookup[k[0]];eid=review['explanation_id']
        if r['patient_id']!=review['patient_id']:raise ValueError('Scored patient differs from frozen review')
        if review['dataset']=='TRACE' and r['subject_group']!=review['subject_group']:raise ValueError('Scored subject cluster differs from frozen review')
        if v=='automatic':delta=max(delta,automatic_gate(r,old[eid,r['probe']]));checks+=1
        pred=None
        if r['probe_available']:
            ps=np.asarray(r['cited_probabilities'],float)
            if ps.shape!=(len(labels),) or not np.isfinite(ps).all() or not np.isclose(ps.sum(),1.,atol=1e-5):raise ValueError('Invalid probabilities')
            pred=labels[int(ps.argmax())]
            if (pred!=r['reference_diagnosis'])!=r['reference_failure'] or (pred!=r['diagnosis'])!=r['proposal_failure']:raise ValueError('Endpoint/probability mismatch')
        out.append(dict(r,dataset=review['dataset'],explanation_id=eid,audit_id=review['audit_id'],version=v,reference_status=review['status'],random_sample=review['random_sample'],targeted_sample=review['targeted_sample'],fact_ids=review['automatic_fact_ids'] if v=='automatic' else review['final_fact_ids'],prediction=pred,scores={m:r['scores'].get(m) for m in CORE_SCORES}))
    if seen!=expected:raise ValueError('Missing score rows')
    by={(r['explanation_id'],r['probe'],r['version']):r for r in out}
    for r in out:
        if r['version']!='adjudicated':continue
        a=by[r['explanation_id'],r['probe'],'automatic']
        for key in ['both_full_correct','full_correct','reference_diagnosis','diagnosis','diagnosis_correct','target_supported']:
            if a[key]!=r[key]:raise ValueError('Full input/proposal changed across mappings')
        x,y=a['full_probabilities'],r['full_probabilities']
        if (x is None)!=(y is None) or (x is not None and not np.allclose(x,y,atol=1e-12,rtol=0)):raise ValueError('Full probabilities changed across mappings')
    return out,dict(status='passed',automatic_rows=checks,core_score_max_delta=delta,tolerance=2e-5,score_scope=CORE_SCORES)

def make_panels(records):
    by={(r['dataset'],r['explanation_id'],r['probe'],r['version']):r for r in records};panels=[];flows=[]
    for ds in ['DDXPlus','TRACE']:
      for group in ['random_sample','targeted_sample']:
       for probe in PROBES:
        for stratum in ['correct','incorrect','all']:
            meta=dict(dataset=ds,audit_group=group,probe=probe,proposal_stratum=stratum,cluster_unit='subject_group' if ds=='TRACE' else 'pattern_id',primary=probe=='encoder' and stratum=='correct')
            base=[r for r in records if r['dataset']==ds and r['version']=='automatic' and r['probe']==probe and r[group]]
            pairs=[]
            for a in base:
                b=by.get((ds,a['explanation_id'],probe,'adjudicated'))
                if b is not None:pairs.append((a,b))
            stages=[('reviewed',len(base)),('complete_reference',len(pairs))]
            for name,fn in [('proposal_stratum',lambda a,b:stratum=='all' or a['diagnosis_correct']==(stratum=='correct')),('both_full_correct',lambda a,b:a['both_full_correct'] and b['both_full_correct']),('mapping_available',lambda a,b:a['mapping_available'] and b['mapping_available']),('supported_target',lambda a,b:a['target_supported'] and b['target_supported']),('paired_probe_available',lambda a,b:a['probe_available'] and b['probe_available'])]:
                pairs=[(a,b) for a,b in pairs if fn(a,b)];stages.append((name,len(pairs)))
            for name,n in stages:flows.append(dict(meta,stage=name,explanations=n))
            comparisons=[]
            for left,right in COMPARISONS:
                needed=sorted({left,right,'fact_coverage','group_coverage'})
                pp=sorted([(a,b) for a,b in pairs if all(x['scores'].get(m) is not None for m in needed for x in [a,b])],key=lambda pair:pair[0]['explanation_id'])
                comparisons.append(dict(left=left,right=right,pairs=pp,eligible_before_common=len(pairs),needed=needed))
            panels.append(dict(meta,panel_id=len(panels),comparisons=comparisons))
    return panels,flows

def contrasts(panel):
    allcs=[]
    for kind in ['automatic','adjudicated_updated','adjudicated_fixed_endpoint']:
        for c in panel['comparisons']:
            rr=[copy.deepcopy(a if kind=='automatic' else b) for a,b in c['pairs']]
            if kind=='adjudicated_fixed_endpoint':
                for r,(a,b) in zip(rr,c['pairs']):r['reference_failure']=a['reference_failure']
            allcs.append(APContrast(rr,c['left'],c['right'],panel['cluster_unit']))
    return allcs

def bootstrap_block(task):
    panel,start,counts,keys=task;cs=contrasts(panel);lookup={k:i for i,k in enumerate(keys)}
    indices=[np.array([lookup[k] for k in c.cluster],int) for c in cs]
    out=np.full((len(counts),len(cs)),np.nan)
    for i,w in enumerate(counts):
        for j,(c,ix) in enumerate(zip(cs,indices)):out[i,j]=c.evaluate(w[ix])
    return panel['panel_id'],start,out

class Point:
    def __init__(self,p):self.p=p;self.n=0
    def evaluate(self,w):return self.p

def interval_rows(points,draws):
    return intervals([Point(p) for p in points],draws)

def finish_panel(panel,draws,reps,seed):
    cs=contrasts(panel);points=np.array([c.evaluate(np.ones(c.n)) for c in cs]);rows=[];trans=[]
    meta={k:v for k,v in panel.items() if k!='comparisons'}
    versions={}
    for off,name in [(0,'automatic'),(3,'adjudicated_updated'),(6,'adjudicated_fixed_endpoint')]:versions[name]=interval_rows(points[off:off+3],draws[:,off:off+3])
    for off,name in [(3,'change_updated'),(6,'change_fixed_endpoint')]:versions[name]=interval_rows(points[off:off+3]-points[:3],draws[:,off:off+3]-draws[:,:3])
    for j,c in enumerate(panel['comparisons']):
        pp=c['pairs'];a=[x for x,y in pp];b=[y for x,y in pp]
        q=dict(meta,left=c['left'],right=c['right'],common_explanations=len(pp),common_patients=len({x['patient_id'] for x in a}),common_clusters=len(set(cs[j].cluster)),eligible_before_common=c['eligible_before_common'],automatic_failures=sum(x['reference_failure'] for x in a),adjudicated_failures=sum(x['reference_failure'] for x in b),automatic_failure_patients=len({x['patient_id'] for x in a if x['reference_failure']}),adjudicated_failure_patients=len({x['patient_id'] for x in b if x['reference_failure']}),bootstrap_requested=reps,seed=seed,interpretation='Finite failure-targeted audit sensitivity; not population AP' if panel['audit_group']=='targeted_sample' else 'Finite random-audit subset sensitivity')
        for name,v in versions.items():
            for k,value in v[j].items():q[name+'_'+k]=value
        for name,rr in [('automatic',a),('adjudicated',b)]:
            y=[x['reference_failure'] for x in rr]
            for side in ['left','right']:q[name+'_'+side+'_ap']=average_precision(y,[-x['scores'][c[side]] for x in rr])
            q[name+'_outcome_classes']=len(set(y))
        rows.append(q)
        trans.append(dict(meta,left=c['left'],right=c['right'],common_explanations=len(pp),changed_sets=sum(set(x['fact_ids'])!=set(y['fact_ids']) for x,y in pp),added_facts=sum(len(set(y['fact_ids'])-set(x['fact_ids'])) for x,y in pp),removed_facts=sum(len(set(x['fact_ids'])-set(y['fact_ids'])) for x,y in pp),prediction_changed=sum(x['prediction']!=y['prediction'] for x,y in pp),success_to_failure=sum(not x['reference_failure'] and y['reference_failure'] for x,y in pp),failure_to_success=sum(x['reference_failure'] and not y['reference_failure'] for x,y in pp),automatic_failures=q['automatic_failures'],adjudicated_failures=q['adjudicated_failures'],automatic_failure_diseases=dict(Counter(x['reference_diagnosis'] for x in a if x['reference_failure'])),adjudicated_failure_diseases=dict(Counter(x['reference_diagnosis'] for x in b if x['reference_failure']))))
    return rows,trans
