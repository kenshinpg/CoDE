"""Reviewed citation-set ledger contracts; scoring remains in TRACE RQ2 worker."""
VERSIONS=['automatic','R01_explicit','R01_compatible','R02_explicit','R02_compatible']
def make_ledger(reviews,original):
    output=[];lookup={}
    if len({r['explanation_id'] for r in reviews})!=len(reviews):raise ValueError('Duplicate review')
    for r in reviews:
        e=original[r['explanation_id']]
        if e['patient_id']!=r['patient_id'] or e['subject_group']!=r['subject_group'] or e['partition']!='test':raise ValueError('Review identity drift')
        if e['repair_status']!='COMPLETE' or set(e['cited_fact_ids'])!=set(r['automatic_fact_ids']):raise ValueError('Automatic set drift')
        if r['random_sample']==r['targeted_sample']:raise ValueError('Disjoint review strata required')
        for who in ['R01','R02']:
            if not set(r[who+'_explicit'])<=set(r[who+'_compatible']):raise ValueError('Policy nesting')
        for v in VERSIONS:
            facts=r['automatic_fact_ids'] if v=='automatic' else r[v]
            if len(facts)!=len(set(facts)):raise ValueError('Duplicate fact')
            eid=e['explanation_id']+'::review100::'+v
            output.append(dict(e,explanation_id=eid,cited_fact_ids=facts))
            lookup[eid]=(r,v)
    return output,lookup

def check_baseline(r,old,tol=2e-5):
    for k in ['patient_id','diagnosis','reference_diagnosis','probe_available','both_full_correct','reference_failure','proposal_failure','cited_fact_count']:
        if r[k]!=old[k]:raise ValueError('Baseline endpoint drift: '+k)
    if set(r['scores'])!=set(old['scores']):raise ValueError('Score keys drift')
    maximum=0.
    for k,v in r['scores'].items():
        prev=old['scores'][k]
        if (v is None)!=(prev is None):raise ValueError('Score availability drift: '+k)
        if v is not None:
            delta=abs(v-prev);maximum=max(maximum,delta)
            if delta>tol:raise ValueError('Baseline score drift: '+k)
    return maximum
