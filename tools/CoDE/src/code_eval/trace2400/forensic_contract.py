"""Diagnostic-only independent arithmetic; no acceptance thresholds are relaxed."""
def ap(y,s):
    if not y or not sum(y):return None
    levels=sorted(set(s),reverse=True);tp=0;count=0;value=0.
    for v in levels:
        ix=[i for i,x in enumerate(s) if x==v];pos=sum(y[i] for i in ix)
        tp+=pos;count+=len(ix);value+=pos/sum(y)*tp/count
    return value

def coverage(weights,groups,cited):
    denominator=sum(weights.values())
    if denominator<=0:return None
    return sum(w*sum(t in cited for t in groups[g])/len(groups[g]) for g,w in weights.items())/denominator

def sets_from_annotations(chain):
    key=chain['key'];task=chain['task'];facts={f['fact_id']:f for f in task['patient_facts']}
    aliases=key['fact_alias_to_original']
    if set(facts)!=set(aliases) or len(set(aliases.values()))!=len(aliases):raise ValueError('Alias identity mismatch')
    result={}
    for who,ann in chain['reviews'].items():
        if len(ann)!=len(facts) or {a['fact'] for a in ann}!=set(facts):raise ValueError('Annotation census')
        for a in ann:
            if a['case']!=key['audit_id']:raise ValueError('Cross-case annotation')
            if a['quote'] and a['quote'] not in task['explanation']:raise ValueError('Quote mismatch')
            if a['decision'] in ['明确一致','语义一致','部分一致','不相容'] and not a['quote']:raise ValueError('Missing quote')
        for policy,allowed in [('explicit',{'明确一致'}),('compatible',{'明确一致','语义一致'})]:
            result[who+'_'+policy]=sorted(aliases[a['fact']] for a in ann if a['decision'] in allowed)
    return result
