"""Full-match to legacy RQ2 contract. No model calls or outcome-dependent matching."""
from collections import Counter

def index(records,key):
    out={}
    for r in records:
        if r[key] in out:raise ValueError('Duplicate key: '+key)
        out[r[key]]=r
    return out

def census(rs):
    return dict(explanations=len(rs),admissions=len({r['patient_id'] for r in rs}),subjects=len({r['subject_group'] for r in rs}))

def adapt(m,g,p,s,labels):
    if m['task_id']!=g['task_id'] or m['patient_id']!=g['patient_id'] or p['patient_id']!=m['patient_id']:raise ValueError('Task/patient join mismatch')
    if m['input_view_id']!=p['view_id'] or m['partition']!=s['partition']:raise ValueError('View/split mismatch')
    if m['generation_status']!=g['status']:raise ValueError('Generation status mismatch')
    status=m['visible_fact_scan_status'];valid=g['status']=='complete'
    if status not in ['COMPLETE','PARTIAL','UNAVAILABLE'] or valid==(status=='UNAVAILABLE'):raise ValueError('Scan/generation availability mismatch')
    facts=index(p['facts'],'fact_id');ds=m['decisions'];cit=m['operational_exact_fact_ids']
    text=g.get('parsed_explanation');diagnoses=g.get('parsed_diagnoses')
    if valid:
        if not isinstance(text,str) or not text.strip() or not isinstance(diagnoses,list) or len(diagnoses)!=3 or len(set(diagnoses))!=3 or not set(diagnoses)<=set(labels):raise ValueError('Invalid completed generation')
        decisions=index(ds,'fact_id')
        if set(decisions)!=set(facts):raise ValueError('Incomplete visible fact census')
        allowed={'exact','partial','contradiction','ambiguous','not_referenced','non_patient'}
        for d in ds:
            if d['status'] not in allowed:raise ValueError('Unknown matching decision')
            if d['status'] in {'exact','partial','contradiction','non_patient'} and not d['quotes']:raise ValueError('Missing supporting span')
            for q in d['quotes']:
                if not isinstance(q['start'],int) or not isinstance(q['end'],int) or not 0<=q['start']<q['end']<=len(text) or text[q['start']:q['end']]!=q['text']:raise ValueError('Invalid verbatim quote')
        exact={d['fact_id'] for d in ds if d['status']=='exact'}
        if not isinstance(cit,list) or len(set(cit))!=len(cit) or set(cit)!=exact:raise ValueError('Exact set mismatch')
        if (status=='PARTIAL')!=any(d['status']=='ambiguous' for d in ds):raise ValueError('Ambiguity status mismatch')
    elif cit is not None or ds:raise ValueError('Unavailable must not become empty evidence')
    diagnosis=diagnoses[0] if valid else None
    return dict(explanation_id=m['task_id'],patient_id=m['patient_id'],subject_group=s['subject_group'],partition=s['partition'],
        reference_diagnosis=s['reference_class'],diagnosis=diagnosis,diagnosis_correct=diagnosis==s['reference_class'],target_supported=diagnosis in labels,
        reviewed_available=valid,repair_status=status,cited_fact_ids=cit,word_count=len(text.split()) if valid else None,
        annotation_protocols=m.get('annotation_protocols',[]),decision_counts=dict(Counter(d['status'] for d in ds)))

def policy_records(records,policy):
    if policy not in ['complete_only','include_partial_exact']:raise ValueError('Unknown mapping policy')
    return [dict(r,mapping_available=r['mapping_available'] and (policy=='include_partial_exact' or r['scan_status']=='COMPLETE')) for r in records]
