"""Reporting-only additions; no change to score inputs or primary populations."""
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
from .proxy_metrics import average_precision


def census(rows):
    return dict(explanations=len(rows), patients=len({r['patient_id'] for r in rows}))


def audit_panel(records, probe, stratum, family, methods, endpoint, bootstrap=1000):
    rs=[r for r in records if r['probe']==probe]
    flow=[dict(stage='raw', **census(rs))]
    filters=[('mapping_available',lambda r:r['mapping_available']),
             ('target_supported',lambda r:family=='quantity_only' or r['target_supported']),
             ('probe_available',lambda r:r['probe_available']),
             ('both_full_correct',lambda r:r['both_full_correct']),
             ('proposal_stratum',lambda r:stratum=='all' or r['diagnosis_correct']==(stratum=='correct')),
             ('common_scores',lambda r:all(r['scores'].get(m) is not None for m in methods))]
    for name,predicate in filters:
        before=census(rs);rs=[r for r in rs if predicate(r)];after=census(rs)
        flow.append(dict(stage=name,**after,excluded_explanations=before['explanations']-after['explanations'],
                         patients_no_longer_represented=before['patients']-after['patients']))
    key='reference_failure' if endpoint=='reference' else 'proposal_failure'
    failed=[r for r in rs if r[key]];success=[r for r in rs if not r[key]]
    diseases={}
    for d in sorted({r['reference_diagnosis'] for r in rs}):
        dr=[r for r in rs if r['reference_diagnosis']==d]
        diseases[d]=dict(total=census(dr),failure=census([r for r in dr if r[key]]))
    by=defaultdict(list)
    for r in rs:by[r['patient_id']].append(r)
    stages={n:dict(pairs=0,patients=set()) for n in ['same_diagnosis','discordant','equal_fact_count_discordant']}
    per_patient=[]
    for pid,pr in sorted(by.items()):
        strict=[]
        for a,b in combinations(pr,2):
            if a['diagnosis']!=b['diagnosis']:continue
            stages['same_diagnosis']['pairs']+=1;stages['same_diagnosis']['patients'].add(pid)
            if a[key]==b[key]:continue
            stages['discordant']['pairs']+=1;stages['discordant']['patients'].add(pid)
            if a['cited_fact_count']!=b['cited_fact_count']:continue
            stages['equal_fact_count_discordant']['pairs']+=1;stages['equal_fact_count_discordant']['patients'].add(pid)
            good,bad=(a,b) if not a[key] else (b,a)
            strict.append([.5 if abs(good['scores'][m]-bad['scores'][m])<=1e-12 else float(good['scores'][m]>bad['scores'][m]) for m in methods])
        if strict:per_patient.append(np.mean(strict,axis=0))
    pair_differences={}
    if family.startswith('paired_'):
        values=np.asarray(per_patient);rng=np.random.default_rng(261006)
        draws=np.array([values[rng.integers(0,len(values),len(values))].mean(axis=0) for _ in range(bootstrap)]) if len(values) and bootstrap else None
        for i,j in combinations(range(len(methods)),2):
            ds=draws[:,i]-draws[:,j] if draws is not None else None
            pair_differences[methods[i]+' minus '+methods[j]]=dict(
                difference=float((values[:,i]-values[:,j]).mean()) if len(values) else None,
                ci95=np.quantile(ds,[.025,.975]).tolist() if ds is not None else None)
    leave_one=[]
    if family.startswith('paired_') and stratum=='correct' and endpoint=='reference':
        # All represented diseases, not only ones whose deletion improves results.
        for d in diseases:
            retained=[r for r in rs if r['reference_diagnosis']!=d]
            nf=sum(bool(r[key]) for r in retained);ns=len(retained)-nf
            aps={m:average_precision([r[key] for r in retained],[-r['scores'][m] for r in retained]) if nf and ns else None for m in methods}
            leave_one.append(dict(removed_disease=d,**census(retained),failure=census([r for r in retained if r[key]]),success_explanations=ns,
                ap=aps,ap_differences={a+' minus '+b:aps[a]-aps[b] if aps[a] is not None and aps[b] is not None else None for a,b in combinations(methods,2)}))
    fp=census(failed)['patients']
    return dict(sequential_flow=flow,success=census(success),failure=census(failed),
        mixed_outcome_patients=len({r['patient_id'] for r in failed}&{r['patient_id'] for r in success}),
        has_outcome_contrast=bool(failed and success),failure_diseases=sum(v['failure']['explanations']>0 for v in diseases.values()),
        largest_failure_disease_share_explanations=max((v['failure']['explanations'] for v in diseases.values()),default=0)/len(failed) if failed else None,
        largest_failure_disease_share_patients=max((v['failure']['patients'] for v in diseases.values()),default=0)/fp if fp else None,
        by_disease=diseases,by_component={c:census([r for r in rs if r['component']==c]) for c in sorted({r['component'] for r in rs})},
        pair_funnel={k:dict(pairs=v['pairs'],patients=len(v['patients'])) for k,v in stages.items()},
        paired_concordance_differences=pair_differences,leave_one_disease=leave_one,
        sensitivity_note='Leave-one-disease is descriptive, uses each fixed common panel, excludes every represented disease in turn, and has no significance claim. Pair differences use patient bootstrap; intervals unadjusted.',
        semantic_audit='not performed; machine mapping availability is not semantic correctness')
