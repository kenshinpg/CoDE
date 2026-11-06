"""Paired diagnostic-proxy evaluation. Labels enter here, never score inputs."""
from collections import defaultdict
import itertools
import numpy as np
from .resampling import cluster_indices


def average_precision(y, score):
    y=np.asarray(y,dtype=int);score=np.asarray(score,dtype=float)
    if not len(y) or not y.sum():return None
    if not np.isfinite(score).all():raise ValueError('Nonfinite score')
    order=np.argsort(-score,kind='stable');s=score[order];z=y[order]
    ends=np.r_[np.where(s[:-1]!=s[1:])[0],len(s)-1]
    tp=np.cumsum(z)[ends];recall=tp/z.sum();precision=tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))


def summarize(records, methods, bootstrap=500, seed=261006, *, baseline_methods=('fact_coverage','absolute_support')):
    """Same cases and paired patient-pattern resamples for every method."""
    eligible=[r for r in records if r['full_correct'] and r['probe_available']]
    common=[r for r in eligible if all(r['scores'].get(m) is not None for m in methods)]
    result=dict(eligible=len(eligible),common=len(common),patients=len({r['patient_id'] for r in common}),
        failure_prevalence=sum(r['failure'] for r in common)/len(common) if common else None,methods={})
    groups=defaultdict(list)
    for i,r in enumerate(common):groups[r['subject_group']].append(i)
    keys=sorted(groups);rng=np.random.default_rng(seed)
    y=np.array([r['failure'] for r in common],dtype=int)
    predictions={m:np.array([-r['scores'][m] for r in common],dtype=float) for m in methods}
    sampled={m:[] for m in methods}
    for _ in range(bootstrap if common else 0):
        indices=np.concatenate([groups[keys[k]] for k in rng.integers(0,len(keys),len(keys))])
        for m in methods:sampled[m].append(average_precision(y[indices],predictions[m][indices]))
    for m in methods:
        values=[v for v in sampled[m] if v is not None]
        diseases=sorted({r['reference_diagnosis'] for r in common})
        macro=[]
        for disease in diseases:
            ix=[i for i,r in enumerate(common) if r['reference_diagnosis']==disease]
            ap=average_precision(y[ix],predictions[m][ix])
            if ap is not None:macro.append(ap)
        result['methods'][m]=dict(ap=average_precision(y,predictions[m]),
            ci95=list(map(float,np.quantile(values,[.025,.975]))) if values else None,
            valid_bootstrap_replicates=len(values),
            disease_macro_ap=float(np.mean(macro)) if macro else None,
            diseases_with_defined_ap=len(macro),diseases_total=len(diseases),
            unavailable=sum(r['scores'].get(m) is None for r in eligible),
            unavailable_rate=sum(r['scores'].get(m) is None for r in eligible)/len(eligible) if eligible else None)
    result['paired_ap_differences']={}
    for baseline in baseline_methods:
        if baseline not in methods:continue
        for m in methods:
            if m==baseline:continue
            ds=[a-b for a,b in zip(sampled[m],sampled[baseline]) if a is not None and b is not None]
            a=result['methods'][m]['ap'];b=result['methods'][baseline]['ap']
            result['paired_ap_differences'][m+' minus '+baseline]=dict(
                difference=a-b if a is not None and b is not None else None,
                ci95=list(map(float,np.quantile(ds,[.025,.975]))) if ds else None)
    # Same patient, proposed diagnosis, fact count. Only discordant probe outcomes.
    patient_pairs=defaultdict(list)
    by_patient=defaultdict(list)
    for r in common:by_patient[r['patient_id']].append(r)
    for pid,rs in by_patient.items():
        for a,b in itertools.combinations(rs,2):
            if a['diagnosis']!=b['diagnosis'] or a['cited_fact_count']!=b['cited_fact_count'] or a['failure']==b['failure']:continue
            good,bad=(a,b) if not a['failure'] else (b,a)
            patient_pairs[pid].append({m:(.5 if abs(good['scores'][m]-bad['scores'][m])<=1e-12 else float(good['scores'][m]>bad['scores'][m])) for m in methods})
    result['pair_patients']=len(patient_pairs);result['discordant_pairs']=sum(map(len,patient_pairs.values()))
    subject_by_id={r['patient_id']:r['subject_group'] for r in common}
    pair_groups=[subject_by_id[pid] for pid in patient_pairs]
    for m in methods:
        perpatient=[np.mean([r[m] for r in rs]) for rs in patient_pairs.values()]
        result['methods'][m]['pair_concordance']=float(np.mean(perpatient)) if perpatient else None
        pair_rng=np.random.default_rng(seed)
        pair_boot=[float(np.mean(np.asarray(perpatient)[cluster_indices(pair_groups,pair_rng)]))
                   for _ in range(bootstrap)] if perpatient else []
        result['methods'][m]['pair_ci95']=list(map(float,np.quantile(pair_boot,[.025,.975]))) if pair_boot else None
    result['interpretation']='exploratory development, common-case comparisons; no multiplicity correction; not clinical omission labels'
    return result
