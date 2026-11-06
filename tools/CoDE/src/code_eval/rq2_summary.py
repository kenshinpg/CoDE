"""Natural-rationale summaries with explicit endpoints and population filters."""
from collections import Counter
from .proxy_metrics import summarize
from .rq2_audit import audit_panel

SOURCES=['cohd','encoder','grasp','lightgbm','tabr','no_retrieval','voting']
BASE=['fact_coverage','group_coverage']
FAMILIES={
 'coverage_sources':BASE+[s+'_coverage' for s in SOURCES[:5]],
 'tree_rules':BASE+['tree_gain','tree_positive_shap','tree_absolute_shap','tree_class_shap','lightgbm_coverage','lightgbm_absolute','lightgbm_retained'],
 'direct_support':BASE+[s+suffix for s in SOURCES[:5] for suffix in ['_absolute','_retained']],
 'retrieval':BASE+[s+suffix for s in SOURCES[4:] for suffix in ['_coverage','_absolute','_retained']],
 'quantity_only':BASE,
 'permutations':BASE+['tree_positive_shap']+[f'permuted_{i:02d}' for i in range(10)]}

def panel(records,probe,stratum,endpoint,family,bootstrap=1000):
    # Saved records retain original factual flags; this metric-only projection
    # supplies the chosen outcome to the unchanged legacy AP/pair estimator.
    source=[r for r in records if r['probe']==probe and r['mapping_available']
            and (stratum=='all' or r['diagnosis_correct']==(stratum=='correct'))]
    admitted=[r for r in source if r['both_full_correct'] and r['probe_available']
              and (family=='quantity_only' or r['target_supported'])]
    projected=[dict(r,full_correct=True,failure=r['reference_failure'] if endpoint=='reference' else r['proposal_failure']) for r in admitted]
    methods=FAMILIES[family]
    baselines=[m for m in ['fact_coverage','group_coverage','lightgbm_absolute','cohd_absolute'] if m in methods]
    if family.startswith('paired_'):baselines=methods
    out=summarize(projected,methods,bootstrap,261006,baseline_methods=baselines)
    common=[r for r in admitted if all(r['scores'].get(m) is not None for m in methods)]
    failure_key='reference_failure' if endpoint=='reference' else 'proposal_failure'
    failures=[r for r in common if r[failure_key]]
    out.update(probe=probe,stratum=stratum,endpoint=endpoint,family=family,input_mapped=len(source),admitted_before_common=len(admitted),
        failures=len(failures),failures_by_disease=dict(Counter(r['reference_diagnosis'] for r in failures)),
        records_by_component=dict(Counter(r['component'] for r in common)),
        interpretation='Reference recovery is not a clinical explanation label; proposal endpoint measures cross-model agreement, including agreement with wrong proposals. Secondary intervals are unadjusted.',
        self_source=probe,selection_filter='both full-input probes recover reference; common scores within named family')
    out['information_audit']=audit_panel(records,probe,stratum,family,methods,endpoint,bootstrap)
    if out['information_audit']['sequential_flow'][-1]['explanations'] != out['common']:
        raise ValueError('Audit population differs from scoring population')
    if out['information_audit']['pair_funnel']['equal_fact_count_discordant']['pairs'] != out['discordant_pairs']:
        raise ValueError('Audit pairs differ from scoring pairs')
    return out

def add_pairwise_families():
    # Direct comparisons use their own common records; they do not infer a
    # difference between methods from separate tests against a third baseline.
    comparisons=[('cohd_coverage','tree_positive_shap'),('tree_positive_shap','tree_class_shap')]
    for source in ['cohd_coverage','tree_positive_shap','tree_class_shap','tabr_coverage']:
        for baseline in ['group_coverage','lightgbm_absolute','cohd_absolute']:comparisons.append((source,baseline))
    for i,(left,right) in enumerate(comparisons):
        FAMILIES['paired_'+left+'_vs_'+right]=list(dict.fromkeys(BASE+[left,right]))
