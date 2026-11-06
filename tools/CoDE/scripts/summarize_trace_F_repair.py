"""Replay-gated repair contrasts. Cached unaffected patients retain their values."""
import copy, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from prepare_trace_F_repair import rows,read,emit,table,dump
from code_eval.round1_followup import APContrast,PairContrast,resample,intervals,primary,census
from code_eval.trace2400.rq1_summary import summarize as summarize_rq1,interval as rq1_interval
from code_eval.trace2400.natural_contract import policy_records

COMPARISONS=[('cohd_coverage','group_coverage'),('tree_positive_shap','group_coverage'),
             ('tree_positive_shap','lightgbm_absolute')]
PROB_TOL=2e-5


def keyed(rs,fields):
    d={tuple(r[k] for k in fields):r for r in rs}
    if len(d)!=len(rs):raise ValueError('Duplicate output identity')
    return d


def merge(old,new,affected):
    if not {r['patient_id'] for r in new}<=affected:raise ValueError('Patch includes an unaffected admission')
    return [r for r in old if r['patient_id'] not in affected]+new


def family(method):
    for n in ['cohd','encoder','grasp','lightgbm','tabr','no_retrieval']:
        if method.startswith(n+'_'):return [n+s for s in ['_coverage','_absolute','_retained']]
    return [method]


def number_equal(x,y,tol=PROB_TOL):
    if x is None or y is None:return x is y
    return bool(np.allclose(x,y,rtol=0,atol=tol))


def score_gate(old,replayed,rq):
    oldindex=keyed(old,['explanation_id','probe']);bad=set();audit=[];errors=[]
    affected={r['patient_id'] for r in replayed}
    expected={k for k,r in oldindex.items() if r['patient_id'] in affected}
    if expected!=set(keyed(replayed,['explanation_id','probe'])):raise ValueError('Incomplete original replay census')
    flags=['patient_id','subject_group','diagnosis','reference_diagnosis','both_full_correct','full_correct',
           'mapping_available','probe_available','target_supported','reference_failure','proposal_failure','cited_tokens']
    maxima=defaultdict(float);mismatches=defaultdict(int)
    for b in replayed:
        key=(b['explanation_id'],b['probe'])
        if key not in oldindex:raise ValueError('Replay identity outside archive')
        a=oldindex[key]
        for k in flags:
            if a.get(k)!=b.get(k):errors.append(dict(rq=rq,key=key,field=k,issue='identity_or_endpoint_drift'))
        for k in ['full_probabilities','cited_probabilities']:
            if not number_equal(a.get(k),b.get(k)):errors.append(dict(rq=rq,key=key,field=k,issue='probe_probability_drift'))
        if a['scores'].keys()!=b['scores'].keys():errors.append(dict(rq=rq,key=key,field='scores',issue='score_schema_drift'))
        for m in a['scores'].keys()|b['scores'].keys():
            x=a['scores'].get(m);y=b['scores'].get(m)
            if x is not None and y is not None:maxima[m]=max(maxima[m],abs(x-y))
            if not number_equal(x,y):bad.update(family(m));mismatches[m]+=1
    for m in sorted(maxima.keys()|mismatches.keys()):
        audit.append(dict(rq=rq,method=m,max_delta=maxima[m],over_tolerance=mismatches[m],
                          withheld_from_repair_comparison=m in bad,tolerance=PROB_TOL))
    return bad,audit,errors


def ranking_gate(old,replay,rq):
    """Detect even sub-tolerance score changes that alter reported rankings."""
    old=policy_records(old,'complete_only') if rq=='rq2' else old
    replay=policy_records(replay,'complete_only') if rq=='rq2' else replay
    bad=set();audit=[]
    methods=sorted(set(k for r in old for k in r['scores']))
    for probe in ['encoder','lightgbm']:
        for stratum in (['correct','incorrect','all'] if rq=='rq2' else ['all']):
            for frac in ([None] if rq=='rq2' else [.25,.5,.75]):
                for m in methods:
                    if m in ['fact_coverage','group_coverage','words']:continue
                    aa=primary(old,['fact_coverage','group_coverage',m],stratum,frac,probe)
                    bb=primary(replay,['fact_coverage','group_coverage',m],stratum,frac,probe)
                    if {r['explanation_id'] for r in aa}!={r['explanation_id'] for r in bb}:
                        bad.update(family(m));audit.append(dict(rq=rq,method=m,probe=probe,stratum=stratum,fraction=frac,issue='score_availability_changes_population'));continue
                    for cls in ([APContrast] if rq=='rq2' else [APContrast,PairContrast]):
                        ca=cls(aa,m,'group_coverage','subject_group');cb=cls(bb,m,'group_coverage','subject_group')
                        x=finite(ca.evaluate(np.ones(ca.n)));y=finite(cb.evaluate(np.ones(cb.n)))
                        if not number_equal(x,y,tol=1e-10):
                            bad.update(family(m));audit.append(dict(rq=rq,method=m,probe=probe,stratum=stratum,fraction=frac,
                                issue='rank_metric_replay_drift',metric=cls.__name__,original=x,replayed=y))
    return bad,audit


def rq1_gate(old,replay):
    ix=keyed(old,['patient_id','selector','probe','fraction']);bad=set();audit=[]
    fields=['subject_group','both_full_correct','available','selected_tokens','fact_count','correct','random_accuracy','random_correct','controls']
    for r in replay:
        k=tuple(r[x] for x in ['patient_id','selector','probe','fraction'])
        if k not in ix:raise ValueError('RQ1 replay identity')
        changed=[f for f in fields if ix[k].get(f)!=r.get(f)]
        if changed:bad.add(r['selector']);audit.append(dict(patient_id=r['patient_id'],selector=r['selector'],probe=r['probe'],fraction=r['fraction'],fields=changed))
    return bad,audit


def strip_scores(rs,bad):
    return [dict(r,scores={k:(None if k in bad else v) for k,v in r['scores'].items()}) for r in rs]


class Change:
    """A paired change of two contrasts, resampled with the same subject counts."""
    def __init__(self,old,new):
        self.old=old;self.new=new;self.cluster=old.cluster+new.cluster;self.n=old.n+new.n
    def evaluate(self,w):return self.new.evaluate(w[self.old.n:])-self.old.evaluate(w[:self.old.n])


def finite(x):return float(x) if np.isfinite(x) else None


def core(old,new,rq,view,reps):
    output=[]
    for probe in ['encoder','lightgbm']:
        strata=['correct','incorrect','all'] if rq=='RQ2' else ['all']
        for stratum in strata:
            for scope in ['common','reselected']:
                comparisons=COMPARISONS if rq=='RQ2' else [('lightgbm_absolute','tree_positive_shap')]*3
                contrasts=[];metadata=[]
                for i,(left,right) in enumerate(comparisons):
                    fraction=None if rq=='RQ2' else [.25,.5,.75][i]
                    methods=list(dict.fromkeys(['fact_coverage','group_coverage',left,right]))
                    aa=primary(old,methods,stratum,fraction,probe);bb=primary(new,methods,stratum,fraction,probe)
                    if scope=='common':
                        if rq=='RQ2':
                            shared={r['explanation_id'] for r in aa}&{r['explanation_id'] for r in bb}
                            aa=[r for r in aa if r['explanation_id'] in shared];bb=[r for r in bb if r['explanation_id'] in shared]
                        else:
                            shared={r['patient_id'] for r in aa}&{r['patient_id'] for r in bb}
                            aa=[r for r in aa if r['patient_id'] in shared];bb=[r for r in bb if r['patient_id'] in shared]
                    cls=APContrast if rq=='RQ2' else PairContrast
                    co=cls(aa,left,right,'subject_group');cn=cls(bb,left,right,'subject_group')
                    meta=dict(rq=rq,view=view,probe=probe,stratum=stratum,population=scope,
                         metric='AP_difference' if rq=='RQ2' else 'within_patient_concordance_difference',
                         left=left,right=right,fraction=fraction,
                         original_counts=census(aa),corrected_counts=census(bb),
                         original_effective_patients=len(set(co.cluster)),corrected_effective_patients=len(set(cn.cluster)),
                         original_pair_patients=co.n if rq=='RQ3' else None,corrected_pair_patients=cn.n if rq=='RQ3' else None,
                         original_pairs=co.pairs if rq=='RQ3' else None,corrected_pairs=cn.pairs if rq=='RQ3' else None,
                         common_unit='explanation' if rq=='RQ2' else 'admission; independently regenerated sets',
                         cluster='inferred_subject',secondary=(probe!='encoder' or stratum!='correct') if rq=='RQ2' else probe!='encoder')
                    for label,c in [('original',co),('corrected_updated_endpoint',cn),('change_updated_endpoint',Change(co,cn))]:
                        contrasts.append(c);metadata.append(dict(meta,version=label))
                    if scope=='common' and rq=='RQ2':
                        endpoints={r['explanation_id']:r['reference_failure'] for r in aa}
                        bf=[dict(r,reference_failure=endpoints[r['explanation_id']]) for r in bb]
                        cf=cls(bf,left,right,'subject_group')
                        contrasts.extend([cf,Change(co,cf)]);metadata.extend([dict(meta,version='corrected_fixed_original_endpoint'),dict(meta,version='change_fixed_original_endpoint')])
                # Same cluster draws; retain the original three-comparison family for each version.
                for version in dict.fromkeys(m['version'] for m in metadata):
                    ix=[i for i,m in enumerate(metadata) if m['version']==version];cs=[contrasts[i] for i in ix]
                    draws=resample(cs,reps,261016)
                    for k,stats in zip(ix,intervals(cs,draws)):
                        output.append(dict(metadata[k],**stats,interval_family='three prespecified core contrasts within RQ/probe/stratum/population/version',
                                           seed=261016))
    return output


def flows(old,new,rq,view):
    output=[]
    for probe in ['encoder','lightgbm']:
        for stratum in (['correct','incorrect','all'] if rq=='RQ2' else ['all']):
            for fraction in ([None] if rq=='RQ2' else [.25,.5,.75]):
                versions={v:[r for r in rs if r['probe']==probe and (fraction is None or r['fraction']==fraction)] for v,rs in [('original',old),('corrected',new)]}
                conditions=[('planned',lambda r:True),('mapped',lambda r:r['mapping_available']),('target_supported',lambda r:r['target_supported']),
                   ('probe_available',lambda r:r['probe_available']),('both_full_correct',lambda r:r['both_full_correct']),
                   ('proposal_stratum',lambda r:stratum=='all' or r['diagnosis_correct']==(stratum=='correct'))]
                for stage,fn in conditions:
                    versions={v:[r for r in rs if fn(r)] for v,rs in versions.items()}
                    ids={v:{r['patient_id'] if rq=='RQ3' else r['explanation_id'] for r in rs} for v,rs in versions.items()}
                    for v,rs in versions.items():
                        output.append(dict(rq=rq,view=view,probe=probe,stratum=stratum,fraction=fraction,stage=stage,version=v,**census(rs),
                           entered=len(ids['corrected']-ids['original']),exited=len(ids['original']-ids['corrected']),
                           transition_unit='admission' if rq=='RQ3' else 'explanation'))
    return output


def endpoints(old,new,view):
    ix=keyed(old,['explanation_id','probe']);out=[]
    for probe in ['encoder','lightgbm']:
        pairs=[(ix[r['explanation_id'],probe],r) for r in new if r['probe']==probe]
        for population in ['all_test','original_primary','common_primary']:
            def eligible(r):return r['mapping_available'] and r['target_supported'] and r['probe_available'] and r['both_full_correct'] and r['diagnosis_correct']
            selected=[(x,y) for x,y in pairs if population=='all_test' or (eligible(x) and (population=='original_primary' or eligible(y)))]
            valid=[(x,y) for x,y in selected if x['mapping_available'] and y['mapping_available'] and x['probe_available'] and y['probe_available']]
            row=dict(view=view,probe=probe,population=population,explanations=len(selected),patients=len({x['patient_id'] for x,y in selected}),
              common_available=len(valid),original_unavailable=sum(not(x['mapping_available'] and x['probe_available']) for x,y in selected),
              corrected_unavailable=sum(not(y['mapping_available'] and y['probe_available']) for x,y in selected),
              success_to_failure=sum(not x['reference_failure'] and y['reference_failure'] for x,y in valid),
              failure_to_success=sum(x['reference_failure'] and not y['reference_failure'] for x,y in valid),
              changed_prediction=sum(np.argmax(x['cited_probabilities'])!=np.argmax(y['cited_probabilities']) for x,y in valid),
              original_failure_patients=len({x['patient_id'] for x,y in valid if x['reference_failure']}),
              corrected_failure_patients=len({y['patient_id'] for x,y in valid if y['reference_failure']}),
              changed_full_eligibility=sum(x['both_full_correct']!=y['both_full_correct'] for x,y in selected))
            out.append(row)
    return out


def summarize(a):
    c=read(a.config);cfg=c['datasets']['TRACE'];root=Path(__file__).resolve().parents[1]
    def resolve(k):
        p=Path(cfg[k]);return p if p.is_absolute() else root/p
    repair=read(a.out/'repair_manifest.json');affected=set(repair['changed_admissions'])
    pending=set(repair.get('replacement_admissions',repair['pending_admissions']))
    saved={'rq1':rows(resolve('rq1_selections')),'rq2':rows(resolve('rq2_scores')),'rq3':rows(resolve('rq3_scores'))}
    raw={};identity=[]
    for variant in ['original','corrected']:
        for rq in ['rq1','rq2','rq3']:
            name='selections.jsonl' if rq=='rq1' else 'scores.jsonl'
            files=sorted(a.out.glob(f'worker_*/{variant}/{rq}/shard_0/{name}'))
            rs=[r for p in files for r in rows(p)]
            if {r['patient_id'] for r in rs}!=affected:raise ValueError('Incomplete affected census: '+variant+rq)
            raw[variant,rq]=rs
    selectors,rq1audit=rq1_gate(saved['rq1'],raw['original','rq1'])
    bad=set();gate=[]
    for rq in ['rq2','rq3']:
        b,g,e=score_gate(saved[rq],raw['original',rq],rq.upper());bad.update(b);gate+=g;identity+=e
    table(a.out/'F_repair_replay_audit.csv',gate);table(a.out/'F_repair_RQ1_replay_PRIVATE.csv',rq1audit)
    emit(a.out/'F_repair_replay_identity_errors_PRIVATE.jsonl',identity)
    if identity:
        dump(a.out/'replay_gate.json',dict(status='FAILED_IDENTITY_OR_ENDPOINT',errors=len(identity)))
        raise ValueError('Frozen replay differs in identity or endpoint; no repaired tables merged')
    rankaudit=[]
    for rq in ['rq2','rq3']:
        b,ra=ranking_gate(saved[rq],merge(saved[rq],raw['original',rq],affected),rq)
        bad.update(b);rankaudit+=ra
    table(a.out/'F_repair_ranking_replay_audit.csv',rankaudit)
    # Full records for controlled regenerated sets deliberately receive new IDs.
    for r in raw['corrected','rq3']:
        r['original_sampling_slot']=r['explanation_id'];r['explanation_id']='F_repair_regenerated|'+r['explanation_id']
    merged={rq:merge(saved[rq],raw['corrected',rq],affected) for rq in saved}
    for rq in ['rq2','rq3']:
        # Both sides lose whole families when the original replay exceeds its numerical gate.
        for r in gate:
            if r['method'] in bad:r['withheld_from_repair_comparison']=True
        saved[rq]=strip_scores(saved[rq],bad);merged[rq]=strip_scores(merged[rq],bad)
    table(a.out/'F_repair_replay_audit.csv',gate)
    dump(a.out/'replay_gate.json',dict(status='passed' if not bad and not selectors else 'passed_with_explicit_unavailable_comparisons',
        withheld_score_methods=sorted(bad),withheld_rq1_selectors=sorted(selectors),original_formal_tables_unchanged=True))
    core_rows=[];flow=[];transition=[]
    old=policy_records(saved['rq2'],'complete_only');projected=policy_records(merged['rq2'],'complete_only')
    strict=[dict(r,mapping_available=r['mapping_available'] and r['patient_id'] not in pending) for r in projected]
    resolved=repair.get('AI_rematched_explanations',0)==8
    views=[('corrected_F_AI_target_rematched_E' if resolved else 'corrected_F_projected_original_E',projected),
           ('corrected_F_AI_rematched_explanations_excluded' if resolved else 'corrected_F_replacement_E_pending_excluded',strict)]
    for view,new in views:
        print('Summary',view,flush=True)
        core_rows+=core(old,new,'RQ2',view,a.bootstrap);flow+=flows(old,new,'RQ2',view);transition+=endpoints(old,new,view)
    core_rows+=core(saved['rq3'],merged['rq3'],'RQ3','corrected_F_regenerated_sets',a.bootstrap)
    flow+=flows(saved['rq3'],merged['rq3'],'RQ3','corrected_F_regenerated_sets')
    # Exact original RQ1 table and interval implementation, with common-patient sensitivity.
    for rs in [saved['rq1'],merged['rq1']]:
        for r in rs:
            if r['selector'] in selectors:r['available']=False;r['unavailable_reason']='original_replay_selection_drift'
    for variant,rs in [('original',saved['rq1']),('corrected',merged['rq1'])]:
        dump(a.out/('F_repair_RQ1_'+variant+'.json'),summarize_rq1(rs))
    both0={r['patient_id'] for r in saved['rq1'] if r['both_full_correct']};both1={r['patient_id'] for r in merged['rq1'] if r['both_full_correct']}
    common=both0&both1
    rq1fields=['patient_id','selector','probe','fraction']
    oi=keyed(saved['rq1'],rq1fields);ni=keyed(merged['rq1'],rq1fields)
    if oi.keys()!=ni.keys():raise ValueError('RQ1 merged identity census')
    for variant,rs in [('original',saved['rq1']),('corrected',merged['rq1'])]:
        same=[dict(r,available=oi[tuple(r[k] for k in rq1fields)]['available'] and ni[tuple(r[k] for k in rq1fields)]['available'])
              for r in rs if r['patient_id'] in common]
        dump(a.out/('F_repair_RQ1_common_'+variant+'.json'),summarize_rq1(same))
    rq1changes=[]
    for probe in ['encoder','lightgbm']:
        for frac in [.25,.5,.75]:
            for sel in sorted({r['selector'] for r in saved['rq1']}):
                kk=[k for k in oi if k[1:]==(sel,probe,frac) and k[0] in common and oi[k]['available'] and ni[k]['available']]
                groups=[oi[k]['subject_group'] for k in kk]
                for metric in ['selected_accuracy','selection_minus_random']:
                    def val(r):return r['correct']-(r['random_accuracy'] if metric=='selection_minus_random' else 0.)
                    changes=[val(ni[k])-val(oi[k]) for k in kk]
                    rq1changes.append(dict(probe=probe,fraction=frac,selector=sel,metric=metric,
                        original=float(np.mean([val(oi[k]) for k in kk])) if kk else None,
                        corrected=float(np.mean([val(ni[k]) for k in kk])) if kk else None,
                        population='common eligible admissions for this selector',**rq1_interval(changes,groups=groups)))
    table(a.out/'F_repair_RQ1_changes.csv',rq1changes)
    for version,rs in [('original',saved['rq1']),('corrected',merged['rq1'])]:
        for probe in ['encoder','lightgbm']:
            for frac in [.25,.5,.75]:
                for sel in sorted({r['selector'] for r in rs}):
                    rr=[r for r in rs if r['probe']==probe and r['fraction']==frac and r['selector']==sel]
                    flow.append(dict(rq='RQ1',view='corrected_F_reselected',version=version,probe=probe,fraction=frac,selector=sel,
                        stage='available_and_full_correct',patients=len({r['patient_id'] for r in rr if r['available'] and r['both_full_correct']})))
    table(a.out/'F_repair_key_comparisons.csv',core_rows);table(a.out/'F_repair_population_flow.csv',flow)
    table(a.out/'F_repair_endpoint_transitions.csv',transition)
    for rq in saved:emit(a.out/('corrected_'+rq+'_merged_PRIVATE.jsonl'),merged[rq])
    (a.out/'README_results.md').write_text('''# Approved TRACE source-F sensitivity

This report holds fitted models, training data, split identities and generated explanations fixed. It repairs 28 test admissions using 81 removals, two state replacements and two unresolved facts withheld in the strict view. The remaining 115 unresolved items and 35 candidates are outside this repair.

RQ1 uses the original selection and ten matched-control implementation. RQ3 regenerates controlled sets under the original seed and group-budget protocol; slot indices are not paired across old/new sets. RQ3 uncertainty compares patient-level concordance using shared subject bootstrap draws. The projection of old controlled sets has not been analysed.

RQ2 retains diagnoses and text generated from original F. One view projects original citations onto surviving unchanged facts. The second additionally marks all eight explanations from state-replacement admissions unavailable pending new-fact citation review. Neither view invents citations to replacement facts. This is source-F sensitivity, separate from the completed human-EXACT mapping sensitivity.

Core AP/concordance differences and their changes use 5,000 shared inferred-subject bootstrap draws (seed 261016 by default), with separate three-contrast simultaneous intervals for each RQ/probe/stratum/population/version. Corrected fixed-original-endpoint analyses appear only for common natural explanations. Undefined replicates and effective event counts are explicit. RQ1 uses the original 1,000-draw percentile protocol.

Original replay is checked before merging corrected and unaffected patients. Whole score families or selectors failing replay are withheld from repair comparisons on both sides, with their raw outputs retained. Endpoint or identity replay failure stops merging. Formal main tables are never overwritten. Inspect replay_gate.json before interpreting a table.

F_repair_key_comparisons.csv contains core RQ2/RQ3 contrasts; the four F_repair_RQ1 JSON files contain selection tables and overlap comparisons. F_repair_population_flow.csv reports eligibility; F_repair_endpoint_transitions.csv reports natural-explanation transitions. Private merged records allow follow-up analyses. This package does not establish full-cohort source-fact correctness or independently classify error types.
''',encoding='utf-8')
    if resolved:
        report=a.out/'README_results.md';text=report.read_text(encoding='utf-8')
        text=text.replace('The second additionally marks all eight explanations from state-replacement admissions unavailable pending new-fact citation review.',
            'All eight replacement-target citations were reviewed as DOES_NOT_CITE_NEW using AI-assisted full-text review. The primary view retains these explanations; a secondary view excludes the eight AI-rematched explanations. The original COMPLETE/PARTIAL status and other unresolved items remain unchanged. This rematch is not a new human adjudication.')
        report.write_text(text,encoding='utf-8')
    return dict(status='complete_conditional_source_F_sensitivity',changed_admissions=len(affected),
        AI_rematched_explanations=repair.get('AI_rematched_explanations',0),human_rematch_confirmed=False,
        pending_replacement_explanations=repair['pending_replacement_explanations'],
        withheld_score_methods=sorted(bad),withheld_rq1_selectors=sorted(selectors),
        core_rows=len(core_rows),endpoint_rows=len(transition),formal_tables_replaced=False)
