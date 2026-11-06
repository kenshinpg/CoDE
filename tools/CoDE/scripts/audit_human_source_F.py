"""Read-only source-F exposure audit; human error labels are not propagated."""
import sys,json
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from code_eval.human_return import read,rows,sha,dump,emit,table

def normalize_source(r):
    return dict(record_id=r['记录ID'],patient_id=str(r['住院ID']),record_type=r['记录类型'],judgment=r['人工结论'],
        fact_id=r.get('fact_id') or None,review_reason=r['人工理由'],proposed_action=r['建议处理'],
        inherited_issue_type=r.get('AI问题类型'),source_spans=json.loads(r.get('原文跨度') or '[]'),
        source_observation_ids=json.loads(r.get('来源关联') or '[]'))

def base_impact(source,patients,split,reviews):
    fi={f['fact_id']:(p['patient_id'],f) for p in patients for f in p['facts']}
    bytoken=defaultdict(list);byfeature=defaultdict(list)
    for pid,f in fi.values():
        bytoken[f['feature_id'],f['state']].append(pid);byfeature[f['feature_id']].append(pid)
    rr=[r for r in reviews if r['dataset']=='TRACE'];out=[];flags=defaultdict(lambda:defaultdict(list))
    for r in source:
        if r['record_type'] not in ['FROZEN_FACT','SOURCE_OMISSION_OR_VIEW_EXCLUSION_CANDIDATE']:continue
        if r['record_type']=='FROZEN_FACT' and r['judgment'] not in ['ERROR_CONFIRMED','UNRESOLVED']:continue
        pid=r['patient_id'];fid=r['fact_id'];s=split.get(pid)
        if not s:raise ValueError('Reviewed admission missing from frozen split')
        exists=fid in fi
        if exists and fi[fid][0]!=pid:raise ValueError('Fact belongs to different admission')
        f=fi[fid][1] if exists else None
        is_candidate=r['record_type']=='SOURCE_OMISSION_OR_VIEW_EXCLUSION_CANDIDATE'
        if not is_candidate and not exists:raise ValueError('Reviewed frozen fact no longer in F')
        if exists:flags[pid][r['judgment']].append(fid)
        x=dict(dataset='TRACE',record_id=r['record_id'],patient_id=pid,record_type=r['record_type'],judgment=r['judgment'],
            fact_id=fid,partition=s['partition'],reference_diagnosis=s.get('reference_class'),in_frozen_F=exists,
            in_training_input=exists and s['partition']=='train',in_development_input=exists and s['partition']=='development',
            in_test_full_input=exists and s['partition']=='test',feature_id=f['feature_id'] if f else None,state=f['state'] if f else None,
            inherited_issue_type=r['inherited_issue_type'],
            candidate_eligibility='UNRESOLVED_requires_mention_level_view_link' if is_candidate else 'already_in_frozen_F',
            human_exact_review_explanations=sum(fid in q['human_return_exact_E'] for q in rr) if exists else 0,
            automatic_review_explanations=sum(fid in q['automatic_E'] for q in rr) if exists else 0,
            broader_rule_status='same_feature_state_occurrence_is_audit_candidate_not_confirmed_error' if exists else 'not_applicable')
        for part in ['train','development','test']:
            x['same_feature_state_'+part+'_patients']=len({p for p in bytoken[f['feature_id'],f['state']] if split[p]['partition']==part}) if f else None
            x['same_feature_'+part+'_patients']=len({p for p in byfeature[f['feature_id']] if split[p]['partition']==part}) if f else None
        out.append(x)
    return out,{p:dict(v) for p,v in flags.items()}

def panel_records(rs,p):
    return [r for r in rs if r['probe']==p['probe'] and r['mapping_available'] and r['probe_available'] and r['both_full_correct']
        and (r.get('scan_status','COMPLETE')=='COMPLETE')
        and (p.get('stratum','all')=='all' or r['diagnosis_correct']==(p['stratum']=='correct'))
        and (p['family']=='quantity_only' or r['target_supported'])
        and (p.get('fraction') is None or r.get('fraction')==p['fraction'])
        and all(r['scores'].get(m) is not None for m in p['methods'])]

def influence(rs,tokens,flags,meta):
    out=[]
    for status in ['ERROR_CONFIRMED','UNRESOLVED']:
        pids={p for p,f in flags.items() if f.get(status)}
        affected={p:{tokens[f] for f in v.get(status,[])} for p,v in flags.items()}
        for route in ['full_F','selected_or_cited_E']:
            chosen=[r for r in rs if r['patient_id'] in pids and (route=='full_F' or affected[r['patient_id']]&set(r.get('cited_tokens') or r.get('selected_tokens') or []))]
            fail=[r for r in chosen if r.get('reference_failure') is True or (r.get('correct') is not None and r['correct']==0)]
            out.append(dict(meta,judgment=status,exposure_route=route,records=len(chosen),patients=len({r['patient_id'] for r in chosen}),
                failures=len(fail),failure_patients=len({r['patient_id'] for r in fail}),panel_records=len(rs),
                panel_patients=len({r['patient_id'] for r in rs}),interpretation='exposure counts; no causal effect or repaired endpoint estimated'))
    return out

def run(a,cfg,reviews):
    view=Path(cfg['view']);raw=Path(cfg['raw']);out=a.out/'source_F';out.mkdir(exist_ok=True)
    anchor=read(ROOT/'inputs/round1_A4_20261016/frozen_F_binding.json');vm=read(view/'manifest.json')
    for name in ['patients_model_input_PRIVATE.jsonl','fact_provenance_PRIVATE.jsonl','split_PRIVATE.jsonl','view_config.json']:
        if sha(view/name)!=anchor['source_sha256'][name]:raise ValueError('Frozen F drift: '+name)
    patients=rows(view/'patients_model_input_PRIVATE.jsonl');split={r['admission_id']:r for r in rows(view/'split_PRIVATE.jsonl')}
    actual_records={}
    for part in ['train','development','test']:
        p=view/(part+'_records_PRIVATE.jsonl')
        if sha(p)!=anchor['source_sha256'][p.name]:raise ValueError('Partition input drift')
        for r in rows(p):
            if r['patient_id'] in actual_records or r['partition']!=part:raise ValueError('Duplicate/changed fitted input partition')
            actual_records[r['patient_id']]=r
    for p in patients:
        r=actual_records[p['patient_id']]
        if set(r['tokens'])!={f['feature_id']+'_@_'+f['state'] for f in p['facts']}:raise ValueError('F differs from actual model input tokens')
    source=[normalize_source(r) for r in rows(a.review/'source_fact_review_human_PRIVATE.jsonl')]
    impact,flags=base_impact(source,patients,split,reviews)
    fp={f['fact_id']:f for p in patients for f in p['facts']};tokens={k:f['feature_id']+'_@_'+f['state'] for k,f in fp.items()}
    provenance={r['fact_id']:r for r in rows(view/'fact_provenance_PRIVATE.jsonl')}
    exposure=[];rq2=rows(Path(cfg['rq2_scores']));rq3=rows(Path(cfg['rq3_scores']))
    for rq,rs,panelpath in [('RQ2',rq2,cfg['rq2_panels']),('RQ3',rq3,cfg['rq3_panels'])]:
        for p in read(Path(panelpath)):
            if p.get('mapping_policy','complete_only')!='complete_only' or p['endpoint']!='reference':continue
            chosen=panel_records(rs,p)
            if len(chosen)!=p['common'] or len({r['patient_id'] for r in chosen})!=p['patients']:raise ValueError('Formal panel population mismatch '+rq+' '+p['family'])
            exposure+=influence(chosen,tokens,flags,dict(rq=rq,probe=p['probe'],family=p['family'],stratum=p['stratum'],fraction=p.get('fraction'),selector=''))
    rq1path=Path(cfg['rq1_selections']);rq1=rows(rq1path)
    panels=read(Path(cfg['rq1_panels']))['panels']
    from code_eval.trace2400.rq1_summary import FAMILIES
    for p in panels:
        rs=[r for r in rq1 if r['probe']==p['probe'] and r['fraction']==p['fraction']]
        common=set.intersection(*[{r['patient_id'] for r in rs if r['selector']==m and r['available'] and (p['population']=='all_test' or r['both_full_correct'])} for m in FAMILIES[p['family']]])
        if len(common)!=p['patients']:raise ValueError('RQ1 formal census differs')
        chosen=[r for r in rs if r['selector']==p['selector'] and r['patient_id'] in common]
        exposure+=influence(chosen,tokens,flags,dict(rq='RQ1',probe=p['probe'],family=p['family'],stratum=p['population'],fraction=p['fraction'],selector=p['selector']))
    for x in impact:
        fid=x['fact_id'];pid=x['patient_id'];token=tokens.get(fid)
        if not token:continue
        x['source_observation_ids']=provenance[fid]['source_observation_ids']
        for probe in ['encoder','lightgbm']:
            rr=[r for r in rq2 if r['probe']==probe and r['patient_id']==pid]
            x[probe+'_automatic_E_all_explanations']=sum(token in (r['cited_tokens'] or []) for r in rr)
            for name,methods in [('quantity',['fact_coverage','group_coverage']),('cohd_vs_group',['fact_coverage','group_coverage','cohd_coverage']),('shap_vs_group',['fact_coverage','group_coverage','tree_positive_shap']),('shap_vs_A',['fact_coverage','group_coverage','tree_positive_shap','lightgbm_absolute'])]:
                pp=dict(probe=probe,stratum='correct',family='quantity_only' if name=='quantity' else name,methods=methods)
                sel=panel_records(rr,pp);hit=[r for r in sel if token in (r['cited_tokens'] or [])]
                x[probe+'_'+name+'_full_input_explanations']=len(sel)
                x[probe+'_'+name+'_full_input_failures']=sum(r['reference_failure'] for r in sel)
                x[probe+'_'+name+'_automatic_E_explanations']=len(hit)
                x[probe+'_'+name+'_automatic_E_failures']=sum(r['reference_failure'] for r in hit)
    # Source candidates have broad spans and no observation IDs. Preserve candidate context links,
    # never infer eligibility solely from overlap with an observation in the same passage.
    candidates=[r for r in source if r['record_type']=='SOURCE_OMISSION_OR_VIEW_EXCLUSION_CANDIDATE']
    candidate_out=[];obs_path=raw/'observations_PRIVATE.jsonl';decision_path=view/'observation_decisions_PRIVATE.jsonl'
    missing=[]
    if obs_path.exists() and decision_path.exists():
        if sha(obs_path)!=vm['input_sha256'][obs_path.name] or sha(decision_path)!=vm['output_sha256'][decision_path.name]:raise ValueError('Original candidate provenance changed')
        decisions={r['observation_id']:r for r in rows(decision_path)}
        final_by_observation=defaultdict(list);all_provenance_by_observation=defaultdict(list)
        for fid,pv in provenance.items():
            for oid in pv['source_observation_ids']:
                all_provenance_by_observation[oid].append(fid)
                if fid in fp:final_by_observation[oid].append(fid)
        conflict_ids=set();cp=view/'conflicts_PRIVATE.jsonl'
        if cp.exists():
            if sha(cp)!=vm['output_sha256'].get(cp.name):raise ValueError('Original conflict ledger changed')
            conflict_ids={oid for r in rows(cp) for oid in r['source_observation_ids']}
        else:missing.append('Original concept conflict ledger unavailable for candidate eligibility')
        wanted={r['patient_id'] for r in candidates};observations=defaultdict(list)
        with obs_path.open(encoding='utf-8') as f:
            for line in f:
                o=json.loads(line)
                if o['admission_id'] in wanted:observations[o['admission_id']].append(o)
        for c in candidates:
            linked=[]
            for o in observations[c['patient_id']]:
                quote=(o.get('raw') or {}).get('quote') or ''
                docs=[s.get('document_id') for s in c['source_spans']]
                if quote and any(doc and o['observation_id'].startswith(doc+':') and quote in (s.get('quote') or '') for doc,s in zip(docs,c['source_spans'])):
                    oid=o['observation_id'];reason=decisions[oid]['reason']
                    if final_by_observation[oid]:fate='represented_in_final_F'
                    elif reason!='candidate':fate=reason
                    elif oid in conflict_ids:fate='opposite_assertion_conflict_excluded'
                    elif all_provenance_by_observation[oid]:fate='fact_created_but_not_in_final_F_training_frequency_filter'
                    else:fate='candidate_final_fate_unresolved'
                    linked.append(dict(observation_id=oid,decision=reason,concept_id=decisions[oid].get('concept_id'),
                                       final_fate=fate,linked_final_F_fact_ids=final_by_observation[oid]))
            candidate_out.append(dict(c,context_observation_links=linked,eligibility='UNRESOLVED_broad_context_is_not_mention_identity',
                next_required='Split bundled mentions; link each to a source observation/concept; inspect conflicts, state and train-frequency eligibility',added_to_F=False))
    else:
        missing.append('Original candidate context/decision records unavailable')
        candidate_out=[dict(c,eligibility='UNRESOLVED_original_context_records_missing',added_to_F=False) for c in candidates]
    table(a.out/'source_F_impact.csv',impact);table(a.out/'source_F_population_impact.csv',exposure)
    emit(out/'source_omission_candidates_PRIVATE.jsonl',candidate_out);emit(out/'source_reviews_normalized_PRIVATE.jsonl',source)
    dump(out/'source_flags_PRIVATE.json',flags)
    summary={s:dict(items=sum(x['judgment']==s and x['in_frozen_F'] for x in impact),
        admissions=len({x['patient_id'] for x in impact if x['judgment']==s and x['in_frozen_F']}),
        partitions=dict(Counter(x['partition'] for x in impact if x['judgment']==s and x['in_frozen_F'])),
        inherited_types=dict(Counter(x['inherited_issue_type'] for x in impact if x['judgment']==s and x['in_frozen_F']))) for s in ['ERROR_CONFIRMED','UNRESOLVED']}
    dump(out/'summary.json',dict(status='complete' if not missing else 'partial',source_audit=summary,candidates=len(candidates),candidate_eligibility_unresolved=len(candidate_out),missing=missing,
        main_tables_changed=False,F_modified=False,model_refit=False,
        caveat='Same-feature occurrences are an inspection scope only. Test input exposure can affect full-input selection and weights even when absent from E.'))
    return flags
