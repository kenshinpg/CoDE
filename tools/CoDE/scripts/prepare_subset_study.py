"""Prepare label-free subset-study inputs from existing CoDE records.

No model fitting, API calls, or clinical label construction. The exported
queries deliberately request all-class predictions without a target diagnosis.
"""
import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def build(records, fractions=(.25, .5, .75, 1.0)):
    patients, explanations, queries, joins = {}, [], {}, []
    ids = set()
    for r in records:
        eid, pid = r['explanation_id'], r['patient_id']
        if eid in ids:
            raise ValueError('Duplicate explanation_id')
        ids.add(eid)
        # Explicit public-field projection prevents labels in source metadata
        # from entering selector/probe inputs.
        facts = []
        for f in r['facts']:
            if not f.get('visible', True):
                raise ValueError('Legacy scoring record contains invisible facts')
            if f.get('coverage_scope') != 'symptoms_antecedents':
                raise ValueError('Unexpected fact scope; update adapter explicitly')
            facts.append({k: f[k] for k in ('fact_id', 'feature_id', 'value',
                          'group_id', 'polarity', 'observation_status')})
            facts[-1]['source_token'] = f['provenance']['source_token']
        facts.sort(key=lambda f: f['fact_id'])
        fact_ids = {f['fact_id'] for f in facts}
        if not facts or len(fact_ids) != len(facts):
            raise ValueError('Empty or duplicate patient facts')
        groups = {k: sorted(v) for k, v in sorted(r['groups'].items())}
        flat = [fid for v in groups.values() for fid in v]
        if any(not v for v in groups.values()) or len(flat) != len(set(flat)) or set(flat) != fact_ids:
            raise ValueError('Groups must partition patient facts')
        if any(f['fact_id'] not in groups.get(f['group_id'], []) for f in facts):
            raise ValueError('Fact/group identity mismatch')
        demo = r.get('demographics') or {}
        context = {k: demo.get(k) for k in ('age', 'sex')}
        visible = dict(facts=facts, groups=groups, fixed_context=context)
        fingerprint = digest(visible)
        if pid in patients and patients[pid]['visible_input_hash'] != fingerprint:
            raise ValueError('Patient has inconsistent visible inputs; separate input views explicitly')
        patient = dict(patient_id=pid, source_pattern_id=r['pattern_id'],
                       visible_input_hash=fingerprint, **visible,
                       budget_group_counts=sorted({math.ceil(t*len(groups)) for t in fractions}),
                       split_status='DEVELOPMENT_ONLY_NOT_SPLIT_VERIFIED')
        if pid in patients and patients[pid]['source_pattern_id'] != r['pattern_id']:
            raise ValueError('Inconsistent source pattern identity')
        patients[pid] = patient
        cited = r['provisional_cited_fact_ids']
        if len(cited) != len(set(cited)) or not set(cited) <= fact_ids:
            raise ValueError('Invalid cited fact IDs')
        complete = r['extraction_status'] == 'INITIAL_COMPLETE_UNREVIEWED'
        eligible = r['generation_status'] == 'VALID' and complete
        explanations.append(dict(explanation_id=eid, patient_id=pid,
            visible_input_hash=fingerprint, cited_fact_ids=sorted(cited),
            proposed_diagnosis=(r.get('ranked_diagnoses') or [None])[0],
            extraction_status=r['extraction_status'], semantic_review='UNREVIEWED',
            scoring_eligible=eligible,
            ineligible_reason=None if eligible else 'generation_or_extraction_incomplete'))
        subsets = [('full', fact_ids)] + ([('cited', set(cited))] if eligible else [])
        for kind, subset in subsets:
            q = dict(facts=[f for f in facts if f['fact_id'] in subset],
                     fixed_context=context, prediction_request='all_class_distribution',
                     missing_semantics='unknown_not_negative')
            qid = digest(q)
            queries[qid] = dict(query_id=qid, **q)
            joins.append(dict(explanation_id=eid, patient_id=pid, kind=kind, query_id=qid))
    patient_records = list(patients.values())
    by_patient = defaultdict(list)
    for e in explanations:
        if e['scoring_eligible']:
            by_patient[e['patient_id']].append(e)
    eligible = [e for e in explanations if e['scoring_eligible']]
    empty = sum(not e['cited_fact_ids'] for e in eligible)
    full = sum(len(e['cited_fact_ids']) == len(patients[e['patient_id']]['facts']) for e in eligible)
    sizes = [len({tuple(e['cited_fact_ids']) for e in es}) for es in by_patient.values()]
    summary = dict(status='INTERFACE_PREPARATION_NOT_RESEARCH_RESULTS',
        patients=len(patients), explanations=len(explanations),
        extraction_statuses=dict(Counter(e['extraction_status'] for e in explanations)),
        eligible_explanations=len(eligible), eligible_patients=len(by_patient),
        eligible_empty_citation_sets=empty, eligible_full_citation_sets=full,
        unique_prediction_queries=len(queries), logical_query_joins=len(joins),
        median_patient_facts=statistics.median(len(r['facts']) for r in patient_records) if patient_records else None,
        median_patient_groups=statistics.median(len(r['groups']) for r in patient_records) if patient_records else None,
        distinct_cited_sets_per_eligible_patient=dict(Counter(sizes)),
        api_calls=0, models_fitted=0, split_isolation='NOT_VERIFIED',
        semantic_mapping='AUTOMATIC_UNREVIEWED',
        note='Partial extractions retained in inventory, not scored as complete sets. Demographics are fixed context. Query hashes need model and encoding hashes when used as prediction cache keys.')
    return {'patients': patient_records, 'explanations': explanations,
            'subset_queries': list(queries.values()), 'query_joins': joins}, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('Use a new output directory')
    records = [json.loads(x) for x in a.records.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
    outputs, report = build(records)
    report['records_sha256'] = hashlib.sha256(a.records.read_bytes()).hexdigest()
    a.out.mkdir(parents=True)
    for name, data in outputs.items():
        (a.out/(name+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data), encoding='utf8')
    (a.out/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
