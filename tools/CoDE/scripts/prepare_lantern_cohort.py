"""Verify the filtered LANTERN export and adapt exact candidate citations.

Preserve original partitions. Cross-partition evidence patterns are inventoried
but quarantined from scoring; all retained patterns are excluded from fitting.
This is engineering acceptance, not medical semantic validation.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import tarfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from code_eval.code_experiment import adapt, fingerprint
from code_eval.tree_subset import Encoding, pattern
from prepare_subset_study import build


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def emit(stream, obj):
    stream.write(json.dumps(obj, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def index(path, key):
    result={}
    for r in rows(path):
        if r[key] in result: raise ValueError('Duplicate key in '+str(path))
        result[r[key]]=r
    return result


def exact_citations(patient, extraction, dictionary):
    facts={f['fact_id']:f for f in patient['facts']}
    if len(facts)!=len(patient['facts']): raise ValueError('Duplicate patient fact')
    cited=set(); counts=Counter()
    for c in extraction['claims']:
        counts[c['status']]+=1
        if not c.get('quote') or c['quote'] not in extraction['explanation']:
            raise ValueError('Quote not in explanation')
        matched=c.get('matched_fact_ids', [])
        if len(matched)!=len(set(matched)) or not set(matched)<=set(facts):
            raise ValueError('Invalid fact association')
        if c['status']!='MATCHED' or c.get('resolution')!='exact' or c.get('value_resolution')!='exact':
            continue
        if c.get('polarity') not in ('affirmed','negated') or c.get('value') is None:
            raise ValueError('Exact matched claim has invalid value/polarity')
        if not matched: raise ValueError('Matched claim has no fact')
        code=c['feature_id']; value=str(c['value']); meta=dictionary['features'][code]
        if meta['type']=='B' and c['polarity']=='negated':
            if value not in ('true','false'): raise ValueError('Invalid binary value')
            value='false' if value=='true' else 'true'
        elif c['polarity']=='negated':
            raise ValueError('Nonbinary negated claim cannot be an exact matched value')
        for fid in matched:
            f=facts[fid]
            equal=(float(value)==float(f['value']) if meta['type']=='N' else value==str(f['value']))
            if f['feature_id']!=code or not equal or not f.get('visible',True):
                raise ValueError('Exact mapping disagrees with patient value')
            if f['coverage_scope']=='symptoms_antecedents': cited.add(fid)
    return sorted(cited), counts


def prepare(source, out, expected_manifest=None, workers=4):
    if not (source/'COMPLETE.json').exists() or (source/'INCOMPLETE.json').exists():
        raise ValueError('Source export is not complete')
    manifest=read(source/'manifest.json')
    if expected_manifest and manifest!=read(expected_manifest):
        raise ValueError('Source differs from the reviewed handoff manifest')
    required={'patients_public.jsonl','d0_initial_complete.jsonl','generation_results.jsonl',
        'generation_requests.jsonl','evidence_dictionary.json','case_selection_private.jsonl',
        'reference_private.jsonl','response_selection_private.jsonl','record_decisions_private.jsonl',
        'patient_retention_private.jsonl','audit_report.json'}
    if not required<=manifest['artifacts'].keys(): raise ValueError('Missing required artifact hash')
    before={}
    def check(item):
        name, expected=item
        if Path(name).name!=name: raise ValueError('Expected flat artifact paths')
        path=source/name
        stat=path.stat(); got=sha(path)
        if got!=expected: raise ValueError('Hash mismatch: '+name)
        print('Hash OK:', name, flush=True)
        return name,(stat.st_size,stat.st_mtime_ns)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        before=dict(pool.map(check,manifest['artifacts'].items()))
    patients=index(source/'patients_public.jsonl','patient_id')
    cases=index(source/'case_selection_private.jsonl','patient_id')
    references=index(source/'reference_private.jsonl','patient_id')
    retention=index(source/'patient_retention_private.jsonl','patient_id')
    decisions=index(source/'record_decisions_private.jsonl','explanation_id')
    responses=index(source/'response_selection_private.jsonl','explanation_id')
    expected={k:v for k,v in decisions.items() if v['decision']=='INCLUDED_INITIAL_UNREVIEWED'}
    if set(patients)!=set(cases) or set(patients)!=set(references) or set(responses)!=set(expected):
        raise ValueError('Patient/response private association sets differ')
    dictionary=read(source/'evidence_dictionary.json'); encoding=Encoding(dictionary)
    pattern_partitions=defaultdict(set); patient_patterns={}
    for pid, p in patients.items():
        case=cases[pid]; partition=case['partition']
        if partition not in ('train','development','evaluation') or case['component'] not in ('coverage','challenge'):
            raise ValueError('Unknown original partition/component')
        if references[pid]['pathology']!=case['disease']:
            raise ValueError('Reference diagnosis and selection disease disagree')
        if p.get('split') is not None and p['split']!=partition:
            raise ValueError('Public patient partition differs from selection')
        for k in ('partition','component','disease'):
            if case[k]!=retention[pid][k]: raise ValueError('Patient selection metadata mismatch')
        fs=[f for f in p['facts'] if f.get('visible',True) and f['coverage_scope']=='symptoms_antecedents']
        tokens=[f['provenance']['source_token'] for f in fs]
        pat=pattern(tokens)
        if pat!=p['pattern_id']: raise ValueError('Patient pattern mismatch')
        encoding.encode(tokens,p['demographics'])
        for f in fs:
            code,*value=f['provenance']['source_token'].split('_@_',1)
            if code!=f['feature_id'] or (value[0] if value else 'true')!=str(f['value']):
                raise ValueError('Patient token/value mismatch')
            if encoding.features[code]['group_id']!=f['group_id']: raise ValueError('Dependency mismatch')
        patient_patterns[pid]=pat; pattern_partitions[pat].add(partition)
    collisions={p for p,s in pattern_partitions.items() if len(s)>1}
    (out/'excluded_fit_patterns.txt').write_text('\n'.join(sorted(pattern_partitions))+'\n',encoding='utf8')
    dump(out/'pattern_audit.json',dict(patterns=len(pattern_partitions),cross_partition_patterns=len(collisions),
        quarantined_patients=sum(p in collisions for p in patient_patterns.values()),
        policy='retain original partition; quarantine all sides of crossing patterns; exclude all retained patterns from teacher fitting'))
    generation={}
    for r in rows(source/'generation_results.jsonl'):
        eid=r['id']
        if eid in generation or eid not in expected or r['patient_id']!=expected[eid]['patient_id'] or r['status']!='VALID':
            raise ValueError('Generation association or status mismatch')
        generation[eid]={k:r.get(k) for k in ('id','patient_id','status','parsed','run_id')}
    if set(generation)!=set(expected): raise ValueError('Generation set incomplete')
    seen=set(); prompt_hashes=Counter(); candidate_labels=None
    for r in rows(source/'generation_requests.jsonl'):
        eid=r['id']
        if eid in seen or eid not in expected or r['patient_id']!=expected[eid]['patient_id']:
            raise ValueError('Request association mismatch')
        if not isinstance(r.get('messages'),list) or not r['messages']:
            raise ValueError('Missing actual generation messages')
        users=[v['content'] for v in r['messages'] if v['role']=='user']
        if len(users)!=1: raise ValueError('Unexpected generation prompt structure')
        payload=json.loads(users[0])
        if set(payload)!={'allowed_diagnoses','patient','missing_policy'}:
            raise ValueError('Unexpected generator input fields; inspect before proceeding')
        if payload['patient']!=patients[r['patient_id']]['input_text']:
            raise ValueError('Generator patient text differs from public visible input')
        labels=payload['allowed_diagnoses']
        if len(labels)<2 or len(labels)!=len(set(labels)):
            raise ValueError('Invalid candidate diagnosis list')
        if candidate_labels is None: candidate_labels=labels
        if labels!=candidate_labels: raise ValueError('Candidate diagnosis lists differ')
        ranking=generation[eid]['parsed']['ranked_diagnoses']
        if len(ranking)!=3 or len(set(ranking))!=3 or not set(ranking)<=set(labels):
            raise ValueError('Invalid generated ranking')
        prompt_hashes[fingerprint([v for v in r['messages'] if v['role']=='system'])]+=1
        seen.add(eid)
    if seen!=set(expected): raise ValueError('Request set incomplete')
    streams={s:(out/(s+'_records.jsonl')).open('w',encoding='utf8') for s in ['train','development','evaluation','quarantine']}
    summaries=defaultdict(Counter); claim_counts=Counter(); pairs=defaultdict(list); seen=set(); retained=Counter()
    try:
        with (out/'analysis_index_private.jsonl').open('w',encoding='utf8') as private, \
             (out/'development_outcomes.jsonl').open('w',encoding='utf8') as devout:
            for n, ext in enumerate(rows(source/'d0_initial_complete.jsonl'),1):
                eid=ext['explanation_id']; pid=ext['patient_id']
                if eid in seen or eid not in expected or expected[eid]['patient_id']!=pid:
                    raise ValueError('Extraction association mismatch')
                if ext['status']!='INITIAL_COMPLETE_UNREVIEWED' or ext.get('issues')!=[]:
                    raise ValueError('Unexpected extraction status/issues')
                gen=generation.pop(eid)
                if ext['ranked_diagnoses']!=gen['parsed']['ranked_diagnoses']:
                    raise ValueError('Extraction/generation ranking drift')
                cited,counts=exact_citations(patients[pid],ext,dictionary); claim_counts.update(counts)
                rec=adapt(patients[pid],gen,ext,manifest['artifacts']['d0_initial_complete.jsonl'])
                rec['provisional_cited_fact_ids']=cited
                rec['mapping_policy']='MATCHED_and_resolution_exact_and_value_resolution_exact'
                rec['record_sha256']=fingerprint({k:v for k,v in rec.items() if k!='record_sha256'})
                # Reuse the public schema contract, without storing bulky per-record query exports.
                build([rec])
                case=cases[pid]; partition=case['partition']
                for k in ('partition','component'):
                    if expected[eid][k]!=case[k] or responses[eid][k]!=case[k]:
                        raise ValueError('Response/patient strata mismatch')
                if responses[eid]['patient_id']!=pid: raise ValueError('Response patient mismatch')
                rank=rec['ranked_diagnoses'].index(case['disease'])+1 if case['disease'] in rec['ranked_diagnoses'] else 4
                if rank!=responses[eid]['reference_rank'] or rank!=expected[eid]['reference_rank']:
                    raise ValueError('Reference rank does not agree with generated diagnoses')
                dest='quarantine' if patient_patterns[pid] in collisions else partition
                emit(streams[dest],rec)
                emit(private,dict(explanation_id=eid,patient_id=pid,pattern_id=patient_patterns[pid],
                    original_partition=partition,analysis_partition=dest,component=case['component'],
                    reference_diagnosis=case['disease'],reference_rank=responses[eid]['reference_rank'],
                    source_reference=references[pid]))
                if dest=='development':
                    emit(devout,dict(explanation_id=eid,patient_id=pid,pattern_id=patient_patterns[pid],
                        component=case['component'],reference_diagnosis=case['disease'],reference_rank=rank,
                        usage='evaluation_sidecar_never_predictor_input'))
                for key in (dest, dest+'|'+case['component']):
                    summaries[key]['explanations']+=1
                    summaries[key]['empty_exact_citations']+=not cited
                pairs[(dest,pid)].append((rec['ranked_diagnoses'][0],len(cited),tuple(cited)))
                seen.add(eid);retained[pid]+=1
                if n%1000==0: print('Adapted',n,'explanations',flush=True)
    finally:
        for stream in streams.values(): stream.close()
    report=read(source/'audit_report.json')
    if seen!=set(expected) or len(seen)!=report['exported_explanations'] or len(retained)!=report['exported_patients']:
        raise ValueError('Final extraction count/ID mismatch')
    for pid,r in retention.items():
        if retained[pid]!=r['retained']: raise ValueError('Retention mismatch')
    for (split,pid),items in pairs.items():
        summaries[split]['patients']+=1
        summaries[split]['distinct_cited_sets']+=len({r[2] for r in items})
        for a,b in itertools.combinations(items,2):
            if a[:2]==b[:2]:
                summaries[split]['same_diagnosis_equal_count_pairs']+=1
                summaries[split]['same_diagnosis_equal_count_distinct_set_pairs']+=a[2]!=b[2]
    for name,(size,mtime) in before.items():
        stat=(source/name).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(size,mtime): raise ValueError('Source changed during reading')
    shutil.copyfile(source/'evidence_dictionary.json',out/'evidence_dictionary.json')
    result=dict(status='complete',acceptance='ENGINEERING_ONLY_UNREVIEWED_SEMANTICS',
        source_manifest_sha256=sha(source/'manifest.json'),patients=len(retained),explanations=len(seen),
        reference_disease_count=len({r['disease'] for r in cases.values()}),
        summaries={k:dict(v) for k,v in summaries.items()},claim_statuses=dict(claim_counts),
        generator_candidate_labels=candidate_labels,system_prompt_hashes=dict(prompt_hashes),
        api_calls=0,evaluation_predictions_computed=False,
        selection='coverage plus error-enriched challenge; retrospective production-train partitions',
        source_sha256=manifest['artifacts'])
    dump(out/'cohort_manifest.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('inputs/ddxplus_prepared'))
    p.add_argument('--out',type=Path,default=Path('artifacts/lantern_cohort_v1'))
    p.add_argument('--expected-manifest',type=Path,default=ROOT/'configs/lantern_filtered_expected_manifest_20261006.json')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:
        result=prepare(a.source,a.out,a.expected_manifest,a.workers)
        dump(a.out/'COMPLETE.json',dict(status='complete',manifest_sha256=sha(a.out/'cohort_manifest.json')))
        print(json.dumps(result['summaries'],indent=2),flush=True)
    except Exception:
        dump(a.out/'failure.json',dict(error=traceback.format_exc()));raise
    finally:
        report=a.out.with_name(a.out.name+'_reports.tar.gz')
        with tarfile.open(report,'w:gz') as tar:
            for f in a.out.glob('*.json'):
                if f.name!='evidence_dictionary.json': tar.add(f,arcname=f.name)
        print('Return report:',report,flush=True)


if __name__=='__main__': main()
