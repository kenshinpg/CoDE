"""Four-source observation adapter; no model, semantic correction or fitting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

VERSION='trace2400_interface_v0.2'
ASSERTIONS={s:s.lower() for s in ['PRESENT','ABSENT','FAMILY','HYPOTHETICAL','POSSIBLE']}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        for l in f:
            if l.strip():yield json.loads(l)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def line(f,x):f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
def attributes(raw,basis):
    return dict(assertion=dict(raw_value=raw,value=ASSERTIONS.get(raw),
        status='normalized' if raw in ASSERTIONS else 'missing' if raw is None else 'unrecognized',
        basis=basis,mapping_version='entitycoding_assertion_v1'),
        **{k:dict(value=None,status='not_assessed',basis=None) for k in ['subject','temporality','site']})
def doc_time(d):
    t=d['time']
    if t.get('raw') is not None or t.get('parsed') is not None:
        return dict(raw=t.get('raw'),parsed=t.get('parsed'),basis='inherited_document',source_document_id=d['document_id'],missing_reason=None)
    return dict(raw=None,parsed=None,basis='missing',source_document_id=d['document_id'],missing_reason=t.get('missing_reason') or 'document_time_not_supplied')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--radiology',type=Path,required=True)
    p.add_argument('--lab',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    tracked={}
    def track(path):
        path=path.resolve();tracked[str(path)]=sha(path);return path
    source=a.source.resolve();rad=a.radiology.resolve();lab=a.lab.resolve()
    sr=read(track(source/'acceptance_report.json'));rr=read(track(rad/'report.json'));lr=read(track(lab/'report.json'))
    for name in ['observations_PRIVATE.jsonl','labels_PRIVATE.jsonl','feature_dictionary.json']:
        path=track(source/name)
        if tracked[str(path)]!=lr['inputs'][name]:raise ValueError('Lab/source identity mismatch '+name)
    if lr['unit_policy']!='source_paired_assumption':raise ValueError('Unexpected lab policy')
    docs={};labels={};subjects={}
    for r in rows(track(source/'labels_PRIVATE.jsonl')):
        if r['admission_id'] in labels:raise ValueError('Duplicate label ID')
        labels[r['admission_id']]=r
    for r in rows(track(source/'admission_subject_map_PRIVATE.jsonl')):
        if r['admission_id'] in subjects:raise ValueError('Duplicate subject mapping')
        subjects[r['admission_id']]=r
    if set(subjects)!=set(labels):raise ValueError('Subject/label coverage mismatch')
    for d in rows(track(source/'source_documents_PRIVATE.jsonl')):
        if d['document_id'] in docs or d['admission_id'] not in labels:raise ValueError('Invalid document linkage')
        docs[d['document_id']]=d
    # Existing history/exam and lab observations are retained, not recomputed.
    observations=[];seen={};groups={};counts=Counter();by_admission={aid:Counter() for aid in labels}
    samples=Counter()
    def add(o):
        oid=o['observation_id'];aid=o['admission_id'];gid=o['representation_group_id']
        if oid in seen or aid not in labels:raise ValueError('Duplicate/orphan observation '+oid)
        if gid in groups and groups[gid]!=aid:raise ValueError('Cross-admission source group')
        groups[gid]=aid;seen[oid]=o;observations.append(o)
        counts[o['source_type']]+=1;by_admission[aid][o['source_type']]+=1
    for o in rows(source/'observations_PRIVATE.jsonl'):add(o)
    print('Loaded source observations:',len(observations),flush=True)
    rad_count=Counter();run='unified_'+str(time.time_ns())
    with (a.out/'radiology_provenance_PRIVATE.jsonl').open('x',encoding='utf-8') as provenance:
        for r in rows(track(rad/'mapping_review_records_PRIVATE.jsonl')):
            u=r['unit'];did=u['document_id'];d=docs.get(did)
            if not d or d['source_type']!='radiology':raise ValueError('Unknown radiology document')
            aid=d['admission_id']
            if u['patient_id']!='cdm_'+aid:raise ValueError('Radiology admission mismatch')
            e=u['entity'];original=u['original_entity'];kind=u['unit_kind']
            if kind not in ['parent','atom']:raise ValueError('Unexpected unit kind')
            # Preserve original classifier assertion; proposed atom assertion remains separate.
            original_assertion=original.get('assertion')
            deps=[] if kind=='parent' else [u['parent_mention_id']+':parent']
            if u.get('derived_from_unit_ids',[])!=deps:raise ValueError('Unexpected parent dependency')
            o=dict(schema_version=VERSION,run_id=run,observation_id=u['unit_id'],admission_id=aid,
                document_id=did,source_type='radiology',source_locator=d['source_locator'],
                raw=dict(quote=e['raw_quote'],start=e['raw_start'],end=e['raw_end']),
                concept=dict(namespace='HPO',id=None,label=None,mapping_status='see_concept_links',
                    note='Zero/multiple original links; not a single clinical fact'),
                concept_links=(r.get('mapping') or {}).get('links',[]),
                proposed_concept_links=(r.get('review') or {}).get('links',[]),
                value=dict(raw=e['raw_quote'],parsed=None,parse_status='not_applicable'),
                attributes=attributes(original_assertion,'original_parent_entitycoding_assertion'),
                proposed_attributes=dict(assertion=u.get('proposed_assertion'),role=u.get('proposed_role'),applied=False),
                time=doc_time(d),representation_group_id=u['representation_group_id'],derived_from_observation_ids=deps,
                quality=dict(map_status=r['map_status'],review_status=r['review_status'],
                    mapping_status=(r.get('mapping') or {}).get('status'),verdict=(r.get('review') or {}).get('verdict'),
                    issues=(r.get('review') or {}).get('issues',[]),clinical_correction_applied=False,clinical_fact_ready=False),
                provenance=dict(unit_kind=kind,parent_mention_id=u['parent_mention_id'],original_record_id=u['unit_id'],
                    assertion_scope='parent_mention; child-specific status not independently established'))
            add(o);line(provenance,dict(observation_id=o['observation_id'],record=r))
            rad_count['units']+=1;rad_count[kind]+=1
            if rad_count['units']%20000==0:print('Radiology units:',rad_count['units'],flush=True)
    if rad_count['units']!=rr['counts']['units'] or rad_count['parent']!=rr['parents']:raise ValueError('Radiology coverage mismatch')
    import code_contracts
    print('Validating document/span/assertion/dependency interface...',flush=True)
    check=code_contracts.validate(list(docs.values()),observations)
    if not check['passed']:
        save(a.out/'FAILED_contract.json',check);raise ValueError('Interface validation failed')
    # Verify lab facts against original observations and derivations; preserve hidden values separately.
    expected={};derivations=0;lab_ids=set()
    for r in rows(track(lab/'derivations_PRIVATE.jsonl')):
        oid=r['observation_id'];o=seen.get(oid)
        if oid in lab_ids or not o or o['source_type']!='laboratory' or r['source_observation']!=o:raise ValueError('Lab derivation identity mismatch')
        lab_ids.add(oid);derivations+=1
        if r['derivation']['state'] is not None:
            if r['derivation']['reasons']:raise ValueError('Visible lab has exclusion reasons')
            expected[r['fact_id']]=(oid,r['admission_id'],r['representation_group_id'],r['derivation']['state'],'LAB_'+str(o['concept']['id']))
    if derivations!=counts['laboratory']:raise ValueError('Lab derivation coverage')
    used=set();patient_ids=set();lab_visible=Counter()
    allowed={'fact_id','feature_id','state','display_text','representation_group_id','source_observation_ids'}
    for r in rows(track(lab/'patient_lab_view_PRIVATE.jsonl')):
        aid=r['admission_id']
        if aid not in labels or aid in patient_ids or r['view_id']!=lr['view_id']:raise ValueError('Lab patient view identity')
        patient_ids.add(aid)
        for f in r['facts']:
            fid=f['fact_id']
            if set(f)!=allowed or fid in used or fid not in expected:raise ValueError('Lab fact invalid/leaked fields')
            oid,ea,gid,state,feature=expected[fid]
            if (aid,f['representation_group_id'],f['state'],f['feature_id'],f['source_observation_ids'])!=(ea,gid,state,feature,[oid]):raise ValueError('Lab fact linkage mismatch')
            used.add(fid);lab_visible[aid]+=1
    if used!=set(expected) or patient_ids!=set(labels):raise ValueError('Lab view coverage mismatch')
    print('Lab facts verified:',len(used),flush=True)
    with (a.out/'observations_PRIVATE.jsonl').open('x',encoding='utf-8') as out,(a.out/'observation_samples_PRIVATE.jsonl').open('x',encoding='utf-8') as sample:
        for o in observations:
            line(out,o);kind=o['source_type']
            if samples[kind]<5:line(sample,o);samples[kind]+=1
    with (a.out/'admission_coverage_PRIVATE.jsonl').open('x',encoding='utf-8') as out:
        for aid,c in by_admission.items():line(out,dict(admission_id=aid,observations=dict(c),visible_lab_facts=lab_visible[aid]))
    copies={
        'source_documents_PRIVATE.jsonl':source/'source_documents_PRIVATE.jsonl',
        'labels_PRIVATE.jsonl':source/'labels_PRIVATE.jsonl',
        'admission_subject_map_PRIVATE.jsonl':source/'admission_subject_map_PRIVATE.jsonl',
        'patient_lab_view_PRIVATE.jsonl':lab/'patient_lab_view_PRIVATE.jsonl',
        'laboratory_derivations_PRIVATE.jsonl':lab/'derivations_PRIVATE.jsonl',
        'lab_feature_dictionary.json':lab/'feature_dictionary.json',
        'source_feature_dictionary.json':source/'feature_dictionary.json'}
    for name in ['review_provenance_PRIVATE.jsonl','development_exposure_PRIVATE.jsonl','microbiology_retained_PRIVATE.jsonl','document_processing_PRIVATE.jsonl','source_availability_PRIVATE.jsonl']:
        if (source/name).exists():copies[name]=source/name
    for name,path in copies.items():track(path);shutil.copyfile(path,a.out/name)
    if (source/'review_split_source_PRIVATE').is_dir():
        for path in (source/'review_split_source_PRIVATE').rglob('*'):
            if path.is_file():track(path)
        shutil.copytree(source/'review_split_source_PRIVATE',a.out/'review_split_source_PRIVATE')
    for path,digest in tracked.items():
        if sha(path)!=digest:raise ValueError('Input changed during build '+path)
    report=dict(dataset='TRACE-2400-v1',run_id=run,status='FOUR_SOURCE_OBSERVATIONS_INTERFACE_ACCEPTED',
        admissions=len(labels),documents=len(docs),source_document_counts=dict(Counter(d['source_type'] for d in docs.values())),
        observation_counts=dict(counts),radiology_counts=dict(rad_count),lab_visible_facts=len(used),
        admissions_with_visible_labs=len(lab_visible),lab_view_id=lr['view_id'],unit_assumption=lr['assumption'],
        interface_check=check,semantic_accuracy_validated=False,combined_experiment_view_frozen=False,
        components=dict(source=str(source),radiology=str(rad),laboratory=str(lab)),input_sha256=tracked,
        source_acceptance=sr.get('status'),radiology_acceptance=rr,lab_acceptance=lr,
        notes=['Observation interface only. Text clinical fact policy not frozen.',
        'Do not sum parent/atom/HPO alternatives as independent evidence.',
        'Source processing flags are historical. Component acceptance above is current.',
        'Labs raw and derived view remain separate. Main consumers must not read hidden values.'])
    save(a.out/'acceptance_report.json',report)
    save(a.out/'manifest.json',dict(dataset='TRACE-2400-v1',run_id=run,
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        output_sha256={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps({k:report[k] for k in ['status','admissions','documents','observation_counts','lab_visible_facts']},indent=2),flush=True)

if __name__=='__main__':main()
