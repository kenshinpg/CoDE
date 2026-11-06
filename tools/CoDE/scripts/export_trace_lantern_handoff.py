"""Export accepted frozen TRACE view for shared rollout production; stdlib only."""
import argparse,hashlib,json,shutil,zipfile,tarfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def emit(p,rs):
    with p.open('w',encoding='utf8') as f:
        for r in rs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def export(view,out,rules,accepted,expected=2400):
    if out.exists() or out.resolve().is_relative_to(view.resolve()):raise ValueError('Use a new output directory outside source view')
    names=['view_config.json','view_report.json','evidence_dictionary.json','split_PRIVATE.jsonl','patients_model_input_PRIVATE.jsonl','fact_provenance_PRIVATE.jsonl','train_records_PRIVATE.jsonl','development_records_PRIVATE.jsonl','test_records_PRIVATE.jsonl']
    manifest=read(view/'manifest.json');identities={n:sha(view/n) for n in names}
    for n,h in identities.items():
        if manifest['output_sha256'].get(n)!=h:raise ValueError('Frozen file changed: '+n)
    for n,h in accepted['input_sha256'].items():
        if identities.get(n)!=h:raise ValueError('Different accepted view: '+n)
    dictionary=read(view/'evidence_dictionary.json');cfg=read(view/'view_config.json');patients=rows(view/'patients_model_input_PRIVATE.jsonl');split=rows(view/'split_PRIVATE.jsonl')
    if cfg['view_id']!=accepted['view_id']:raise ValueError('Wrong view identity')
    pi={r['patient_id']:r for r in patients};si={r['admission_id']:r for r in split}
    if len(pi)!=len(patients) or len(si)!=len(split) or set(pi)!=set(si) or len(pi)!=expected:raise ValueError('Admission census differs')
    record={}
    for part in ['train','development','test']:
        for r in rows(view/(part+'_records_PRIVATE.jsonl')):
            pid=r['patient_id']
            if pid in record or pid not in si or si[pid]['partition']!=part or r['partition']!=part or r['context'] or r['label']!=si[pid]['reference_class']:raise ValueError('Structured model/split mismatch')
            record[pid]=r
    if set(record)!=set(pi):raise ValueError('Structured model census differs')
    prov={}
    for r in rows(view/'fact_provenance_PRIVATE.jsonl'):
        if r['fact_id'] in prov:raise ValueError('Duplicate provenance fact identity')
        prov[r['fact_id']]=r
    visible=set();groups={};safe=[];input_digests={};tasks=[]
    allowed_patient={'patient_id','view_id','context','facts'};allowed_fact={'fact_id','feature_id','state','display_text','group_id'}
    for pid in sorted(pi):
        r=pi[pid];sp=si[pid]
        if set(r)!=allowed_patient or r['view_id']!=accepted['view_id'] or r['context'] or not r['facts']:raise ValueError('Unexpected patient fields/context/empty input')
        if sp['subject_group'] in groups and groups[sp['subject_group']]!=sp['partition']:raise ValueError('Subject crosses split')
        groups[sp['subject_group']]=sp['partition'];tokens=[]
        for fact in r['facts']:
            if set(fact)!=allowed_fact:raise ValueError('Unexpected fact fields')
            fid=fact['fact_id'];feature=dictionary['features'].get(fact['feature_id'])
            if fid in visible or fid not in prov or not feature or fact['state'] not in feature['values'] or fact['group_id']!=feature['group_id'] or not isinstance(fact['display_text'],str) or not fact['display_text'].strip():raise ValueError('Invalid visible fact')
            visible.add(fid);tokens.append(fact['feature_id']+'_@_'+fact['state'])
        if len(tokens)!=len(set(tokens)) or sorted(tokens)!=sorted(record[pid]['tokens']):raise ValueError('LLM visible facts differ from model tokens')
        safe.append(r);input_digests[pid]=digest(r)
        for index in range(4):tasks.append(dict(task_id='TRACE_'+digest([accepted['view_id'],pid,index])[:24],patient_id=pid,rollout_index=index,input_view_id=accepted['view_id'],input_digest=input_digests[pid]))
    # Only the exporter accesses outcome columns, for consistency checking.
    # The shared production directory has no reference_class or reference label rows.
    out.mkdir(parents=True);production=out/'production_PRIVATE';production.mkdir();evaluation=out/'evaluation_PRIVATE';evaluation.mkdir()
    emit(production/'patients_model_input_PRIVATE.jsonl',safe)
    emit(production/'fact_provenance_PRIVATE.jsonl',[prov[f] for f in sorted(visible)])
    emit(production/'split_without_outcomes_PRIVATE.jsonl',[{'patient_id':pid,'subject_group':si[pid]['subject_group'],'partition':si[pid]['partition'],'subject_identity_status':'inferred_from_source_note_identifier'} for pid in sorted(pi)])
    emit(production/'generation_tasks_PRIVATE.jsonl',tasks)
    emit(evaluation/'reference_outcomes_PRIVATE.jsonl',[{'patient_id':pid,'reference_class':si[pid]['reference_class']} for pid in sorted(pi)])
    for n in ['view_config.json','evidence_dictionary.json']:shutil.copyfile(view/n,production/n)
    for n in ['matching_rules.json','delivery_contract.json']:shutil.copyfile(rules/n,production/n)
    labels=sorted({r['label'] for r in record.values()})
    dump(production/'diagnosis_vocabulary.json',{'labels':labels,'applies_to_all_admissions':True,'ranked_distinct_diagnoses_required':3})
    glossary=[]
    for fid,f in sorted(dictionary['features'].items()):
        glossary.append(dict(feature_id=fid,concept_id=fid.replace('HP_','HP:',1) if fid.startswith('HP_') else None,concept_system='HPO' if fid.startswith('HP_') else 'native_lab_item',label=f['label'],allowed_states=list(f['values']),group_id=f['group_id'],assertion_rule='present/absent are explicit assertions' if fid.startswith('HP_') else 'LOW/WITHIN_RANGE/HIGH are frozen lab states, not HPO assertions'))
    dump(production/'feature_semantics.json',{'features':glossary,'fact_identity':'Use supplied fact_id; never replace with concept_id','lab_reference_scope':'Snapshot, same-scale paired-reference assumption from frozen view; no new reference interval inference'})
    # Pilot identities chosen only from development, without outcomes or score selection.
    development=sorted((pid for pid in pi if si[pid]['partition']=='development'),key=lambda pid:digest(['pilot',pid]))[:20]
    emit(production/'pilot_tasks_PRIVATE.jsonl',[r for r in tasks if r['patient_id'] in set(development)])
    (production/'README.md').write_text("""# TRACE shared rollout production input

Use this frozen input view for all production tasks. The concept extraction and HPO/lab mapping pipeline remains general and retains extra or contradictory claims.

patients_model_input_PRIVATE.jsonl contains the exact facts used by CoDE models. Render every display_text once in the generation input. Do not render patient/subject identifiers, split membership, provenance, rule metadata or dictionary frequency statistics. The shared four-label vocabulary is allowed. Do not add raw records, demographics, hidden findings or reference outcomes. Store full actual messages, tokenizer length, model identity and request parameters. Overflow must be explicit; no silent truncation.

Feature IDs, legal states and group IDs come from evidence_dictionary.json. feature_semantics.json supplies concept identities. fact_provenance_PRIVATE.jsonl links only visible facts to underlying observations; those linked records are not additional model input. Source documents stay in the TRACE dataset.

Extract what the explanation says before matching against visible facts. Keep extra concepts, partial mentions, negation, uncertainty and contradictions. Map HPO concepts generally; use native items and frozen states for laboratory claims. Missing mention review must not fill in facts absent from the explanation. Follow matching_rules.json for CoDE exact-coverage projection; shared extraction retains all other statuses.

The pilot contains 20 development admissions and 80 logical tasks, selected without reference labels or performance. Full inventory is 2400 admissions by four rollouts. Retries remain attempts of the same task. Retain errors and wrong diagnoses; never overwrite raw responses. Freeze prompt and mapping configuration after the pilot. Production must record deviations for review.

This is a private research handoff, not public supplementary data. Reference outcomes remain outside this directory and outside the shared ZIP. The export contains no generation results; downstream production creates and validates them.
""",encoding='utf8')
    hashes={f.name:sha(f) for f in production.iterdir() if f.is_file()}
    dump(production/'manifest.json',dict(input_view_id=accepted['view_id'],source_sha256=identities,output_sha256=hashes,admissions=len(pi),logical_tasks=len(tasks),matching_rule_id=read(rules/'matching_rules.json')['rule_id'],reference_outcomes_included=False))
    archive=out/'TRACE2400_frozen_rollout_input_PRIVATE.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in production.iterdir():z.write(f,f.name)
    report=dict(status='complete',input_view_id=accepted['view_id'],admissions=len(pi),partitions=dict(Counter(r['partition'] for r in split)),subjects=len(groups),visible_facts=len(visible),logical_tasks=len(tasks),pilot_admissions=len(development),pilot_tasks=4*len(development),dictionary_features=len(dictionary['features']),diagnosis_vocabulary=labels,model_input_tokens_equal_visible_facts=True,reference_outcomes_in_production_zip=False,input_sha256=identities,private_zip_sha256=sha(archive),matching_rule_id=read(rules/'matching_rules.json')['rule_id'])
    dump(out/'handoff_report.json',report)
    with tarfile.open(str(out)+'_reports.tar.gz','w:gz') as t:t.add(out/'handoff_report.json',arcname='handoff_report.json')
    print(json.dumps(report,ensure_ascii=False,indent=2));print('For LANTERN:',archive);print('Return audit:',str(out)+'_reports.tar.gz')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--view',type=Path,default=Path('artifacts/trace2400_view_v2_names'));p.add_argument('--out',type=Path,default=Path('artifacts/trace2400_lantern_input_v1'));p.add_argument('--rules',type=Path,default=ROOT/'configs/trace_lantern_handoff');a=p.parse_args()
    export(a.view,a.out,a.rules,read(a.rules/'accepted_view.json'))
