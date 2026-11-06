"""Read-only TRACE2400 delivery audit. Standard library; no API, GPU or fitting."""
import argparse, hashlib, json, re, sys, tarfile, time
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from code_eval.trace2400.contracts import validate
from code_eval.trace2400.lab_reference_v2 import classify

REQUIRED = ['manifest.json','acceptance_report.json','labels_PRIVATE.jsonl',
 'admission_subject_map_PRIVATE.jsonl','source_documents_PRIVATE.jsonl','observations_PRIVATE.jsonl',
 'patient_lab_view_PRIVATE.jsonl','laboratory_derivations_PRIVATE.jsonl','lab_feature_dictionary.json']
EXPECTED_CLASSES = {'appendicitis':957,'cholecystitis':648,'diverticulitis':257,'pancreatitis':538}
EXPECTED_SOURCES = {'history':69934,'examination':46933,'radiology':234113,'laboratory':138788}
DIAG = re.compile(r'\b(?:appendicitis|cholecystitis|diverticulitis|pancreatitis)\b',re.I)
POST = re.compile(r'\b(?:postoperative|post-operative|status post|s/p|appendectomy|cholecystectomy)\b',re.I)
ALLOWED_FACT = {'fact_id','feature_id','state','display_text','representation_group_id','source_observation_ids'}
PHRASES = {'LOW':'below the supplied reference interval','WITHIN_RANGE':'within the supplied reference interval','HIGH':'above the supplied reference interval'}

def rows(path):
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def stats(values):
    v=sorted(values)
    return {'n':len(v),'min':v[0] if v else None,'median':v[len(v)//2] if v else None,'max':v[-1] if v else None,'zero':v.count(0)}

def audit(root,strict_expected=True):
    errors=Counter(); warnings=Counter()
    def check(ok,code):
        if not ok:errors[code]+=1
    absent=[n for n in REQUIRED if not (root/n).is_file()]
    if absent:return {'audit_completed':False,'missing_files':absent,'training_ready':False}
    manifest=read(root/'manifest.json'); acceptance=read(root/'acceptance_report.json')
    hashes=manifest.get('output_sha256',{});check(bool(hashes),'manifest_no_outputs')
    print('Stage 1: verify delivery identity',flush=True)
    tracked={}; verified=0
    for name,expected in hashes.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):check(False,'manifest_path_escape');continue
        if not path.is_file():check(False,'manifest_file_missing');continue
        tracked[name]=sha(path);check(tracked[name]==expected,'manifest_hash_mismatch');verified+=1
    for name in REQUIRED:
        if name!='manifest.json':check(name in hashes,'required_file_not_manifested')
    before={n:(root/n).stat() for n in REQUIRED}
    print('Stage 2: admission, document and observation contracts',flush=True)
    labels={};subjects={}
    for r in rows(root/'labels_PRIVATE.jsonl'):
        aid=r['admission_id'];check(aid not in labels,'duplicate_admission_label');labels[aid]=r['reference_class']
    for r in rows(root/'admission_subject_map_PRIVATE.jsonl'):
        aid=r['admission_id'];check(aid not in subjects,'duplicate_subject_mapping');subjects[aid]=r
    check(set(labels)==set(subjects),'subject_label_coverage')
    group_members=defaultdict(set);authoritative=0
    for aid,s in subjects.items():
        g=s.get('subject_group');check(bool(g),'subject_group_missing')
        if g:group_members[g].add(aid)
        authoritative+=bool(s.get('subject_id'))
    docs=list(rows(root/'source_documents_PRIVATE.jsonl'));observations=[]
    for o in rows(root/'observations_PRIVATE.jsonl'):
        observations.append(o)
        if len(observations)%100000==0:print('Read observations',len(observations),flush=True)
    contract=validate(docs,observations)
    for e in contract.pop('errors'):errors['contract/'+e.split(': ',1)[-1]]+=1
    index={o['observation_id']:o for o in observations}
    source_counts=Counter();attributes=Counter();quality=Counter();availability=defaultdict(set)
    feature_stats=defaultdict(Counter);patient_concepts=defaultdict(set);assertion_sets=defaultdict(set)
    grouped=defaultdict(set);diagnosis_hits=Counter();post_hits=Counter();linked_diag=Counter()
    for d in docs:
        check(d['admission_id'] in labels,'orphan_document')
        availability[d['admission_id']].add(d['source_type'])
        if DIAG.search(d.get('raw_text','')):diagnosis_hits[d['source_type']]+=1
        if POST.search(d.get('raw_text','')):post_hits[d['source_type']]+=1
    for o in observations:
        aid=o['admission_id'];source=o['source_type'];check(aid in labels,'orphan_observation');source_counts[source]+=1
        if source=='laboratory':availability[aid].add(source);continue
        kind=(o.get('provenance') or {}).get('unit_kind','text');tag=source+'/'+kind
        a=(o.get('attributes') or {}).get('assertion',{}).get('value')
        attributes[tag+'/assertion/'+str(a)]+=1
        attributes[tag+'/time/'+str(o.get('time',{}).get('basis'))]+=1
        for key in ('subject','temporality','site'):
            attributes[tag+'/'+key+'/'+str(o.get('attributes',{}).get(key,{}).get('status'))]+=1
        for key in ('audit_state','map_status','review_status','verdict'):
            if key in o.get('quality',{}):quality[tag+'/'+key+'/'+str(o['quality'][key])]+=1
        links=o.get('concept_links',[]) if source=='radiology' else [{'hpo_id':(o.get('concept') or {}).get('id')}]
        concepts={l.get('hpo_id') for l in links if l.get('hpo_id')}
        feature_stats[tag]['observations']+=1;feature_stats[tag]['with_original_mapping']+=bool(concepts)
        feature_stats[tag]['multiple_concepts']+=len(concepts)>1
        proposed={l.get('hpo_id') for l in o.get('proposed_concept_links',[]) if l.get('hpo_id')}
        if source=='radiology':feature_stats[tag]['original_vs_proposal_set_diff']+=concepts!=proposed
        if concepts and DIAG.search(o.get('raw',{}).get('quote','')):linked_diag[tag]+=1
        for c in concepts:
            patient_concepts[(aid,tag)].add(c);assertion_sets[(aid,c)].add(a)
            grouped[(aid,o['representation_group_id'],c)].add(kind)
    print('Stage 3: independently join and replay lab derivations',flush=True)
    dictionary_rows=read(root/'lab_feature_dictionary.json')['features']
    dictionary={x['feature_id']:x for x in dictionary_rows}
    check(len(dictionary)==len(dictionary_rows),'duplicate_lab_dictionary_feature')
    for f in dictionary.values():check(set(f.get('states',[]))==set(PHRASES),'lab_dictionary_domain')
    expected={};seen=set();lab_states=Counter();reasons=Counter();unit_basis=Counter()
    for i,r in enumerate(rows(root/'laboratory_derivations_PRIVATE.jsonl'),1):
        oid=r['observation_id'];o=index.get(oid)
        check(oid not in seen,'duplicate_lab_derivation');seen.add(oid)
        check(o is not None and o.get('source_type')=='laboratory','orphan_lab_derivation')
        if o is None:continue
        check(r.get('source_observation')==o,'lab_source_record_mismatch')
        check(r.get('admission_id')==o['admission_id'],'lab_derivation_admission_mismatch')
        recomputed=classify(o,unit_policy='source_paired_assumption')
        feature='LAB_'+str(o['concept']['id']);meta=dictionary.get(feature,{})
        if not meta.get('label'):
            recomputed['reasons'].append('dictionary_label_missing');recomputed['state']=None
        check(recomputed==r['derivation'],'lab_rule_replay_mismatch')
        st=recomputed['state'];lab_states[str(st)]+=1;reasons.update(recomputed['reasons'])
        unit_basis[recomputed['unit_comparability']['basis']]+=1
        if st:
            fid=r['fact_id'];check(fid not in expected,'duplicate_derived_fact')
            label=meta['label']+((' ('+meta['fluid']+')') if meta.get('fluid') else '')
            expected[fid]=(o['admission_id'],feature,st,o['representation_group_id'],[oid],label+': '+PHRASES[st])
        if i%40000==0:print('Rechecked lab snapshots',i,flush=True)
    check(seen=={o['observation_id'] for o in observations if o['source_type']=='laboratory'},'lab_derivation_coverage')
    used=set();patients=set();lab_per_patient=[];view_ids=set()
    for r in rows(root/'patient_lab_view_PRIVATE.jsonl'):
        aid=r['admission_id'];check(aid in labels and aid not in patients,'lab_patient_coverage_duplicate')
        check(set(r)=={'admission_id','view_id','facts'},'lab_patient_unexpected_fields')
        patients.add(aid);view_ids.add(r['view_id']);features=set();lab_per_patient.append(len(r['facts']))
        for f in r['facts']:
            fid=f['fact_id'];check(set(f)==ALLOWED_FACT,'lab_fact_unexpected_fields')
            check(fid not in used and fid in expected,'lab_fact_duplicate_or_unexpected');used.add(fid)
            check(f['feature_id'] not in features,'duplicate_lab_project_per_admission');features.add(f['feature_id'])
            actual=(aid,f['feature_id'],f['state'],f['representation_group_id'],f['source_observation_ids'],f['display_text'])
            check(actual==expected.get(fid),'lab_fact_link_or_display_mismatch')
    check(used==set(expected),'lab_visible_fact_coverage');check(patients==set(labels),'lab_admission_coverage')
    check(view_ids=={acceptance.get('lab_view_id')},'lab_view_identity')
    classes=Counter(labels.values())
    if strict_expected:
        check(dict(classes)==EXPECTED_CLASSES,'delivery_label_counts_drift')
        check(dict(source_counts)==EXPECTED_SOURCES,'delivery_observation_counts_drift')
        check(len(docs)==10760,'delivery_document_counts_drift');check(len(used)==94069,'delivery_lab_counts_drift')
    check(acceptance.get('admissions')==len(labels),'acceptance_admission_mismatch')
    check(acceptance.get('observation_counts')==dict(source_counts),'acceptance_source_mismatch')
    check(acceptance.get('lab_visible_facts')==len(used),'acceptance_lab_mismatch')
    for name,stat in before.items():
        after=(root/name).stat();check((stat.st_size,stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'input_changed_during_audit')
    optional={n:(root/n).is_file() for n in ['development_exposure_PRIVATE.jsonl','document_processing_PRIVATE.jsonl','source_availability_PRIVATE.jsonl','radiology_provenance_PRIVATE.jsonl','review_provenance_PRIVATE.jsonl']}
    exposure=Counter()
    if optional['development_exposure_PRIVATE.jsonl']:
        for r in rows(root/'development_exposure_PRIVATE.jsonl'):
            # Only schema keys are returned: no identities, notes or source text.
            exposure.update(r.keys())
    result=dict(audit_completed=True,structural_pass=not errors,error_counts=dict(errors),
        manifest_files_verified=verified,interface=contract,admissions=len(labels),classes=dict(classes),
        subject_groups=len(group_members),authoritative_subject_id_rows=authoritative,
        repeated_subject_groups=sum(len(v)>1 for v in group_members.values()),
        cross_class_subject_groups=sum(len({labels.get(a) for a in v})>1 for v in group_members.values()),
        source_counts=dict(source_counts),source_available_admissions={s:sum(s in availability[a] for a in labels) for s in EXPECTED_SOURCES},
        attributes=dict(attributes),quality=dict(quality),text_mapping={k:dict(v) for k,v in feature_stats.items()},
        text_concept_counts_by_admission={tag:stats([len(patient_concepts[(a,tag)]) for a in labels]) for tag in feature_stats},
        present_absent_admission_concept_pairs=sum({'present','absent'}<=s for s in assertion_sets.values()),
        parent_atom_same_concept_groups=sum({'parent','atom'}<=s for s in grouped.values()),
        diagnosis_name_document_hits=dict(diagnosis_hits),post_treatment_document_hits=dict(post_hits),
        diagnosis_name_linked_observation_hits=dict(linked_diag),
        lab_states=dict(lab_states),lab_exclusion_reasons=dict(reasons),lab_unit_basis=dict(unit_basis),
        lab_facts_per_admission=stats(lab_per_patient),optional_files=optional,development_exposure_field_counts=dict(exposure),
        training_ready=False,semantic_accuracy_validated=False,
        pending=['Freeze text mapping/assertion/parent-atom selection and evidence grouping',
         'Define diagnostic-name and post-treatment visibility policy; keyword screening is not adjudication',
         'Freeze subject-group train/development/test split and document prior development exposure',
         'Fit vocabulary/feature selection on training only and adapt existing CoDE encoders',
         'Generate and align rationales from the same frozen view for natural-rationale RQ2'],
        scope='Structural audit plus feasibility counts. Same lab rule is replayed from preserved implementation; this does not independently validate reference intervals or semantics. No experiment view or split is created.')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.data.resolve()
    if not (root/'manifest.json').exists() and (root/'data'/'manifest.json').exists():root=root/'data'
    out=a.out.resolve()
    if out==root or out.is_relative_to(root):p.error('Output must be outside production data')
    out.mkdir(parents=True,exist_ok=False);started=time.time()
    try:report=audit(root)
    except Exception as e:
        # Do not copy exceptions that may contain identifiers or clinical source text.
        report={'audit_completed':False,'structural_pass':False,'training_ready':False,'exception_type':type(e).__name__}
        import traceback
        report['error_locations']=[{'module':Path(x.filename).name,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(e.__traceback__)]
    report['elapsed_seconds']=round(time.time()-started,2);dump(out/'audit_summary.json',report)
    with tarfile.open(str(out)+'_reports.tar.gz','w:gz') as tar:tar.add(out/'audit_summary.json',arcname='audit_summary.json')
    print(json.dumps({k:report.get(k) for k in ['audit_completed','structural_pass','admissions','training_ready','error_counts','exception_type']},ensure_ascii=False),flush=True)
    print('Return report:',str(out)+'_reports.tar.gz',flush=True)
    if not report.get('structural_pass'):sys.exit(1)
if __name__=='__main__':main()
