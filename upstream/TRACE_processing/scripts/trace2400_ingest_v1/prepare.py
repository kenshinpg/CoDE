"""Server-only deterministic source adapter. No model calls or clinical aggregation."""
import argparse, hashlib, json, math, pickle, re, time
from collections import Counter
from pathlib import Path

VERSION='trace2400_interface_v0.2'
ASSERTIONS={x:x.lower() for x in ('PRESENT','ABSENT','FAMILY','HYPOTHETICAL','POSSIBLE')}
SOURCES={'Patient History':'history','Physical Examination':'examination','Radiology':'radiology'}
def clean(x):
    if x is None or isinstance(x,(str,bool,int)): return x
    if isinstance(x,float): return x if math.isfinite(x) else None
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if hasattr(x,'item'): return clean(x.item())
    return str(x)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def missing_time(did=None):
    return dict(raw=None,parsed=None,basis='missing',source_document_id=did,missing_reason='document_time_not_supplied' if did else 'measurement_time_not_supplied')
def assertion(raw,lab=False):
    return dict(raw_value=raw,value=ASSERTIONS.get(raw),status='normalized' if raw in ASSERTIONS else ('not_applicable' if lab else 'missing') if raw is None else 'unrecognized',basis='entitycoding_v1' if not lab else 'native_measurement',mapping_version='entitycoding_assertion_v1')
def attributes(raw=None,lab=False):
    return dict(assertion=assertion(raw,lab),**{k:dict(value=None,status='not_assessed',basis=None) for k in ('subject','temporality','site')})
def parse_lab(v):
    out=dict(raw=clean(v),parsed=None,comparator=None,unit_raw=None,unit_canonical=None,parse_status='unparsed',method='conservative_numeric_syntax_v1')
    if v is None or isinstance(v,float) and not math.isfinite(v): out['parse_status']='missing'; return out
    if isinstance(v,(int,float)) and not isinstance(v,bool): out.update(parsed=v,comparator='=',parse_status='parsed'); return out
    if isinstance(v,str):
        m=re.fullmatch(r'\s*(<=|>=|<|>|=)?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\d\s].*)?\s*',v)
        if m:
            n=float(m[2])
            if math.isfinite(n): out.update(parsed=n,comparator=m[1] or '=',unit_raw=m[3].strip() if m[3] else None,parse_status='parsed')
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw-dir',type=Path,required=True);p.add_argument('--entity-dir',type=Path,required=True)
    p.add_argument('--reviewed',type=Path,required=True);p.add_argument('--lab-inventory',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--contract',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    run='ingest_'+str(time.time_ns());files={};counts=Counter();issues=[];docs={};admissions={};original_entities={};hashes={}
    def emit(name,r):
        if name not in files: files[name]=(a.out/(name+'.jsonl')).open('x',encoding='utf-8')
        files[name].write(json.dumps(clean(dict(schema_version=VERSION,run_id=run,**r)),ensure_ascii=False,allow_nan=False)+'\n'); counts[name]+=1
    try:
        inventory=json.loads(a.lab_inventory.read_text(encoding='utf-8-sig'))
        labnames={str(x['itemid']):x for x in inventory} if inventory and 'itemid' in inventory[0] else {}
        # Exact dictionary schema is retained even if no itemid column is recognized.
        hashes[str(a.lab_inventory)]=sha(a.lab_inventory)
        (a.out/'feature_dictionary.json').write_text(json.dumps({'schema_version':VERSION,'run_id':run,'laboratory':inventory,'HPO_dictionary_path':'outputs/hpo/index/terms.json','note':'Native lab dictionary only; HPO terms referenced separately. No experimental feature selection.'},ensure_ascii=False,indent=2),encoding='utf-8')
        for disease in ('appendicitis','cholecystitis','diverticulitis','pancreatitis'):
            f=a.raw_dir/(disease+'_hadm_info_first_diag.pkl'); digest=sha(f);hashes[str(f)]=digest
            # Trusted, user-supplied local dataset only. Do not load arbitrary pickle files.
            with f.open('rb') as stream: dataset=pickle.load(stream)
            for hadm,r in dataset.items():
                aid=str(hadm);pid='cdm_'+aid
                if aid in admissions: raise ValueError('Duplicate admission '+aid)
                admissions[aid]=r
                emit('labels_PRIVATE',dict(admission_id=aid,reference_class=disease,label_source=dict(file=str(f),sha256=digest),discharge_diagnosis=r.get('Discharge Diagnosis'),icd_diagnosis=r.get('ICD Diagnosis'),model_visible=False))
                prefixes=set()
                for rad in r.get('Radiology',[]):
                    m=re.fullmatch(r'(\d+)-[A-Za-z]+-\d+',str(rad.get('Note ID','')))
                    if m: prefixes.add(m[1])
                emit('admission_subject_map_PRIVATE',dict(admission_id=aid,legacy_patient_id=pid,subject_id=None,subject_group='note_prefix_'+next(iter(prefixes)) if len(prefixes)==1 else None,identity_status='inferred' if len(prefixes)==1 else 'unresolved',basis='radiology_note_id_prefix_not_authoritative',candidate_prefixes=sorted(prefixes)))
                for field,source in SOURCES.items():
                    entries=list(enumerate(r.get(field,[]))) if field=='Radiology' else [(None,r.get(field))]
                    present=bool(r.get(field))
                    emit('source_availability_PRIVATE',dict(admission_id=aid,source_type=source,presence_status='present' if present else 'missing',parse_status='raw_preserved',processing_status='pending' if source=='radiology' else 'existing_extraction_to_link'))
                    for i,v in entries:
                        text=v.get('Report') if field=='Radiology' else v
                        did=pid+':'+field+(f':{i}' if i is not None else '')
                        doc=dict(document_id=did,admission_id=aid,source_type=source,raw_text=text if isinstance(text,str) else None,source_locator=dict(file=str(f),file_sha256=digest,field=field,list_index=i),native_metadata=clean({k:x for k,x in v.items() if k!='Report'}) if isinstance(v,dict) else {},sections=[],time=missing_time(did),processing=dict(presence_status='present' if text else 'missing',parse_status='raw_preserved',extraction_status='not_run' if source=='radiology' else 'pending_link',review_status='not_run'))
                        docs[did]=doc;emit('source_documents_PRIVATE',doc)
                        if source=='radiology' and isinstance(text,str) and text.strip(): emit('radiology_entity_input_PRIVATE',dict(document_id=did,patient_id=pid,source_field=field,text=text))
                labs=r.get('Laboratory Tests',{})
                emit('source_availability_PRIVATE',dict(admission_id=aid,source_type='laboratory',presence_status='present' if labs else 'missing',parse_status='per_observation',processing_status='snapshot_parsed'))
                for key,val in labs.items():
                    oid=pid+':lab:'+str(key);value=parse_lab(val)
                    for name,field in [('reference_lower','Reference Range Lower'),('reference_upper','Reference Range Upper')]: value[name]=clean(r.get(field,{}).get(key));value[name+'_status']='native_preserved' if value[name] is not None else 'not_supplied'
                    obs=dict(observation_id=oid,admission_id=aid,source_type='laboratory',document_id=None,source_locator=dict(file=str(f),file_sha256=digest,field='Laboratory Tests',item_key=str(key)),raw=dict(value=clean(val),type=type(val).__name__),concept=dict(namespace='source_lab_itemid',id=str(key),label=labnames.get(str(key),{}).get('label'),mapping_status='native_identifier'),value=value,attributes=attributes(lab=True),time=missing_time(),representation_group_id=oid,derived_from_observation_ids=[],quality=dict(review_status='not_run',clinical_correction_applied=False),provenance=dict(source_version='received_pickle_snapshot',original_record_id=oid))
                    emit('observations_PRIVATE',obs); counts['lab_parse_'+value['parse_status']]+=1
                emit('microbiology_retained_PRIVATE',dict(admission_id=aid,records=r.get('Microbiology'),specimens=r.get('Microbiology Spec'),included_in_first_experiment=False))
                if len(admissions)%200==0: print('Source admissions:',len(admissions),flush=True)
        seen_docs=set()
        for f in sorted(a.entity_dir.glob('node[0-7].jsonl')):
            hashes[str(f)]=sha(f)
            for r in rows(f):
                did=r['document_id']
                if did not in docs or did in seen_docs or r['text']!=docs[did]['raw_text']: raise ValueError('Source extraction mismatch '+did)
                seen_docs.add(did)
                emit('document_processing_PRIVATE',dict(document_id=did,admission_id=docs[did]['admission_id'],extraction_status=r.get('status'),entity_count=len(r['entities'])))
                if not r['entities']: emit('zero_entity_documents_PRIVATE',dict(document_id=did,admission_id=docs[did]['admission_id'],extraction_status=r.get('status'),entity_count=0,reason='unknown',source_locator=docs[did]['source_locator']))
                for e in r['entities']:
                    mid=e['mention_id']
                    if mid in original_entities: raise ValueError('Duplicate mention '+mid)
                    original_entities[mid]=(did,e)
        expected={k for k,d in docs.items() if d['source_type']!='radiology'}
        if seen_docs!=expected: raise ValueError('Missing or extra history/exam extraction documents')
        hashes[str(a.reviewed)]=sha(a.reviewed);seen=set()
        for r in rows(a.reviewed):
            mid=r['mention_id']
            if mid in seen or mid not in original_entities: raise ValueError('Duplicate or foreign reviewed mention '+mid)
            seen.add(mid);did,e=original_entities[mid];d=docs[did]
            start,end=e['raw_start'],e['raw_end']
            if not (0<=start<end<=len(d['raw_text'])) or d['raw_text'][start:end]!=e['raw_quote']: raise ValueError('Invalid span '+mid)
            hpo=(r.get('mapping') or {}).get('hpo_id')
            emit('review_provenance_PRIVATE',dict(observation_id=mid,original_review_record=r))
            emit('observations_PRIVATE',dict(observation_id=mid,admission_id=d['admission_id'],document_id=did,source_type=d['source_type'],source_locator=d['source_locator'],raw=dict(quote=e['raw_quote'],start=start,end=end),concept=dict(namespace='HPO',id=hpo,label=None,mapping_status='mapped' if hpo else 'null_mapping'),value=dict(raw=e['raw_quote'],parsed=None,parse_status='not_applicable'),attributes=attributes(e.get('assertion')),time=missing_time(did),representation_group_id=mid,derived_from_observation_ids=[],quality={k:r.get(k) for k in ('audit_state','audit_flags','audit_task_ids','clinical_correction_applied')},provenance=dict(original_record_id=mid,entity_type=e.get('entity_type'),source_link_file=r.get('source_link_file'),source_review_file=str(a.reviewed))))
        if seen!=set(original_entities): raise ValueError('Reviewed export does not cover all original entities')
        emit('development_exposure_PRIVATE',dict(scope='dataset',status='incomplete_known_exposure_inventory',source='outputs/hpo_review_split_v1',note='Review partitions are not diagnostic splits; consult original manifests before freezing tests.'))
        for f in files.values(): f.flush()
        exposure_dir=Path('outputs/hpo_review_split_v1')
        if exposure_dir.is_dir():
            import shutil
            target=a.out/'review_split_source_PRIVATE';target.mkdir()
            for ef in sorted(exposure_dir.glob('*.json*')):
                shutil.copy2(ef,target/ef.name);hashes[str(ef)]=sha(ef)
        report=dict(status='SOURCE_ADAPTER_COMPLETE_RADIOLOGY_EXTRACTION_PENDING',run_id=run,counts=dict(counts),admissions=len(admissions),documents=len(docs),inputs=hashes,code_sha256=sha(Path(__file__)),issues=issues,semantic_acceptance=False,training_ready=False)
        if a.contract:
            import importlib.util
            spec=importlib.util.spec_from_file_location('code_contract',a.contract);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            report['CoDE_contract']=mod.validate(list(rows(a.out/'source_documents_PRIVATE.jsonl')),list(rows(a.out/'observations_PRIVATE.jsonl')))
            if not report['CoDE_contract']['passed']: raise ValueError('CoDE contract failed: '+str(report['CoDE_contract']['errors'][:10]))
        (a.out/'acceptance_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    except Exception as e:
        (a.out/'FAILED.json').write_text(json.dumps({'error':repr(e),'counts':dict(counts)}),encoding='utf-8');raise
    finally:
        for f in files.values(): f.close()
if __name__=='__main__': main()