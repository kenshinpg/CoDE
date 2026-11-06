"""Build a private TRACE common input and group split. No fitting/test scoring."""
import argparse,json,re,sys,tarfile,time
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audit_trace2400_delivery import rows,read,sha,dump
from code_eval.trace2400.view import digest,original_links,choose,grouped_split

def emit(path,records):
    with path.open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def ontology_names(path):
    result={}
    if path:
        cid=None
        for line in path.read_text(encoding='utf-8').splitlines():
            if line=='[Term]':cid=None
            elif line.startswith('id: HP:'):cid=line[4:]
            elif line.startswith('name: ') and cid:result[cid]=line[6:]
    return result

def build(a):
    root=a.data.resolve()
    if not (root/'manifest.json').exists() and (root/'data/manifest.json').exists():root=root/'data'
    if a.out.resolve().is_relative_to(root):raise ValueError('Output must be outside production')
    a.out.mkdir(parents=True,exist_ok=False)
    cfg=read(a.config);required_policy={'text_mapping':'original_only','radiology_unit':'parent_only','context_features':[], 'text_assertions':['present','absent'], 'multiple_concepts':'exclude_observation_keep_ledger', 'opposite_assertions':'exclude_admission_concept_keep_ledger', 'duplicate_concepts':'collapse_across_sources_by_hpo_id', 'diagnosis_filter':'exclude_technical_group_with_keyword_in_quote_or_original_concept_label', 'post_treatment':'retain_under_retrospective_task_and_report'}
    for k,v in required_policy.items():
        if cfg[k]!=v:raise ValueError('Unsupported policy change')
    manifest=read(root/'manifest.json');used=['labels_PRIVATE.jsonl','admission_subject_map_PRIVATE.jsonl','source_documents_PRIVATE.jsonl','observations_PRIVATE.jsonl','patient_lab_view_PRIVATE.jsonl','lab_feature_dictionary.json','acceptance_report.json']
    hashes={}
    print('Stage 1: bind accepted production inputs',flush=True)
    for n in used:
        hashes[n]=sha(root/n)
        if hashes[n]!=manifest['output_sha256'].get(n):raise ValueError('Input identity mismatch')
    report=read(root/'acceptance_report.json')
    if report.get('admissions')!=2400 or not report.get('interface_check',{}).get('passed'):raise ValueError('Unaccepted production input')
    labels={r['admission_id']:r['reference_class'] for r in rows(root/'labels_PRIVATE.jsonl')}
    sm={r['admission_id']:r for r in rows(root/'admission_subject_map_PRIVATE.jsonl')}
    if set(labels)!=set(sm) or len(labels)!=2400:raise ValueError('Admission/subject coverage')
    subjects={aid:r.get('subject_group') for aid,r in sm.items()}
    partition,split_counts=grouped_split(labels,subjects,cfg)
    dump(a.out/'view_config.json',cfg)
    emit(a.out/'split_PRIVATE.jsonl',({'admission_id':aid,'subject_group':subjects[aid],'partition':partition[aid],'reference_class':labels[aid]} for aid in sorted(labels)))
    print('Stage 2: text names and diagnosis-group screen',flush=True)
    names=ontology_names(a.hpo_obo);name_variants=defaultdict(set);blocked=set();pat=re.compile(cfg['diagnosis_name_pattern'],re.I)
    for o in rows(root/'observations_PRIVATE.jsonl'):
        if o['source_type']=='laboratory':continue
        links=original_links(o)
        for c,n in links:
            if n:name_variants[c].add(n)
        if pat.search(o.get('raw',{}).get('quote','')) or any(pat.search(n or '') for c,n in links):blocked.add((o['admission_id'],o['representation_group_id']))
    for c,ns in name_variants.items():
        if c not in names:names[c]=sorted(ns)[0]
    # A canonical label can reveal a direct diagnosis even when the quote does not.
    for o in rows(root/'observations_PRIVATE.jsonl'):
        if o['source_type']!='laboratory' and any(pat.search(names.get(c,'')) for c,n in original_links(o)):blocked.add((o['admission_id'],o['representation_group_id']))
    candidates=defaultdict(dict);counts=Counter();by_source=defaultdict(Counter)
    with (a.out/'observation_decisions_PRIVATE.jsonl').open('w',encoding='utf-8') as ledger:
        for i,o in enumerate(rows(root/'observations_PRIVATE.jsonl'),1):
            why,c,state=choose(o,blocked);source=o['source_type'];counts[why]+=1;by_source[source][why]+=1
            ledger.write(json.dumps({'observation_id':o['observation_id'],'reason':why,'concept_id':c,'assertion':state})+'\n')
            if why=='candidate':
                aid=o['admission_id'];entry=candidates[aid].setdefault(c,{'states':set(),'ids':[],'sources':set(),'groups':set()})
                entry['states'].add(state);entry['ids'].append(o['observation_id']);entry['sources'].add(source);entry['groups'].add(o['representation_group_id'])
            if i%100000==0:print('Observation decisions',i,flush=True)
    print('Stage 3: deduplicate facts and retain conflict ledger',flush=True)
    facts={a:[] for a in labels};provenance=[];conflicts=[]
    for aid,cs in sorted(candidates.items()):
        for c,e in sorted(cs.items()):
            if len(e['states'])!=1:
                conflicts.append({'admission_id':aid,'concept_id':c,'states':sorted(e['states']),'source_observation_ids':e['ids']});continue
            feature=c.replace(':','_');state=next(iter(e['states']));fid='F_'+digest([aid,feature,state])[:24]
            name=names.get(c);display=(('Record mentions: ' if state=='present' else 'Record explicitly denies: ')+name) if name else None
            facts[aid].append({'fact_id':fid,'feature_id':feature,'state':state,'display_text':display,'group_id':feature})
            provenance.append({'fact_id':fid,'source_observation_ids':sorted(e['ids']),'sources':sorted(e['sources']),'technical_groups':sorted(e['groups'])})
    lab_dict={r['feature_id']:r for r in read(root/'lab_feature_dictionary.json')['features']}
    lab_seen=set()
    for r in rows(root/'patient_lab_view_PRIVATE.jsonl'):
        aid=r['admission_id']
        if r['view_id']!=cfg['lab_view_id'] or aid not in labels or aid in lab_seen:raise ValueError('Lab view identity')
        lab_seen.add(aid)
        for f in r['facts']:
            facts[aid].append({k:f[k] for k in ['fact_id','feature_id','state','display_text']}|{'group_id':f['feature_id']})
            provenance.append({'fact_id':f['fact_id'],'source_observation_ids':f['source_observation_ids'],'sources':['laboratory'],'technical_groups':[f['representation_group_id']]})
    if lab_seen!=set(labels):raise ValueError('Missing lab patient')
    freq=defaultdict(set)
    for aid,fs in facts.items():
        if partition[aid]=='train':
            for f in fs:freq[f['feature_id']].add(subjects[aid])
    keep={k for k,v in freq.items() if len(v)>=cfg['minimum_training_subjects_per_feature']}
    dictionary={'features':{},'fixed_context':[],'view_id':cfg['view_id']}
    for k in sorted(keep):
        lab=k.startswith('LAB_');values=['LOW','WITHIN_RANGE','HIGH'] if lab else ['present','absent']
        label=lab_dict[k]['label'] if lab else names.get(k.replace('HP_','HP:',1))
        dictionary['features'][k]={'type':'C','values':{v:v for v in values},'group_id':k,'label':label,'training_subjects':len(freq[k])}
    dump(a.out/'evidence_dictionary.json',dictionary)
    emit(a.out/'fact_provenance_PRIVATE.jsonl',provenance);emit(a.out/'conflicts_PRIVATE.jsonl',conflicts)
    records=[];patients=[];losses=Counter();missing_names=set();empty=Counter();sizes=defaultdict(list);unknown=Counter()
    known_tokens={f['feature_id']+'_@_'+f['state'] for aid,fs in facts.items() if partition[aid]=='train' for f in fs if f['feature_id'] in keep}
    for aid in sorted(labels):
        visible=[f for f in facts[aid] if f['feature_id'] in keep]
        losses[partition[aid]]+=len(facts[aid])-len(visible)
        for f in visible:
            if not f['display_text']:missing_names.add(f['feature_id'])
        tokens=sorted(f['feature_id']+'_@_'+f['state'] for f in visible)
        if len(tokens)!=len(set(tokens)):raise ValueError('Duplicate final feature state')
        empty[partition[aid]]+=not tokens;sizes[partition[aid]].append(len(tokens));unknown[partition[aid]]+=len(set(tokens)-known_tokens)
        records.append({'input_group_id':aid,'patient_id':aid,'pattern_id':digest(tokens),'tokens':tokens,'context':{},'label':labels[aid],'partition':partition[aid]})
        patients.append({'patient_id':aid,'view_id':cfg['view_id'],'context':{},'facts':visible})
    emit(a.out/'patients_model_input_PRIVATE.jsonl',patients)
    for p in cfg['split_fractions']:emit(a.out/(p+'_records_PRIVATE.jsonl'),(r for r in records if r['partition']==p))
    # Test is stored separately and is not read by subsequent development-only fits.
    token_sets={p:{r['pattern_id'] for r in records if r['partition']==p} for p in cfg['split_fractions']}
    collisions={x+'__'+y:len(token_sets[x]&token_sets[y]) for x,y in [('train','development'),('train','test'),('development','test')]}
    for n,h in hashes.items():
        if sha(root/n)!=h:raise ValueError('Source changed during processing')
    exposure_path=root/'development_exposure_PRIVATE.jsonl'
    if exposure_path.exists():
        # Preserve privately for interpretation; do not assume an existing file means no exposure.
        (a.out/'development_exposure_PRIVATE.jsonl').write_bytes(exposure_path.read_bytes())
    summary={'status':'INPUT_VIEW_PREPARED_FOR_ACCEPTANCE','view_id':cfg['view_id'],'admissions':len(labels),'subject_groups':len(set(subjects.values())),
      'partition_by_class':split_counts,'observation_decisions':dict(counts),'decisions_by_source':{k:dict(v) for k,v in by_source.items()},
      'conflicting_admission_concepts_excluded':len(conflicts),'blocked_technical_groups':len(blocked),'training_features':len(keep),
      'fact_counts':{p:{'n':len(v),'min':min(v),'median':sorted(v)[len(v)//2],'max':max(v)} for p,v in sizes.items()},
      'training_frequency_filtered_facts':dict(losses),'empty_input_admissions':dict(empty),'unseen_training_state_occurrences':dict(unknown),
      'cross_partition_exact_patterns':collisions,'missing_display_names':sorted(missing_names),'common_llm_input_ready':not missing_names,
      'clinical_leakage_cleared':False,'development_exposure_cleared':False,'formal_training_ready':False,
      'notes':['Source selection and input loss require acceptance before formal fitting.', 'Keywords screen direct names only; no pre-diagnosis or current-patient guarantee.', 'Patient tokens contain no age/sex placeholders.','All admissions retained. Full text is not part of LLM model input.','Raw IDs and private provenance must not be rendered in diagnosis prompts.']}
    dump(a.out/'view_report.json',summary)
    dump(a.out/'manifest.json',{'input_sha256':hashes,'config_sha256':sha(a.config),'hpo_obo_sha256':sha(a.hpo_obo) if a.hpo_obo else None,
      'code_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).resolve().parents[1]/'src/code_eval/trace2400/view.py']},
      'output_sha256':{p.name:sha(p) for p in a.out.iterdir() if p.is_file()}})
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/trace2400_view_v1.json');p.add_argument('--hpo-obo',type=Path)
    a=p.parse_args()
    if a.out.exists():p.error('Output exists; use a new directory')
    if a.out.resolve().is_relative_to(a.data.resolve()):p.error('Output must be outside production data')
    try:r=build(a)
    except Exception as e:
        import traceback
        r={'status':'FAILED','exception_type':type(e).__name__,'locations':[{'module':Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]}
        # Never write a report into a preexisting directory after refusing reuse.
        if a.out.exists() and isinstance(e,FileExistsError):raise SystemExit('Output exists; use a new output directory')
        if not a.out.exists():raise
        dump(a.out/'view_report.json',r)
    with tarfile.open(str(a.out)+'_reports.tar.gz','w:gz') as t:t.add(a.out/'view_report.json',arcname='view_report.json')
    print(json.dumps(r,ensure_ascii=False,indent=2),flush=True);print('Return report:',str(a.out)+'_reports.tar.gz',flush=True)
    if r['status']=='FAILED':sys.exit(1)
if __name__=='__main__':main()
