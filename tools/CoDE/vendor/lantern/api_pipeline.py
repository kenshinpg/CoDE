"""Frozen requests, resumable API generation and independent evidence extraction."""
import argparse,os,time,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from contract import *
from extraction_contract import extraction_dictionary,extraction_model,feature_chunks
import math

def obj(props):return dict(type='object',properties=props,required=list(props),additionalProperties=False)
def arr(items,**kw):return dict(type='array',items=items,**kw)
def string(**kw):return dict(type='string',**kw)
def schema_gen(labels):return obj(dict(explanation=string(minLength=1),ranked_diagnoses=arr(string(enum=labels),minItems=3,maxItems=3)))
def schema_extract(codes):return obj(dict(claims=arr(obj(dict(
    quote=string(minLength=1),feature_id=dict(anyOf=[string(enum=codes),dict(type='null')]),
    value=dict(anyOf=[string(),dict(type='null')]),polarity=string(enum=['affirmed','negated','uncertain']),
    interpretation=string(),relation=string(enum=['supports','opposes','describes','uncertain']),
    diagnosis=dict(anyOf=[string(),dict(type='null')]))))))
GEN_SYSTEM='Based on the supplied patient information, first provide a sufficiently developed natural-language analysis in English of the clinical presentation and its interpretation. Explain how the case information informs your diagnostic assessment. Then conclude with exactly three distinct diagnoses from the allowed labels, ranked most likely first. Provide substantive explanation rather than an overly brief or repetitive answer. No fixed number of reasoning steps or pieces of evidence is required. Unlisted findings are unknown, not negative. Return only JSON, writing the explanation field first and the ranked_diagnoses field second. Do not output a separate thinking channel.'
EXT_SYSTEM="""Extract only patient-specific assertions explicitly present in the explanation and belonging to the supplied feature slice. Never force a concept into a similar feature. Return an empty claims list when none apply; do not emit out-of-slice or unmappable assertions in this pass. Copy exact contiguous quotes. Each claim maps ONE feature and ONE value; use multiple claims for multiple values. Binary legal values are strings true/false. Polarity asserts or denies feature=value; negative meaning encoded in a value must not be negated again. For example a no-travel value is affirmed by no travel. Do not infer patient facts from diagnoses or general medical rules. Do not infer units, numeric severity, time, subject, locations or missing details. For an exact numeric value, surface_value must be the verbatim numeric substring alone (for example 9 in 9/10). Numeric values may be exact ONLY if explicitly stated for that attribute in the quote. Qualitative descriptions such as severe, mild, rapid or swollen have value=null and value_resolution=qualitative: preserve the description in surface_value without translating to a number or interval. Non-numeric categorical paraphrases may map only when meaning is unambiguous in the dictionary. Unexplained abbreviations must remain unresolved. Use value_resolution=unresolved and value=null otherwise. Retain scope, subject and uncertainty in a brief interpretation, no internal discussion or rationale. relation/diagnosis describe the author's explicit relationship, not your inferred justification; use describes and null when not explicit. surface_value is copied wording, never a code such as true unless that word appears in the quote. exact means a resolved code (including binary present/absent), not necessarily a number. For numeric type N, an explicit number is allowed even when values is empty. qualitative/unresolved MUST use null; never populate a speculative code. No thinking channel. Return only the requested JSON."""

def verify_data(folder):
    m=read(folder/'manifest.json')
    for n,h in m['artifacts'].items():
        if sha(folder/n)!=h:raise ValueError('Shared data drift '+n)
    return m
def prepare(a):
    if a.responses<1 or a.dictionary_chunk<1 or a.max_tokens<1:raise ValueError('Counts must be positive')
    if a.num_shards<1:raise ValueError('num-shards must be positive')
    if a.out.exists():raise ValueError('Use new run directory')
    verify_data(a.data)
    binding=read(a.binding)
    if binding['model']!=a.model:raise ValueError('Binding/model mismatch')
    patients=[r for r in rows(a.data/'patients_public.jsonl') if r['split']==a.split]
    if not patients:raise ValueError('No patients in requested split')
    if a.patient_limit:
        patients=sorted(patients,key=lambda r:digest(r['patient_id']))[:a.patient_limit]
    contract=read(a.data/'evidence_dictionary.json');labels=read(a.data/'labels.json');tasks=[]
    if a.kind=='generate':
        for pat in patients:
            for repeat in range(a.responses):
                tasks.append(dict(id=digest([VERSION,pat.get('input_view_id',pat['patient_id']),a.model,repeat]),patient_id=pat['patient_id'],input_view_id=pat.get('input_view_id'),repeat=repeat,
                    schema=schema_gen(labels),messages=[dict(role='system',content=GEN_SYSTEM),dict(role='user',content=json.dumps(dict(allowed_diagnoses=labels,patient=pat['input_text'],missing_policy=pat['missing_policy'])))]))
    else:
        if not a.generation:raise ValueError('--generation required for extraction')
        gm=read(a.generation/'manifest.json')
        if gm['kind']!='generate' or gm['data_sha256']!=sha(a.data/'manifest.json'):raise ValueError('Generation/data mismatch')
        generated=latest_results(a.generation)
        chunks=feature_chunks(contract,a.dictionary_chunk)
        ids={r['patient_id'] for r in patients}
        for r in generated.values():
            if r['patient_id'] not in ids or r['status']!='VALID':continue
            for i,chunk in enumerate(chunks):
                tasks.append(dict(id=digest([r['id'],a.model,i]),patient_id=r['patient_id'],explanation_id=r['id'],chunk=i,
                    explanation=r['parsed']['explanation'],feature_dictionary=extraction_dictionary(contract,chunk),schema=extraction_model(extraction_dictionary(contract,chunk)).model_json_schema(),messages=[dict(role='system',content=EXT_SYSTEM),dict(role='user',content=json.dumps(dict(dictionary=extraction_dictionary(contract,chunk),ranked_diagnoses=r['parsed']['ranked_diagnoses'],explanation=r['parsed']['explanation'])))]))
    if not tasks:raise ValueError('No eligible requests')
    run_id=digest(dict(version=VERSION,binding=binding['fingerprint'],data=sha(a.data/'manifest.json'),tasks=tasks,max_tokens=a.max_tokens,response_mode=a.response_mode,kind=a.kind,seed=261004,temperature=.7 if a.kind=='generate' else 0.,top_p=.8 if a.kind=='generate' else 1.,thinking=False,code={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')}))
    for t in tasks:t['id']=digest([run_id,t['id']]);t['run_id']=run_id
    a.out.mkdir(parents=True);write(a.out/'requests.jsonl',tasks)
    dump(a.out/'manifest.json',dict(version=VERSION,run_id=run_id,binding=binding,kind=a.kind,split=a.split,data_path=str(a.data.resolve()),data_sha256=sha(a.data/'manifest.json'),
        generation_path=str(a.generation.resolve()) if a.generation else None,
        generation_results_sha256={f.name:sha(f) for f in result_files(a.generation)} if a.generation else None,
        num_shards=a.num_shards,
        endpoint=a.endpoint,model=a.model,temperature=.7 if a.kind=='generate' else 0.,top_p=.8 if a.kind=='generate' else 1.,
        max_tokens=a.max_tokens,seed=261004,enable_thinking=False,response_mode=a.response_mode,requests_sha256=sha(a.out/'requests.jsonl'),
        code={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')}))
    print('Prepared requests:',len(tasks),'kind:',a.kind,flush=True)

def request_json(url,body=None):
    headers={'Content-Type':'application/json'}
    if os.environ.get('OPENAI_API_KEY'):headers['Authorization']='Bearer '+os.environ['OPENAI_API_KEY']
    req=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,headers=headers)
    with urllib.request.urlopen(req,timeout=300) as response:return json.load(response)

def result_files(folder):
    return sorted(Path(folder).glob('results*.jsonl'))
def latest_results(folder):
    result={}
    for f in result_files(folder):
        for r in rows(f):
            if r['id'] not in result or r.get('attempt_time',0)>=result[r['id']].get('attempt_time',0):result[r['id']]=r
    return result
def validate(parsed,task,kind):
    from schemas import Explanation
    parsed=(Explanation if kind=='generate' else extraction_model(task['feature_dictionary'])).model_validate(parsed).model_dump()
    if not isinstance(parsed,dict):raise ValueError('Expected object')
    if kind=='generate':
        if set(parsed)!=set(['ranked_diagnoses','explanation']):raise ValueError('Output fields differ')
        ds=parsed['ranked_diagnoses'];legal=task['schema']['properties']['ranked_diagnoses']['items']['enum']
        if not isinstance(ds,list) or len(ds)!=3 or any(not isinstance(d,str) or d not in legal for d in ds) or len(set(ds))!=3:raise ValueError('Invalid ranked diagnoses')
        if not isinstance(parsed['explanation'],str) or not parsed['explanation'].strip():raise ValueError('Empty explanation')
    else:
        if set(parsed)!=set(['claims']) or not isinstance(parsed['claims'],list):raise ValueError('Invalid extraction')
        for c in parsed['claims']:
            if c['quote'] not in task['explanation']:raise ValueError('Quote not present verbatim')
            if c['surface_value'] and c['surface_value'] not in c['quote']:raise ValueError('surface_value must be copied from quote')
            if c['value_resolution']!='exact' and c['value'] is not None:raise ValueError('Non-exact claim must have null value')
            if c['value_resolution']=='exact' and c['value'] is None:raise ValueError('Exact claim requires a value')
            meta=task['feature_dictionary'][c['feature_id']]
            if meta['numeric_values'] and c['value'] is not None:
                # surface_value is the exact numeric substring, e.g. 9 in 9/10.
                try: explicit=float(c['surface_value'])
                except ValueError:raise ValueError('Numeric surface_value must be an explicit number')
                if not math.isfinite(explicit) or float(c['value'])!=explicit:raise ValueError('Numeric value lacks explicit surface evidence')
    return parsed

def audit_claims(payload,task):
    if not isinstance(payload,dict) or set(payload)!= {'claims'} or not isinstance(payload['claims'],list):
        raise ValueError('Invalid extraction envelope')
    accepted=[];rejected=[]
    for index,c in enumerate(payload['claims']):
        try:accepted.extend(validate({'claims':[c]},task,'extract')['claims'])
        except (ValueError,TypeError,KeyError) as e:
            rejected.append(dict(index=index,original_claim=c,error=str(e)))
    return dict(parsed={'claims':accepted},initial_parsed=payload,rejected_claims=rejected,
                status='PARTIAL' if rejected else 'VALID')

def run(a):
    m=read(a.run/'manifest.json');n=m.get('num_shards',1)
    if not 0<=a.shard_index<n:raise ValueError('Shard index outside frozen num_shards')
    lock=a.run/f'worker_{a.shard_index:04d}.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps(dict(pid=os.getpid(),host=__import__('socket').gethostname())).encode());os.close(fd)
    try:_run(a)
    finally:lock.unlink()

def stable_service(record):
    # vLLM model-card creation times and permission IDs are response metadata.
    # Neither identifies the lifetime of the serving process.
    return {k:v for k,v in record.items() if k not in ('created','permission')}

def check_execution_code(frozen,current):
    if current==frozen:return 'exact'
    previous='c1df6d22183f35a46f9e86729ec981c06d4c3e2c30a2ae4a2ff5bc4942b10774'
    expected=dict(frozen)
    if expected.get('api_pipeline.py')==previous:
        expected['api_pipeline.py']=current.get('api_pipeline.py')
        if current==expected:return 'service_identity_hotfix_20261004'
    raise ValueError('Code drift: not the supported service-identity-only patch')

def eligible_for_attempt(task_id,done,attempts,retry_failed):
    return task_id not in done or (retry_failed and done[task_id]['status']!='VALID' and attempts[task_id]<2)

def _run(a):
    invocation_start=time.time()
    m=read(a.run/'manifest.json');tasks=list(rows(a.run/'requests.jsonl'))
    if sha(a.run/'requests.jsonl')!=m['requests_sha256']:raise ValueError('Requests drift')
    execution_code={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')}
    compatibility=check_execution_code(m['code'],execution_code)
    local_binding=read(a.binding)
    if local_binding['fingerprint']!=m['binding']['fingerprint'] or local_binding['model']!=m['model']:raise ValueError('Worker model fingerprint differs')
    for n,h in local_binding['files'].items():
        if sha(Path(local_binding['model_dir'])/n)!=h:raise ValueError('Local model files changed: '+n)
    models=request_json(m['endpoint'].rsplit('/chat/completions',1)[0]+'/models')
    if m['model'] not in {r['id'] for r in models['data']}:raise ValueError('Served model name not found')
    n=m.get('num_shards',1);tasks=[t for t in tasks if int(t['id'],16)%n==a.shard_index]
    suffix='' if n==1 else f'_shard_{a.shard_index:04d}'
    result_path=a.run/f'results{suffix}.jsonl';report_path=a.run/f'report{suffix}.json'
    service=next(r for r in models['data'] if r['id']==m['model'])
    before=stable_service(local_binding['service_record']);after=stable_service(service)
    dump(a.run/f'service_identity_check{suffix}.json',dict(expected=before,observed=after,
        binding_service_record=local_binding['service_record'],current_service_record=service,
        ignored_response_fields=['created','permission'],matched=before==after,
        code_compatibility=compatibility,execution_code=execution_code,
        note='Model-card stability and local file hashes do not attest loaded weights or detect all server restarts.'))
    if before!=after:
        changed=[k for k in sorted(set(before)|set(after)) if before.get(k)!=after.get(k)]
        raise ValueError('Stable service fields changed: '+', '.join(changed)+'. Inspect service_identity_check'+suffix+'.json')
    dump(a.run/f'worker_binding{suffix}.json',local_binding)
    history=list(rows(result_path)) if result_path.exists() else []
    done={r['id']:r for r in history}
    attempts=Counter(r['id'] for r in history)
    todo=[t for t in tasks if eligible_for_attempt(t['id'],done,attempts,a.retry_failed)]
    if a.workers<1:raise ValueError('workers must be positive')
    if todo:
        # Count rendered prompts before generation; never silently truncate patients.
        root=m['endpoint'].rsplit('/v1/chat/completions',1)[0]
        maxlen=next(r for r in models['data'] if r['id']==m['model']).get('max_model_len')
        if not maxlen:raise ValueError('Server did not report max_model_len')
        probe=[dict(role='user',content='Return JSON with ok true.')]
        switches=[request_json(root+'/tokenize',dict(model=m['model'],messages=probe,add_generation_prompt=True,chat_template_kwargs={'enable_thinking':v})) for v in [False,True]]
        if not switches[0].get('tokens') or switches[0].get('tokens')==switches[1].get('tokens'):raise ValueError('Thinking template switch not observable; inspect server first')
        def count(t):
            r=request_json(root+'/tokenize',dict(model=m['model'],messages=t['messages'],add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False}))
            if r['count']+m['max_tokens']>maxlen:raise ValueError('Request exceeds context: '+t['id'])
            return r['count']
        # Identical repeat prompts need only one token check.
        unique={digest(t['messages']):t for t in todo}
        counts=[]
        with ThreadPoolExecutor(max_workers=min(a.workers,8)) as pool:
            for i,value in enumerate(pool.map(count,unique.values()),1):
                counts.append(value)
                if i%500==0:print(f'Token checked {i}/{len(unique)} unique prompts',flush=True)
        dump(a.run/f'preflight{suffix}.json',dict(max_prompt_tokens=max(counts),max_model_len=maxlen,max_tokens=m['max_tokens'],models=models,
            thinking_switch_observed=True,note='Server template switch check; not proof of internal computation or model file identity.'))
        print('All prompts fit context; starting requests',flush=True)
    def call(t):
        body=dict(model=m['model'],messages=t['messages'],temperature=m['temperature'],top_p=m['top_p'],max_tokens=m['max_tokens'],seed=t.get('sampling_seed',(m['seed']+int(t['id'][:8],16))%2147483647),chat_template_kwargs={'enable_thinking':False})
        body['response_format']=dict(type='json_schema',json_schema=dict(name='shared_evidence',strict=True,schema=t['schema'])) if m['response_mode']=='schema' else dict(type='json_object')
        result={k:t[k] for k in ['id','run_id','patient_id','input_view_id','explanation_id','chunk','repeat'] if k in t};result['attempt_time']=time.time()
        try:
            raw=request_json(m['endpoint'],body);result['response']=raw;c=raw['choices'][0];msg=c['message']
            if c['finish_reason']!='stop':raise ValueError('Unfinished response: '+str(c['finish_reason']))
            if msg.get('reasoning_content') or msg.get('reasoning') or '<think>' in (msg.get('content') or ''):raise ValueError('Unexpected thinking output')
            payload=json.loads(msg['content'])
            if m['kind']=='extract':
                audit=audit_claims(payload,t)
                result.update(audit)
            else:
                result['parsed']=validate(payload,t,m['kind']);result['status']='VALID'
        except Exception as e:result.update(status='FAILED',error=str(e))
        result['finished_at']=time.time();result['latency_seconds']=result['finished_at']-result['attempt_time']
        return result
    generation_start=time.time();pending_count=len(todo)
    with result_path.open('a',encoding='utf-8',newline='\n') as f,ThreadPoolExecutor(max_workers=a.workers) as pool:
        if todo:
            first=call(todo.pop(0));f.write(json.dumps(first,ensure_ascii=False)+'\n');f.flush();done[first['id']]=first
            if first['status'] not in ['VALID','PARTIAL']:
                dump(report_path,dict(planned=len(tasks),completed=len(done),first_request_failed=first))
                raise ValueError('First pending request failed; stopped before bulk requests. Inspect report.json.')
        futures=[pool.submit(call,t) for t in todo]
        for i,fut in enumerate(as_completed(futures),1):
            r=fut.result();f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();done[r['id']]=r
            if i%100==0 or i==len(todo):
                elapsed=time.time()-generation_start;rate=(i+1)/max(elapsed,1e-9);remaining=len(todo)-i
                print(f'Completed {i+1}/{pending_count}; elapsed {elapsed/60:.1f} min; rate {rate*3600:.0f} answers/hour; ETA {remaining/rate/3600:.2f} h',flush=True)
    timing=dict(started_at=invocation_start,finished_at=time.time(),generation_seconds=time.time()-generation_start,preflight_seconds=generation_start-invocation_start,attempts=pending_count,workers=a.workers)
    with (a.run/f'timing{suffix}.jsonl').open('a',encoding='utf-8') as tf:tf.write(json.dumps(timing)+'\n')
    dump(report_path,dict(timing=timing,max_attempts_per_request=2,planned=len(tasks),completed=len(done),shard_index=a.shard_index,num_shards=n,statuses=dict(Counter(r['status'] for r in done.values())),
        note='Schema validity is not extraction recall or semantic correctness. Failed first attempts retained; no automatic content regeneration.'))

def align_run(a):
    m=read(a.run/'manifest.json')
    if m['kind']!='extract':raise ValueError('Expected extraction run')
    if sha(a.run/'requests.jsonl')!=m['requests_sha256']:raise ValueError('Request drift')
    data=Path(m['data_path']);verify_data(data)
    if sha(data/'manifest.json')!=m['data_sha256']:raise ValueError('Data manifest drift')
    contract=read(data/'evidence_dictionary.json');patients={r['patient_id']:r for r in rows(data/'patients_public.jsonl')}
    tasks=list(rows(a.run/'requests.jsonl'));results=latest_results(a.run);groups={}
    for t in tasks:
        g=groups.setdefault(t['explanation_id'],dict(explanation_id=t['explanation_id'],patient_id=t['patient_id'],expected_chunks=0,valid_chunks=0,claims=[],mapping_errors=[]))
        g['expected_chunks']+=1;r=results.get(t['id'])
        if not r or r['status'] not in ['VALID','PARTIAL']:continue
        g.setdefault('rejected_claims',[]).extend(r.get('rejected_claims',[]))
        valid=r['status']=='VALID'
        for c in r['parsed']['claims']:
            try:
                mapped=align(c,patients[t['patient_id']],contract)
                mapped['initial_extracted_claim']=dict(c)
                mapped['feature_reference_status']='IDENTIFIED'
                if c['value_resolution']!='exact':mapped['status']='QUALITATIVE_UNASSESSED' if c['value_resolution']=='qualitative' else 'UNCERTAIN'
                quote=c['quote'];text=t['explanation'];starts=[];start=0
                while True:
                    pos=text.find(quote,start)
                    if pos<0:break
                    starts.append([pos,pos+len(quote)]);start=pos+1
                mapped['quote_offsets']=starts;mapped['span_ambiguous']=len(starts)!=1
                if mapped not in g['claims']:g['claims'].append(mapped)
            except ValueError as e:valid=False;g['mapping_errors'].append(str(e))
        if valid:g['valid_chunks']+=1
    for g in groups.values():
        g['status']='MAPPING_COMPLETE_UNVALIDATED' if g['valid_chunks']==g['expected_chunks'] else 'PARTIAL_DO_NOT_SCORE'
        g['overlapping_claim_pairs']=[{'left':i,'right':j,'reason':'OVERLAPPING_TEXT_DIFFERENT_FEATURE_REVIEW'} for i,c in enumerate(g['claims']) for j,d in enumerate(g['claims']) if j>i and c['feature_id']!=d['feature_id'] and any(max(x[0],y[0])<min(x[1],y[1]) for x in c['quote_offsets'] for y in d['quote_offsets'])]
        g['coverage_unit']='whole_explanation_union_not_top1_support'
        g['referenced_feature_ids']=sorted({c['feature_id'] for c in g['claims']})
        g['qualitative_feature_ids']=sorted({c['feature_id'] for c in g['claims'] if c['value_resolution']=='qualitative'})
        g['unmapped_claim_audit']='NOT_PERFORMED_SLICE_ONLY_EXTRACTION' 
        g['cited_fact_ids']=sorted({fid for c in g['claims'] if c['status']=='MATCHED' and c['polarity']!='uncertain' for fid in c['matched_fact_ids']})
        visible={f['fact_id']:f for f in patients[g['patient_id']]['facts']}
        g['symptom_antecedent_fact_ids']=[k for k in g['cited_fact_ids'] if visible[k].get('coverage_scope')=='symptoms_antecedents']
        g['demographic_fact_ids']=[k for k in g['cited_fact_ids'] if visible[k].get('coverage_scope')=='demographics']
    write(a.run/'aligned_evidence.jsonl',groups.values())
    dump(a.run/'alignment_report.json',dict(explanations=len(groups),statuses=dict(Counter(g['status'] for g in groups.values())),
        claim_statuses=dict(Counter(c['status'] for g in groups.values() for c in g['claims'])),
        empty_extractions=sum(not g['claims'] for g in groups.values()),
        note='No C/E labels or clinical correctness inferred. COSI subset must exclude partial mappings and report exclusions; semantic recall requires independent checks.'))
    print('Aligned explanations:',len(groups),flush=True)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--kind',choices=['generate','extract'],required=True);q.add_argument('--data',type=Path,required=True)
    q.add_argument('--split',choices=['train','semantic_audit','development','heldout'],default='semantic_audit');q.add_argument('--model',required=True)
    q.add_argument('--endpoint',default='http://127.0.0.1:8001/v1/chat/completions');q.add_argument('--out',type=Path,required=True)
    q.add_argument('--generation',type=Path);q.add_argument('--responses',type=int,default=2);q.add_argument('--dictionary-chunk',type=int,default=12)
    q.add_argument('--max-tokens',type=int,default=2048);q.add_argument('--response-mode',choices=['schema','json'],default='schema')
    q.add_argument('--num-shards',type=int,default=1);q.add_argument('--binding',type=Path,required=True);q.add_argument('--patient-limit',type=int,default=16)
    q=sub.add_parser('run');q.add_argument('--run',type=Path,required=True);q.add_argument('--workers',type=int,default=32);q.add_argument('--retry-failed',action='store_true');q.add_argument('--shard-index',type=int,default=0);q.add_argument('--binding',type=Path,required=True)
    q=sub.add_parser('align');q.add_argument('--run',type=Path,required=True)
    a=p.parse_args();{'prepare':prepare,'run':run,'align':align_run}[a.command](a)
if __name__=='__main__':main()
