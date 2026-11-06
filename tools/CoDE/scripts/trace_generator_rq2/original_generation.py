"""TRACE shared generation pilot. No extraction, evaluation outcomes or model launch."""
import argparse, collections, concurrent.futures as cf, hashlib, importlib.metadata
import json, os, threading, time, urllib.request, urllib.error, tarfile
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

VERSION='trace_rollouts_v1'
LABELS=['appendicitis','cholecystitis','diverticulitis','pancreatitis']
SYSTEM=('Based on the supplied patient information, first provide a sufficiently developed '
 'natural-language analysis in English of the clinical presentation and its interpretation. '
 'Explain how the case information informs your diagnostic assessment. Then conclude with '
 'exactly three distinct diagnoses from the allowed labels, ranked most likely first. '
 'Provide substantive explanation rather than an overly brief or repetitive answer. '
 'No fixed number of reasoning steps or pieces of evidence is required. '
 'Unlisted findings are unknown, not negative. Return only JSON, writing the explanation '
 'field first and the ranked_diagnoses field second. Do not output a separate thinking channel.')

class Answer(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    explanation:str=Field(min_length=1)
    ranked_diagnoses:list[Literal['appendicitis','cholecystitis','diverticulitis','pancreatitis']]=Field(min_length=3,max_length=3)
    @field_validator('explanation')
    @classmethod
    def nonempty(cls,v):
        if not v.strip():raise ValueError('Blank explanation')
        if '<think>' in v or '</think>' in v:raise ValueError('Thinking tags in explanation')
        return v
    @field_validator('ranked_diagnoses')
    @classmethod
    def distinct(cls,v):
        if len(set(v))!=3:raise ValueError('Diagnoses must be distinct')
        return v

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        for n,l in enumerate(f,1):
            if l.strip():
                try:yield json.loads(l)
                except Exception as e:raise ValueError(f'Invalid JSONL at {p}:{n}; preserve file and inspect before resuming') from e
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def codes():return {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
def api(url,body=None,timeout=240):
    headers={'Content-Type':'application/json'}
    if os.environ.get('OPENAI_API_KEY'):headers['Authorization']='Bearer '+os.environ['OPENAI_API_KEY']
    req=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,headers=headers)
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
def service(endpoint,model):
    if not endpoint.endswith('/v1/chat/completions'):raise ValueError('Endpoint must end /v1/chat/completions')
    data=api(endpoint.removesuffix('/chat/completions')+'/models',timeout=15)
    cards=[c for c in data['data'] if c['id']==model]
    if len(cards)!=1:raise ValueError('Requested model absent or ambiguous in /models')
    return cards[0]
def messages(patient):
    # No identifiers, split, reference answer, provenance or hidden records.
    return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps({
        'allowed_diagnoses':LABELS,'patient_facts':[f['display_text'] for f in patient['facts']]},ensure_ascii=False)}]
def freeze_input(folder):
    m=read(folder/'manifest.json')
    for n,h in m['output_sha256'].items():
        p=(folder/n).resolve()
        if not p.is_relative_to(folder.resolve()) or sha(p)!=h:raise ValueError('Input drift '+n)
    if read(folder/'diagnosis_vocabulary.json')['labels']!=LABELS:raise ValueError('Vocabulary changed')
    patients=list(rows(folder/'patients_model_input_PRIVATE.jsonl'));pi={r['patient_id']:r for r in patients}
    splits=list(rows(folder/'split_without_outcomes_PRIVATE.jsonl'));si={r['patient_id']:r for r in splits}
    if len(patients)!=len(pi) or len(pi)!=2400 or set(pi)!=set(si):raise ValueError('Admission/split mismatch')
    d=read(folder/'evidence_dictionary.json')['features'];ids=set();subjects={}
    for pid,r in pi.items():
        if set(r)!={'patient_id','view_id','context','facts'} or r['context'] or not r['facts']:raise ValueError('Unexpected patient fields')
        s=si[pid];sg=s['subject_group']
        if subjects.setdefault(sg,s['partition'])!=s['partition']:raise ValueError('Subject crosses partitions')
        if r['view_id']!=m['input_view_id']:raise ValueError('Input view mismatch')
        for f in r['facts']:
            if set(f)!={'fact_id','feature_id','state','display_text','group_id'}:raise ValueError('Unexpected fact fields')
            if f['fact_id'] in ids:raise ValueError('Duplicate fact ID')
            ids.add(f['fact_id']);dd=d[f['feature_id']]
            if f['state'] not in dd['values'] or f['group_id']!=dd['group_id'] or not f['display_text'].strip():raise ValueError('Invalid fact')
    tasks=list(rows(folder/'generation_tasks_PRIVATE.jsonl'));ti={r['task_id']:r for r in tasks};counts=collections.defaultdict(list)
    if len(tasks)!=9600 or len(ti)!=9600:raise ValueError('Full task census mismatch')
    for t in tasks:
        pid=t['patient_id'];counts[pid].append(t['rollout_index'])
        if t['input_digest']!=digest(pi[pid]) or t['input_view_id']!=m['input_view_id']:raise ValueError('Task digest mismatch')
        if t['task_id']!='TRACE_'+digest([m['input_view_id'],pid,t['rollout_index']])[:24]:raise ValueError('Task identity mismatch')
    if set(counts)!=set(pi) or any(sorted(x)!=[0,1,2,3] for x in counts.values()):raise ValueError('Rollout count mismatch')
    pilot=list(rows(folder/'pilot_tasks_PRIVATE.jsonl'));pp={t['patient_id'] for t in pilot}
    expected=set(sorted((p for p in pi if si[p]['partition']=='development'),key=lambda p:digest(['pilot',p]))[:20])
    if len(pilot)!=80 or len({t['task_id'] for t in pilot})!=80 or pp!=expected or any(t!=ti[t['task_id']] for t in pilot):raise ValueError('Pilot identity mismatch')
    return m,pi,pilot,tasks

def prepare(a):
    if a.out.exists():raise ValueError('Use fresh run directory; resume via run command')
    m,pi,pilot,tasks=freeze_input(a.input)
    contract=read(a.input/'delivery_contract.json');want=contract['generation_requested']
    if contract['generation_model']!=a.model or want!={'temperature':0.7,'top_p':0.8,'max_output_tokens':2048,'thinking':False,'diagnoses_count':3,'diagnoses_distinct':True,'explanation_language':'English'}:raise ValueError('Contract differs; review before production')
    # Full production intentionally unavailable until this pilot is reviewed.
    a.out.mkdir(parents=True)
    with (a.out/'requests_PRIVATE.jsonl').open('w',encoding='utf-8') as f:
        for t in pilot:
            r={**t,'messages':messages(pi[t['patient_id']])}
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    manifest={'protocol':VERSION,'phase':'pilot','input':str(a.input.resolve()),'input_manifest_sha256':sha(a.input/'manifest.json'),
     'input_view_id':m['input_view_id'],'model':a.model,'endpoint':a.endpoint,'model_dir':str(a.model_dir.resolve()),
     'requests_sha256':sha(a.out/'requests_PRIVATE.jsonl'),'system':SYSTEM,'schema':Answer.model_json_schema(),
     'temperature':0.7,'top_p':0.8,'max_tokens':2048,'enable_thinking':False,'response_mode':'json_schema',
     'base_seed':261013,'max_attempts_per_invocation':3,'planned':80,'admissions':20,'code':codes(),
     'prompt_origin':'Verbatim GEN_SYSTEM from existing tools/lantern/forward_review_v1/api_pipeline.py; patient rendering adapted to frozen TRACE display texts.'}
    manifest['run_id']=digest(manifest);write(a.out/'manifest.json',manifest)
    print('Prepared 20 development admissions / 80 tasks. No API requests yet.',flush=True)

def token_preflight(a,m,tasks):
    from transformers import AutoTokenizer
    model=Path(m['model_dir'])
    if not model.is_dir():raise FileNotFoundError('Client model directory missing: '+str(model))
    card=service(m['endpoint'],m['model']);limit=card.get('max_model_len')
    if not isinstance(limit,int) or limit<1:raise ValueError('Server max_model_len unavailable')
    print('Service model root (metadata only):',card.get('root'),'| Local checkpoint:',model,flush=True)
    # Paths may differ across service/client containers. No string equality requirement.
    candidates=sorted(p for p in model.rglob('*') if p.is_file() and '.cache' not in p.parts and p.suffix in {'.json','.safetensors','.bin','.model','.jinja','.txt'})
    if not any(p.suffix in {'.bin','.safetensors'} for p in candidates):raise ValueError('No checkpoint weights')
    hashes={}
    for p in candidates:
        print('Fingerprint:',p.name,flush=True);hashes[str(p.relative_to(model))]=sha(p)
    fingerprint=digest(hashes);old=a.run/'binding.json'
    if old.exists() and read(old)['fingerprint']!=fingerprint:raise ValueError('Local model changed since run binding')
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    root=m['endpoint'].removesuffix('/v1/chat/completions')
    def ids(msgs,thinking):
        local=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,enable_thinking=thinking)
        server=api(root+'/tokenize',{'model':m['model'],'messages':msgs,'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':thinking}},timeout=60)
        if local!=server.get('tokens') or server.get('count')!=len(local):raise ValueError('Server/local template token mismatch')
        return local
    probe=[{'role':'user','content':'Return a JSON object with ok set to true.'}]
    no,yes=ids(probe,False),ids(probe,True)
    if no==yes:raise ValueError('Thinking switch not observable')
    counts={};seen={}
    for t in tasks:
        key=digest(t['messages'])
        if key not in seen:
            n=len(ids(t['messages'],False));seen[key]=n
            print('Token checked',len(seen),'/ 20; tokens',n,flush=True)
            if n+m['max_tokens']>limit:raise ValueError('Context overflow; no truncation: '+t['task_id'])
        counts[t['task_id']]=seen[key]
    packages={}
    for name in ['torch','transformers','vllm','pydantic']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    binding={'model':m['model'],'model_dir':str(model),'files':hashes,'fingerprint':fingerprint,'service':card,
      'packages':packages,'attestation':'Local checkpoint fingerprint plus server/template agreement; API does not attest weights, dtype or quantization.'}
    write(old,binding)
    report={'status':'PASS','template_tokens_match':True,'thinking_switch_verified':True,'counts':counts,'max_prompt_tokens':max(counts.values()),'max_model_len':limit,'max_output_tokens':m['max_tokens'],'silent_truncation':False}
    write(a.run/'preflight.json',report);return counts,binding

def parse_response(raw,expected_tokens):
    choice=raw['choices'][0];msg=choice['message']
    if choice['finish_reason']!='stop':raise ValueError('Non-stop finish_reason='+str(choice['finish_reason']))
    if msg.get('reasoning_content') or msg.get('reasoning'):raise ValueError('Unexpected reasoning channel')
    obj=Answer.model_validate_json(msg['content']).model_dump()
    if raw.get('usage',{}).get('prompt_tokens')!=expected_tokens:raise ValueError('Prompt token count mismatch')
    return obj

def run(a):
    import fcntl
    if not 1<=a.workers<=128:raise ValueError('Workers must be 1..128')
    with (a.run/'writer.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another writer owns this run; do not start twice')
        _run(a)

def _run(a):
    m=read(a.run/'manifest.json')
    if codes()!=m['code'] or sha(a.run/'requests_PRIVATE.jsonl')!=m['requests_sha256']:raise ValueError('Frozen code/requests changed')
    if sha(Path(m['input'])/'manifest.json')!=m['input_manifest_sha256']:raise ValueError('Source manifest drift')
    freeze_input(Path(m['input']))
    tasks=list(rows(a.run/'requests_PRIVATE.jsonl'));task_ids={t['task_id'] for t in tasks};output=a.run/'results_PRIVATE.jsonl'
    latest={r['task_id']:r for r in rows(output)} if output.exists() else {}
    if not set(latest)<=task_ids:raise ValueError('Unexpected results in run')
    pending=[t for t in tasks if t['task_id'] not in latest or (a.retry_failed and latest[t['task_id']]['status']=='failed')]
    if not pending:print('No pending tasks; use export for review package.',flush=True);return
    counts,binding=token_preflight(a,m,tasks)
    mutex=threading.Lock();start=time.monotonic();invocation=str(time.time_ns())
    with (a.run/'attempts_PRIVATE.jsonl').open('a',encoding='utf-8',buffering=1) as af,output.open('a',encoding='utf-8',buffering=1) as of:
        def call(t):
            attempts=[];parsed=None
            for i in range(m['max_attempts_per_invocation']):
                seed=int(digest([m['base_seed'],t['task_id'],invocation if a.retry_failed else 'initial',i])[:8],16)%2147483647
                payload={'model':m['model'],'messages':t['messages'],'temperature':m['temperature'],'top_p':m['top_p'],'max_tokens':m['max_tokens'],'seed':seed,
                 'chat_template_kwargs':{'enable_thinking':False},'response_format':{'type':'json_schema','json_schema':{'name':'trace_diagnostic_explanation','strict':True,'schema':m['schema']}}}
                at={'task_id':t['task_id'],'invocation':invocation,'attempt_index':i,'started_at':time.time(),'request_parameters':payload,'raw_response':None,'status':'failed'}
                fatal=False
                try:
                    at['raw_response']=api(m['endpoint'],payload)
                    parsed=parse_response(at['raw_response'],counts[t['task_id']]);at['status']='complete'
                except Exception as e:
                    at['error']=str(e)
                    if isinstance(e,urllib.error.HTTPError):
                        at['http_status']=e.code;at['http_body']=e.read().decode('utf-8',errors='replace')[:20000]
                        fatal=e.code in {400,401,403,404,422}
                at['seconds']=time.time()-at['started_at'];at['configuration_error']=fatal
                with mutex:af.write(json.dumps(at,ensure_ascii=False)+'\n')
                attempts.append(at)
                if parsed is not None or fatal:break
                if i+1<m['max_attempts_per_invocation']:time.sleep(min(2**i,4))
            result={**t,'run_id':m['run_id'],'model_identity':{'binding_file':'binding.json','fingerprint':binding['fingerprint'],'served_model':m['model']},
             'request_parameters':attempts[-1]['request_parameters'],'attempts':attempts,'selected_attempt':len(attempts)-1 if parsed else None,
             'raw_response':attempts[-1]['raw_response'],'status':'complete' if parsed else 'failed','parsed_explanation':parsed['explanation'] if parsed else None,
             'parsed_diagnoses':parsed['ranked_diagnoses'] if parsed else None,'prompt_token_count':counts[t['task_id']],'invocation':invocation}
            return result
        def save(r):
            of.write(json.dumps(r,ensure_ascii=False)+'\n');latest[r['task_id']]=r
        # First actual pilot task detects an unsupported service/schema before fan-out.
        first=call(pending[0]);save(first)
        if first['attempts'][-1]['configuration_error']:raise ValueError('Service/configuration error recorded; inspect first attempt before retrying')
        done_count=1
        print('Completed 1 /',len(pending),'first status:',first['status'],flush=True)
        with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(call,t) for t in pending[1:]}
            while futures:
                ready,futures=cf.wait(futures,timeout=20,return_when=cf.FIRST_COMPLETED)
                for f in ready:save(f.result());done_count+=1
                elapsed=time.monotonic()-start;rate=done_count/elapsed*3600
                print(json.dumps({'completed_this_run':done_count,'pending':len(pending)-done_count,'statuses':dict(collections.Counter(r['status'] for r in latest.values())),'seconds':round(elapsed,1),'answers_per_hour':round(rate,1)}),flush=True)
    write(a.run/f'timing_{invocation}.json',{'invocation':invocation,'workers':a.workers,'seconds':time.monotonic()-start,'completed_this_run':done_count})
    report(a.run)

def report(folder):
    m=read(folder/'manifest.json');tasks=list(rows(folder/'requests_PRIVATE.jsonl'));latest={r['task_id']:r for r in rows(folder/'results_PRIVATE.jsonl')} if (folder/'results_PRIVATE.jsonl').exists() else {}
    counts=collections.Counter(r['status'] for r in latest.values());valid=[r for r in latest.values() if r['status']=='complete'];attempts=list(rows(folder/'attempts_PRIVATE.jsonl')) if (folder/'attempts_PRIVATE.jsonl').exists() else []
    result={'planned':len(tasks),'recorded':len(latest),'missing':len(tasks)-len(latest),'statuses':dict(counts),'attempts_all_invocations':len(attempts),
     'valid_schema_fraction':len(valid)/len(tasks),'first_attempt_statuses_latest_invocation':dict(collections.Counter(r['attempts'][0]['status'] for r in latest.values())),
     'input_view_id':m['input_view_id'],'generation_finished':len(latest)==len(tasks),'ready_for_full_production':False,
     'note':'Pilot engineering outputs for content review. No reference answers loaded, no accuracy criterion, no extraction yet. Full production requires review of messages and outputs.'}
    write(folder/'generation_report.json',result);print(json.dumps(result,indent=2),flush=True)

def export(a):
    import fcntl
    target=a.run.parent/(a.run.name+'_review_PRIVATE.tar.gz')
    with (a.run/'writer.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Writer still running; export after it finishes')
        report(a.run)
        with tarfile.open(target,'w:gz') as tf:
            for p in sorted(a.run.iterdir()):
                if p.is_file() and p.suffix in {'.json','.jsonl'}:tf.add(p,arcname=p.name)
    print('Review package:',target,flush=True)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--input',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q.add_argument('--model-dir',type=Path,required=True);q.add_argument('--model',default='Qwen3-4B');q.add_argument('--endpoint',default='http://127.0.0.1:8001/v1/chat/completions')
    q=sub.add_parser('run');q.add_argument('--run',type=Path,required=True);q.add_argument('--workers',type=int,default=32);q.add_argument('--retry-failed',action='store_true')
    q=sub.add_parser('export');q.add_argument('--run',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)

if __name__=='__main__':main()
