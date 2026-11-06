import os
import argparse,collections,concurrent.futures as cf,hashlib,json,shutil,time,tarfile
from pathlib import Path
import base as b
import segment as s

def codes():return {p.name:b.sha(p) for p in Path(__file__).parent.glob('*.py')}
def unique(items,key):
    out={}
    for r in items:
        if r[key] in out:raise ValueError('DUPLICATE '+str(r[key]))
        out[r[key]]=r
    return out
def prepare(a):
    if a.root.exists():raise ValueError('Use a new output root')
    source=a.source
    if source is None:
        found=[p.parent for p in Path('outputs/trace_release_v1').glob('run_*/acceptance.json') if b.sha(p)=='bee7ee4580d9409a5ef13bd82cbb515fe9d098768451f08402532176d5be6cf8']
        if len(found)!=1:raise ValueError('Specify --source release directory')
        source=found[0]
    sm=b.read(source/'release_manifest.json')
    needed=['explanations_PRIVATE.jsonl','documents_PRIVATE.jsonl','frozen_input/patients_model_input_PRIVATE.jsonl']
    for name in needed:
        b.print('Checking',name,flush=True)
        if name not in sm['outputs'] or b.sha(source/name)!=sm['outputs'][name]:raise ValueError('SOURCE_HASH '+name)
    ledger=unique(b.rows(source/needed[0]),'task_id');docs=unique(b.rows(source/needed[1]),'task_id');pts=unique(b.rows(source/needed[2]),'patient_id')
    if len(ledger)!=9600 or len(pts)!=2400:raise ValueError('Unexpected fixed denominator')
    if set(docs)-set(ledger):raise ValueError('FOREIGN_DOCUMENT')
    a.root.mkdir(parents=True);(a.root/'nodes').mkdir()
    loads=[0]*8;handles=[];meta=[]
    try:
        for n in range(8):
            d=a.root/'nodes'/str(n);d.mkdir();handles.append((d/'tasks_PRIVATE.jsonl').open('w',encoding='utf-8'))
        for tid in sorted(ledger):
            row=dict(ledger[tid]);pid=row['patient_id']
            if pid not in pts:raise ValueError('PATIENT_MISSING')
            if tid not in docs:
                if row['generation_status']=='complete':raise ValueError('COMPLETE_GENERATION_DOCUMENT_MISSING')
                row.update(match_node=None,match_groups=0,match_input_status='GENERATION_UNAVAILABLE');meta.append(row);continue
            d=docs[tid]
            if row['generation_status']!='complete':raise ValueError('DOCUMENT_GENERATION_STATUS_MISMATCH '+json.dumps(dict(task_id=tid,status=row['generation_status'])))
            if d['patient_id']!=pid:raise ValueError('DOCUMENT_PATIENT_MISMATCH '+json.dumps(dict(task_id=tid,ledger_patient=pid,document_patient=d['patient_id'])))
            if d['input_view_id']!=row['input_view_id'] or d['input_view_id']!=pts[pid]['view_id']:raise ValueError('DOCUMENT_VIEW_MISMATCH '+tid)
            facts=pts[pid]['facts']
            if len({f['fact_id'] for f in facts})!=len(facts):raise ValueError('DUPLICATE_FACT')
            n=min(range(8),key=lambda i:(loads[i],i));ng=(len(facts)+7)//8
            row.update(match_node=n,match_groups=ng,match_input_status='READY');meta.append(row)
            for start in range(0,len(facts),8):
                t=dict(id=tid+f':g{start//8:04d}',task_id=tid,patient_id=pid,input_view_id=d['input_view_id'],explanation=d['text'],
                    facts=[{k:f[k] for k in ['fact_id','feature_id','state','display_text']} for f in facts[start:start+8]])
                handles[n].write(json.dumps(t,ensure_ascii=False)+'\n')
            loads[n]+=ng
    finally:
        for h in handles:h.close()
    b.dump(a.root/'ledger_PRIVATE.jsonl',meta)
    m=dict(source=str(source.resolve()),source_manifest_sha256=b.sha(source/'release_manifest.json'),nodes=8,planned_explanations=9600,
        groups=sum(loads),per_node=loads,code=codes(),source_files={x:sm['outputs'][x] for x in needed},
        files={f'nodes/{n}/tasks_PRIVATE.jsonl':b.sha(a.root/'nodes'/str(n)/'tasks_PRIVATE.jsonl') for n in range(8)},
        ledger_sha256=b.sha(a.root/'ledger_PRIVATE.jsonl'),protocol='v4 free quote max 2 attempts + failed group segment max 2 attempts',
        note='All partitions retained. No prior pilot cache reuse: uniform first-pass protocol. No labels or scoring outcomes sent to API.')
    b.write(a.root/'manifest.json',m);b.print(json.dumps(m,ensure_ascii=False,indent=2),flush=True)

def check(a):
    m=b.read(a.root/'manifest.json')
    if m['code']!=codes():raise ValueError('CODE_DRIFT')
    return m
def run(a):
    import fcntl
    m=check(a);d=a.root/'nodes'/str(a.node)
    with (d/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=d/'tasks_PRIVATE.jsonl'
        if b.sha(path)!=m['files'][f'nodes/{a.node}/tasks_PRIVATE.jsonl']:raise ValueError('TASK_DRIFT')
        tasks=unique(b.rows(path),'id');out=d/'results_PRIVATE.jsonl'
        old=unique(b.rows(out),'id') if out.exists() else {}
        if set(old)-set(tasks):raise ValueError('FOREIGN_RESULT')
        cfg=dict(endpoint=a.endpoint,model=a.model,workers=a.workers,free_max_tokens=3072,segment_max_tokens=2048,attempts_per_mode=2,thinking=False)
        if (d/'config.json').exists() and b.read(d/'config.json')!=cfg:raise ValueError('CONFIG_DRIFT')
        b.write(d/'config.json',cfg)
        cards=[x for x in b.api(a.endpoint.removesuffix('/chat/completions')+'/models')['data'] if x['id']==a.model]
        if len(cards)!=1 or not isinstance(cards[0].get('max_model_len'),int):raise ValueError('SERVICE_INVALID')
        b.write(d/'service.json',cards[0]);limit=cards[0]['max_model_len'];start=time.monotonic()
        def work(t):
            attempts=[]
            for mode in ['free','segment']:
                module=b if mode=='free' else s; budget=3072 if mode=='free' else 2048
                body=dict(facts=[dict(index=i,feature_id=f['feature_id'],state=f['state'],text=f['display_text']) for i,f in enumerate(t['facts'])])
                if mode=='free':body['explanation']=t['explanation']
                else:body['segments']=[dict(id=x['id'],text=x['text']) for x in s.segments(t['explanation'])]
                msgs=[dict(role='system',content=module.PROMPT),dict(role='user',content=json.dumps(body,ensure_ascii=False))]
                white=False
                for attempt in range(2):
                    tick=time.monotonic();at=dict(mode=mode,attempt=attempt+1);retry=False
                    try:
                        count=b.api(a.endpoint.removesuffix('/v1/chat/completions')+'/tokenize',dict(model=a.model,messages=msgs,add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False}),timeout=30)['count']
                        at['prompt_tokens_preflight']=count
                        if count+budget>limit:raise ValueError('CONTEXT_OVERFLOW')
                        req=dict(model=a.model,messages=msgs,temperature=(0.2 if (white or (mode=='segment' and attempt)) else 0),max_tokens=budget,
                            chat_template_kwargs={'enable_thinking':False},response_format={'type':'json_schema','json_schema':{'name':'trace_match','strict':True,'schema':module.Answer.model_json_schema()}})
                        at['request']=req;r=b.api(a.endpoint,req,timeout=150);at['response']=r;c=r['choices'][0];msg=c['message'];text=msg.get('content') or ''
                        if c['finish_reason']!='stop':
                            if c['finish_reason']=='length' and b.whitespace_stall(text):white=True;retry=True;raise ValueError('WHITESPACE_STALL')
                            raise ValueError('FINISH_'+str(c['finish_reason']))
                        if msg.get('reasoning_content') or msg.get('reasoning') or '<think>' in text:raise ValueError('THINKING_OUTPUT')
                        retry=True;parsed=module.validate(json.loads(text),t);at['seconds']=time.monotonic()-tick;attempts.append(at)
                        return dict(id=t['id'],task_id=t['task_id'],status='COMPLETE',parsed=parsed,attempts=attempts,annotation_protocol='full_v1_'+mode)
                    except Exception as e:
                        at.update(error=str(e),seconds=time.monotonic()-tick);attempts.append(at)
                        b.print(json.dumps(dict(event='ATTEMPT_FAILED',node=a.node,id=t['id'],mode=mode,attempt=attempt+1,error=str(e))),flush=True)
                        if mode=='free' and not retry:break
                        if white:msgs=msgs+[dict(role='user',content='Restart as compact JSON; no formatting whitespace outside strings. Include every field.')]
                        else:
                            if at.get('response'):msgs=msgs+[dict(role='assistant',content=at['response']['choices'][0]['message'].get('content') or '')]
                            msgs=msgs+[dict(role='user',content='Correct this validation error and return the complete small-group schema: '+str(e)[:700])]
            return dict(id=t['id'],task_id=t['task_id'],status='FAILED',attempts=attempts,annotation_protocol='full_v1_failed')
        pending=iter([t for k,t in tasks.items() if k not in old]);counts=collections.Counter(r['status'] for r in old.values());done=len(old);initial=done;recent=collections.deque(maxlen=50);stopped=False;exhausted=False
        def progress():
            elapsed=time.monotonic()-start;rate=(done-initial)/elapsed*3600 if elapsed else 0
            p=dict(node=a.node,planned=len(tasks),completed=done,cached=initial,statuses=dict(counts),pending=len(tasks)-done,seconds=round(elapsed,1),rate_per_hour=round(rate,1),eta_hours=(len(tasks)-done)/rate if rate else None,circuit_stopped=stopped)
            b.write(d/'progress.json',p);b.print(json.dumps(p),flush=True)
        progress()
        with out.open('a',encoding='utf-8') as f,cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
            active={}
            while active or not exhausted:
                cap=1 if done-initial<3 else a.workers
                while not exhausted and not stopped and len(active)<cap:
                    t=next(pending,None)
                    if t is None:exhausted=True;break
                    active[pool.submit(work,t)]=t['id']
                if not active:break
                complete,_=cf.wait(active,timeout=20,return_when=cf.FIRST_COMPLETED)
                for future in complete:
                    active.pop(future);r=future.result();f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();done+=1;counts[r['status']]+=1;recent.append(r['status']!='COMPLETE')
                    if (done-initial<=3 and r['status']=='FAILED') or (len(recent)>=5 and all(list(recent)[-5:])) or (len(recent)==50 and sum(recent)>=10):stopped=True
                progress()
        b.write(d/f'timing_{time.time_ns()}.json',dict(seconds=time.monotonic()-start,new_results=done-initial,cached=initial,complete=done==len(tasks)))
        if stopped:raise SystemExit('Circuit stopped; inspect failures. Existing results retained.')

def logs(a):
    m=b.read(a.root/'manifest.json');total=0;ok=0;failed=0;etas=[]
    for n in range(8):
        p=a.root/'nodes'/str(n)/'progress.json'
        if not p.exists():b.print('NODE',n,'NOT_STARTED');continue
        r=b.read(p);total+=r['completed'];ok+=r['statuses'].get('COMPLETE',0);failed+=r['statuses'].get('FAILED',0)
        if r['eta_hours'] is not None:etas.append(r['eta_hours'])
        b.print(json.dumps(r),flush=True)
    b.print(json.dumps(dict(total=total,planned=m['groups'],successful=ok,failed=failed,remaining=m['groups']-total,max_reported_eta_hours=max(etas) if etas else None,note='Snapshot only; ETA invalid for stopped/stale/not-started nodes.')))

def merge(a):
    m=check(a)
    if b.sha(a.root/'ledger_PRIVATE.jsonl')!=m['ledger_sha256']:raise ValueError('LEDGER_DRIFT')
    by=collections.defaultdict(list)
    for n in range(8):
        d=a.root/'nodes'/str(n);tasks=unique(b.rows(d/'tasks_PRIVATE.jsonl'),'id');rs=unique(b.rows(d/'results_PRIVATE.jsonl'),'id')
        if b.sha(d/'tasks_PRIVATE.jsonl')!=m['files'][f'nodes/{n}/tasks_PRIVATE.jsonl'] or set(rs)!=set(tasks):raise ValueError('NODE_INCOMPLETE_OR_DRIFT '+str(n))
        for rid,r in rs.items():
            if r['status']=='COMPLETE':
                t=tasks[rid];ds=r['parsed']['decisions']
                if len(ds)!=len(t['facts']) or {x['fact_id'] for x in ds}!={x['fact_id'] for x in t['facts']}:raise ValueError('FACT_CENSUS')
                for dec in ds:
                    for q in dec['quotes']:
                        if t['explanation'][q['start']:q['end']]!=q['text']:raise ValueError('QUOTE_DRIFT')
            elif r['status']!='FAILED':raise ValueError('UNKNOWN_STATUS')
            by[r['task_id']].append(r)
    out=a.root/'export';out.mkdir(exist_ok=True);rows=[]
    for r in b.rows(a.root/'ledger_PRIVATE.jsonl'):
        rs=by[r['task_id']];ds=[x for z in rs if z['status']=='COMPLETE' for x in z['parsed']['decisions']]
        available=r['match_input_status']=='READY';complete=available and len(rs)==r['match_groups'] and all(z['status']=='COMPLETE' for z in rs) and not any(x['status']=='ambiguous' for x in ds)
        r.update(decisions=ds,operational_exact_fact_ids=sorted({x['fact_id'] for x in ds if x['status']=='exact'}) if available else None,
            visible_fact_scan_status='COMPLETE' if complete else ('PARTIAL' if available else 'UNAVAILABLE'),accepted_exact_fact_ids=None,formal_quality_accepted=False,
            upstream_issues_status='RETAINED_NOT_ADJUDICATED',annotation_protocols=sorted({z['annotation_protocol'] for z in rs}))
        rows.append(r)
    b.dump(out/'matching_candidates_PRIVATE.jsonl',rows)
    report=dict(explanations=len(rows),groups=m['groups'],group_statuses=dict(collections.Counter(r['status'] for rs in by.values() for r in rs)),
        explanation_statuses=dict(collections.Counter(r['visible_fact_scan_status'] for r in rows)),formal_coverage_ready=False,
        note='Operational candidate matching. No expert semantic certification; unresolved upstream audit flags are retained. No failed or unavailable task dropped.')
    b.write(out/'report.json',report)
    # Deterministic case-level sample, plus all incomplete explanation IDs.
    ids=sorted({r['patient_id'] for r in rows},key=lambda x:hashlib.sha256(('review-fixed-'+str(x)).encode()).hexdigest())[:20]
    chosen={r['task_id'] for r in rows if r['patient_id'] in ids};sample=[]
    for n in range(8):
        for t in b.rows(a.root/'nodes'/str(n)/'tasks_PRIVATE.jsonl'):
            if t['task_id'] in chosen:sample.append(dict(task=t,result=next(z for z in by[t['task_id']] if z['id']==t['id'])))
    b.dump(out/'review_sample_PRIVATE.jsonl',sample)
    b.dump(out/'incomplete_PRIVATE.jsonl',[r for r in rows if r['visible_fact_scan_status']!='COMPLETE'])
    with tarfile.open(out/'full_match_review_PRIVATE.tar.gz','w:gz') as tf:
        for name in ['report.json','matching_candidates_PRIVATE.jsonl','review_sample_PRIVATE.jsonl','incomplete_PRIVATE.jsonl']:tf.add(out/name,arcname=name)
        tf.add(a.root/'manifest.json',arcname='manifest.json')
    b.print(json.dumps(report,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','logs','merge']);p.add_argument('--root',type=Path,default=Path('outputs/trace_match_full_v1'));p.add_argument('--source',type=Path)
    p.add_argument('--node',type=int,choices=range(8),default=0);p.add_argument('--workers',type=int,default=16);p.add_argument('--endpoint',default=os.environ.get('TRACE_ENDPOINT','http://localhost:8802/v1/chat/completions'));p.add_argument('--model',default='Qwen3.8-27B');a=p.parse_args()
    if not 1<=a.workers<=128:p.error('workers must be 1..128')
    globals()[a.command](a)
if __name__=='__main__':main()
