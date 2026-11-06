"""Two calls per explanation: dictionary routing then value mapping; no repair."""
import argparse,os,time,math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field
from contract import *
from api_pipeline import request_json
from extraction_contract import extraction_dictionary

class Strict(BaseModel):model_config=ConfigDict(extra='forbid',strict=True)
class Mention(Strict):
    quote:str=Field(min_length=1)
    feature_id:str|None
class Routed(Strict):mentions:list[Mention]
class Mapping(Strict):
    mention_index:int
    value:str|None
    surface_value:str
    resolution:Literal['exact','qualitative','unresolved']
    polarity:Literal['affirmed','negated','uncertain']
class Mapped(Strict):mappings:list[Mapping]

ROUTE='''Extract patient-specific feature mentions from the complete explanation. Select feature IDs from the supplied catalogue. Split distinct features into separate mentions; preserve all values for multivalued features as separate mentions where possible. Copy exact contiguous quotes. Include demographics, negation, history, qualifiers and uncertain patient assertions. Do not convert general medical rules or hypothetical findings to patient facts. Never infer findings from a diagnosis. Use feature_id=null for assertions that cannot be mapped confidently; retain those assertions. Do not force a similar feature. This is initial extraction, not medical correction. JSON only, no thinking.'''
MAP='''Map each supplied mention to its feature's legal value using only the explanation, never imagined patient truth. Return exactly one mapping per mention_index. If a quote contains multiple values, the routing step must already have separated them; otherwise use unresolved instead of silently selecting one. exact requires explicit wording or an unambiguous paraphrase with the same laterality, degree, timing and subject. Unspecified side is not both sides. Qualitative severity/speed cannot be converted to numbers. For numeric exact values, surface_value must be the explicit numeric substring and equal the value. Qualitative or unresolved mappings have value=null. polarity affirms/negates the proposition feature=value: a value meaning no travel is affirmed by no travel, not negated again. Retain ambiguities. JSON only, no thinking.'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--generation',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:8802/v1/chat/completions');p.add_argument('--model',default='Qwen3.8-27B')
    p.add_argument('--patients',type=int,default=1000,help='0 selects all patients; otherwise fixed hash subset')
    p.add_argument('--num-shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--workers',type=int,default=32);p.add_argument('--max-tokens',type=int,default=4096)
    a=p.parse_args()
    if a.workers<1 or a.patients<0 or a.num_shards<1 or not 0<=a.shard_index<a.num_shards:raise ValueError('Invalid counts')
    gm=read(a.generation/'manifest.json');data=Path(gm['data_path']);dm=read(data/'manifest.json')
    if sha(data/'manifest.json')!=gm['data_sha256']:raise ValueError('Data drift')
    for name in ['patients_public.jsonl','evidence_dictionary.json']:
        if sha(data/name)!=dm['artifacts'][name]:raise ValueError('Data artifact drift')
    patients={r['patient_id']:r for r in rows(data/'patients_public.jsonl')};contract=read(data/'evidence_dictionary.json')
    requested={r['patient_id'] for r in rows(a.generation/'requests.jsonl')}
    patients={pid:pat for pid,pat in patients.items() if pid in requested}
    chosen=set(sorted(patients,key=lambda pid:digest(['d0-initial-261005',pid]))[:a.patients] if a.patients else patients)
    global_chosen=set(chosen)
    chosen={pid for pid in chosen if int(digest(['d0-shard',pid]),16)%a.num_shards==a.shard_index}
    catalogue={k:dict(question=v['question'],type=v['type'],group_id=v['group_id']) for k,v in contract['features'].items()}
    root=a.endpoint.rsplit('/v1/chat/completions',1)[0]
    models=request_json(root+'/v1/models');service=next(r for r in models['data'] if r['id']==a.model);context=service.get('max_model_len')
    if not context:raise ValueError('Missing server context limit')
    identity=dict(num_shards=a.num_shards,shard_index=a.shard_index,global_selection_sha256=digest(sorted(global_chosen)),generation_manifest_sha256=sha(a.generation/'manifest.json'),results_sha256={f.name:sha(f) for f in a.generation.glob('results*.jsonl')},selected_patients=sorted(chosen),model=a.model,endpoint=a.endpoint,max_tokens=a.max_tokens,code={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},protocol='initial_d0_two_pass_no_patient_truth_no_repair')
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'manifest.json').exists():
        if read(a.out/'manifest.json')!=identity:raise ValueError('Resume identity mismatch; use new output')
    else:dump(a.out/'manifest.json',identity)
    dump(a.out/'service_observed.json',dict(service=service,note='Observed model card only; not a loaded-weight attestation'))
    lock=a.out/'worker.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    output=a.out/'d0.jsonl';done={r['explanation_id'] for r in rows(output)} if output.exists() else set()
    start=time.time();count=0
    def api(system,payload,typ,record):
        messages=[dict(role='system',content=system),dict(role='user',content=json.dumps(payload,ensure_ascii=False))]
        attempt=dict(messages=messages,started_at=time.time());record.setdefault('attempts',[]).append(attempt)
        tokens=request_json(root+'/tokenize',dict(model=a.model,messages=messages,add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False}))
        attempt['prompt_tokens_preflight']=tokens['count']
        if tokens['count']+a.max_tokens>context:raise ValueError('Context overflow; no truncation')
        raw=request_json(a.endpoint,dict(model=a.model,messages=messages,temperature=0,top_p=1,max_tokens=a.max_tokens,chat_template_kwargs={'enable_thinking':False},response_format={'type':'json_object'}))
        attempt.update(response=raw,finished_at=time.time())
        c=raw['choices'][0];msg=c['message']
        if c['finish_reason']!='stop':raise ValueError('Truncated/unfinished extraction')
        if msg.get('reasoning_content') or msg.get('reasoning') or '<think>' in (msg.get('content') or ''):raise ValueError('Unexpected thinking output')
        return typ.model_validate(json.loads(msg['content'])).model_dump()
    def extract(r):
        result=dict(explanation_id=r['id'],patient_id=r['patient_id'],input_view_id=r.get('input_view_id'),source_run_id=gm['run_id'],generation_status=r['status'],status='FAILED',claims=[],issues=[],coverage_semantic_status='UNREVIEWED',relation_status='NOT_EXTRACTED',started_at=time.time())
        if r['status']!='VALID':return dict(result,status='GENERATION_FAILED',generation_error=r.get('error'))
        text=r['parsed']['explanation'];result['explanation']=text;result['ranked_diagnoses']=r['parsed']['ranked_diagnoses']
        try:
            # Schema is sent as text: constrained types checked by Pydantic locally.
            route=api(ROUTE,dict(explanation=text,features=catalogue,output_schema=Routed.model_json_schema()),Routed,result)
            result['routing_initial']=route
            valid=[]
            for i,mention in enumerate(route['mentions']):
                if mention['quote'] not in text or (mention['feature_id'] is not None and mention['feature_id'] not in catalogue):
                    result['issues'].append(dict(mention_index=i,reason='INVALID_QUOTE_OR_FEATURE',initial=mention));continue
                if mention['feature_id'] is None:
                    result['claims'].append(dict(mention_index=i,**mention,status='UNMAPPED',matched_fact_ids=[]));continue
                valid.append(dict(mention_index=i,**mention))
            if valid:
                dictionary=extraction_dictionary(contract,sorted({m['feature_id'] for m in valid}))
                mapped=api(MAP,dict(explanation=text,mentions=valid,dictionary=dictionary,output_schema=Mapped.model_json_schema()),Mapped,result)
                result['mapping_initial']=mapped;byindex={m['mention_index']:m for m in valid};seen=set()
                for mapping in mapped['mappings']:
                    i=mapping['mention_index']
                    try:
                        if i not in byindex or i in seen:raise ValueError('Unknown or duplicate mention index')
                        seen.add(i);mention=byindex[i];meta=dictionary[mention['feature_id']];v=mapping['value']
                        if (mapping['resolution']=='exact')!=(v is not None):raise ValueError('Resolution/value mismatch')
                        if mapping['surface_value'] and mapping['surface_value'] not in mention['quote']:raise ValueError('Surface not verbatim')
                        if v is not None:
                            if meta['values'] and v not in meta['values']:raise ValueError('Illegal value')
                            if meta['numeric_values']:
                                number=float(mapping['surface_value'])
                                if not math.isfinite(number) or number!=float(v):raise ValueError('Numeric inference forbidden')
                        claim=dict(mention,**{k:v for k,v in mapping.items() if k!='mention_index'},value_resolution=mapping['resolution'])
                        aligned=align(claim,patients[r['patient_id']],contract)
                        if mapping['resolution']!='exact':aligned['status']='QUALITATIVE_UNASSESSED' if mapping['resolution']=='qualitative' else 'UNCERTAIN'
                        quote=mention['quote'];positions=[];pos=0
                        while (pos:=text.find(quote,pos))>=0:positions.append([pos,pos+len(quote)]);pos+=1
                        aligned.update(quote_offsets=positions,exact_coverage_validated=False)
                        result['claims'].append(aligned)
                    except (ValueError,TypeError,KeyError) as e:result['issues'].append(dict(mention_index=i,reason=str(e),initial=mapping))
                for i in byindex.keys()-seen:result['issues'].append(dict(mention_index=i,reason='MISSING_MAPPING'))
            result['status']='PARTIAL' if result['issues'] else 'INITIAL_COMPLETE_UNREVIEWED'
        except Exception as e:result['error']=str(e)
        result['finished_at']=time.time();return result
    def source():
        visited=set()
        for f in sorted(a.generation.glob('results*.jsonl')):
            for r in rows(f):
                if r['patient_id'] not in chosen:continue
                if r['id'] in visited:raise ValueError('Duplicate generation result; consolidate attempts explicitly')
                visited.add(r['id'])
                if r['id'] not in done:yield r
    try:
        iterator=iter(source())
        with output.open('a',encoding='utf-8',newline='\n') as out,ThreadPoolExecutor(max_workers=a.workers) as pool:
            pending=set()
            while True:
                while len(pending)<a.workers*2:
                    r=next(iterator,None)
                    if r is None:break
                    pending.add(pool.submit(extract,r))
                if not pending:break
                ready,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in ready:
                    out.write(json.dumps(future.result(),ensure_ascii=False)+'\n');out.flush();count+=1
                if count%20<len(ready):print(f'Extracted {count}; {count*3600/max(time.time()-start,1):.0f} explanations/hour',flush=True)
        counts=Counter();claim_counts=Counter();calls=0;usage=Counter();ids=set()
        for r in rows(output):
            ids.add(r['explanation_id']);counts[r['status']]+=1;claim_counts.update(c['status'] for c in r['claims']);calls+=len(r.get('attempts',[]))
            for t in r.get('attempts',[]):usage.update({k:v for k,v in t.get('response',{}).get('usage',{}).items() if isinstance(v,(int,float))})
        expected={t['id'] for t in rows(a.generation/'requests.jsonl') if t['patient_id'] in chosen}
        write(a.out/'patients_public.jsonl',(patients[pid] for pid in sorted(chosen)))
        dump(a.out/'report.json',dict(patients=len(chosen),planned_explanations=len(expected),completed=len(ids&expected),missing=len(expected-ids),statuses=dict(counts),claim_statuses=dict(claim_counts),api_attempts=calls,tokens=dict(usage),invocation_seconds=time.time()-start,note='d0 initial extraction. MATCHED is record compatibility, not independently verified exact coverage. Routing may omit mentions. No repair, no clinical validity labels. Null mappings and failures retained.'))
        print(read(a.out/'report.json'))
    finally:lock.unlink()

if __name__=='__main__':main()
