"""Server-only adapter: EntityCoding -> HPO candidates -> local LLM selection.
Not an exact RAG-HPO reproduction. No dependencies installed by this script.
"""
import argparse, concurrent.futures, hashlib, json, os, time, re
from pathlib import Path
from collections import Counter

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        for l in f:
            if l.strip(): yield json.loads(l)
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def lock(folder):
    import fcntl
    folder.mkdir(parents=True,exist_ok=True)
    f=(folder/'.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f

def prepare(a):
    import pronto
    if a.out.exists():raise ValueError('Use a new output directory')
    ontology=pronto.Ontology(str(a.obo))
    terms=[]
    for t in ontology.terms():
        if t.obsolete:continue
        if t.id!='HP:0000118' and not any(x.id=='HP:0000118' for x in t.superclasses()):continue
        aliases=sorted({str(s.description) for s in t.synonyms if str(s.scope).upper().endswith('EXACT')})
        terms.append({'id':t.id,'name':t.name,'definition':str(t.definition or ''),'aliases':aliases})
    if not terms:raise ValueError('No phenotype terms parsed')
    terms.sort(key=lambda t:t['id'])
    a.out.mkdir(parents=True)
    write(a.out/'terms.json',terms)
    phrases=[{'term_index':i,'text':x} for i,t in enumerate(terms) for x in sorted({t['name'],*t['aliases']})]
    write(a.out/'phrases.json',phrases)
    patients={};inputs=[];seen=set()
    for f in sorted(a.entity_dir.glob('node[0-9].jsonl')):
        inputs.append({'path':str(f.resolve()),'sha256':sha(f)})
        for d in rows(f):
            if d['document_id'] in seen:raise ValueError('Duplicate document '+d['document_id'])
            seen.add(d['document_id']);patients.setdefault(d['patient_id'],[]).append(d)
    if not patients:raise ValueError('No node0.jsonl .. node7.jsonl found in --entity-dir')
    # Stable patient-level selection, independent of disease, assertion and mapping success.
    picked=sorted(patients,key=lambda x:hashlib.sha256(('261011:'+x).encode()).hexdigest())
    if a.patients:picked=picked[:a.patients]
    counts=Counter();mention_ids=set()
    with (a.out/'mentions.jsonl').open('w',encoding='utf-8') as f:
        for pid in picked:
            for d in patients[pid]:
                counts['documents']+=1
                for e in d['entities']:
                    if e['mention_id'] in mention_ids:raise ValueError('Duplicate mention ID')
                    mention_ids.add(e['mention_id'])
                    s,end=e['raw_start'],e['raw_end'];text=d['text']
                    valid=isinstance(s,int) and isinstance(end,int) and 0<=s<end<=len(text) and text[s:end]==e['raw_quote']
                    r={'mention_id':e['mention_id'],'patient_id':pid,'document_id':d['document_id'],'source_field':d['source_field'],'document_status':d['status'],'entity':e,'document_text':text,'context':text[max(0,s-400):min(len(text),end+400)] if valid else None,'span_valid':valid}
                    counts['mentions']+=1;counts['invalid_spans']+=not valid
                    f.write(json.dumps(r,ensure_ascii=False)+'\n')
    files={str(p.relative_to(a.model)):sha(p) for p in a.model.rglob('*') if p.is_file() and '.cache' not in p.parts}
    write(a.out/'manifest.json',{'model':str(a.model.resolve()),'model_files':files,'obo':str(a.obo.resolve()),'obo_sha256':sha(a.obo),'inputs':inputs,'patients':len(picked),'patient_ids':picked,'counts':dict(counts),'terms':len(terms),'phrases':len(phrases),'code_sha256':sha(__file__),'protocol':'exact HPO synonyms + labels; phenotype subtree; no inferred parent merging; original assertions retained; NOT exact RAG-HPO reproduction'})
    print(json.dumps({'patients':len(picked),'terms':len(terms),'phrases':len(phrases),**counts}),flush=True)

def encoder(model_path,device):
    import torch
    from transformers import AutoTokenizer,AutoModel
    cfg=read(Path(model_path)/'1_Pooling/config.json')
    if not cfg.get('pooling_mode_mean_tokens') or any(cfg.get(k) for k in ('pooling_mode_cls_token','pooling_mode_max_tokens','pooling_mode_mean_sqrt_len_tokens')):raise ValueError('Unsupported pooling')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModel.from_pretrained(model_path,local_files_only=True,attn_implementation='eager').to(device).eval()
    def encode(texts):
        lengths=[len(x) for x in tokenizer(texts,truncation=False)['input_ids']]
        batch=tokenizer(texts,padding=True,truncation=True,max_length=100,return_tensors='pt').to(device)
        with torch.inference_mode():
            hidden=model(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            v=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
            if not torch.isfinite(v).all():raise ValueError('Nonfinite embeddings')
            v=torch.nn.functional.normalize(v,dim=1)
        return v.cpu().numpy(),lengths
    return encode

def embed(a):
    import torch,numpy as np
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    if world!=8:raise ValueError('Use torchrun --nproc_per_node=8')
    torch.set_num_threads(2);torch.cuda.set_device(rank)
    encode=encoder(read(a.run/'manifest.json')['model'],f'cuda:{rank}')
    phrases=read(a.run/'phrases.json');ids=list(range(rank,len(phrases),world));vectors=[];lengths=[]
    for off in range(0,len(ids),a.batch_size):
        batch=ids[off:off+a.batch_size];v,n=encode([phrases[i]['text'] for i in batch]);vectors.append(v);lengths.extend(n)
        if off%(a.batch_size*20)==0:print(f'worker={rank} embedded={off+len(batch)}/{len(ids)}',flush=True)
    np.savez(a.run/f'embeddings{rank}.npz',ids=ids,vectors=np.concatenate(vectors),lengths=lengths,phrases_sha256=sha(a.run/'phrases.json'))

def retrieve(a):
    import numpy as np,torch
    guard=lock(a.run/'retrieval_lock');torch.set_num_threads(a.cpu_threads)
    if (a.run/'candidates.jsonl').exists():raise ValueError('Candidates already exist; do not overwrite')
    phrases=read(a.run/'phrases.json');terms=read(a.run/'terms.json');vec=np.zeros((len(phrases),768),dtype=np.float32);covered=set();long_alias=0
    for rank in range(8):
        data=np.load(a.run/f'embeddings{rank}.npz')
        if str(data['phrases_sha256'])!=sha(a.run/'phrases.json'):raise ValueError('Stale embeddings')
        covered.update(data['ids'].tolist());vec[data['ids']]=data['vectors'];long_alias+=int((data['lengths']>100).sum())
    if covered!=set(range(len(phrases))):raise ValueError('Incomplete embedding shards')
    matrix=torch.from_numpy(vec);encode=encoder(read(a.run/'manifest.json')['model'],'cpu')
    mentions=list(rows(a.run/'mentions.jsonl'));cache={};queries=sorted({r['entity']['raw_quote'] for r in mentions if r['span_valid']})
    for off in range(0,len(queries),a.batch_size):
        batch=queries[off:off+a.batch_size];v,lengths=encode(batch);scores=torch.from_numpy(v)@matrix.T
        for j,q in enumerate(batch):
            order=torch.argsort(scores[j],descending=True).tolist();chosen=[];seen=set()
            for idx in order:
                ti=phrases[idx]['term_index']
                if ti in seen:continue
                seen.add(ti);chosen.append({**terms[ti],'matched_alias':phrases[idx]['text'],'cosine':float(scores[j,idx])})
                if len(chosen)==20:break
            cache[q]={'candidates':chosen,'query_tokens':lengths[j],'query_truncated':lengths[j]>100}
        print(f'Retrieved {min(off+len(batch),len(queries))}/{len(queries)} unique strings',flush=True)
    tmp=a.run/'candidates.jsonl.part'
    with tmp.open('w',encoding='utf-8') as f:
        for r in mentions:
            payload=cache[r['entity']['raw_quote']] if r['span_valid'] else {'candidates':[],'query_tokens':None,'query_truncated':False}
            f.write(json.dumps({**r,**payload},ensure_ascii=False)+'\n')
    tmp.replace(a.run/'candidates.jsonl')
    write(a.run/'retrieval_report.json',{'mentions':len(mentions),'unique_strings':len(queries),'truncated_aliases':long_alias,'truncated_queries':sum(v['query_truncated'] for v in cache.values()),'note':'Retrieval scores are not calibrated correctness probabilities.'})
