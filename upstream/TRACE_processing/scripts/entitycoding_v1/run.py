"""Eight independent GPU workers; upstream NER/AC weights, all assertions retained.
Adapter differences: per-sentence NER; offset-safe splitting of overlong sentences;
no entity filtering; batched AC; explicit partial failures. No ontology normalization.
"""
import argparse,json,os,sys,time,traceback,importlib.metadata
from pathlib import Path
from common import sha,write,alignment,raw_span,span_issue

def chunks(text,tokenizer,start=0):
    if len(tokenizer.encode(text))<=512:
        return [(start,text)]
    if len(text)<2:raise ValueError('Cannot split overlong input')
    middle=len(text)//2
    # Keep every character, including whitespace; never shift offsets by strip().
    return chunks(text[:middle],tokenizer,start)+chunks(text[middle:],tokenizer,start+middle)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--input',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--ac-batch-size',type=int,default=16)
    p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
    rank=int(os.environ.get('LOCAL_RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    if world!=8:raise ValueError('Run with torch.distributed.run --nproc_per_node=8')
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7').split(',')
    if len(devices)!=8:raise ValueError('Expected eight visible GPU identifiers')
    os.environ['CUDA_VISIBLE_DEVICES']=devices[rank]
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no silent CPU fallback')
    sys.path.insert(0,str(a.repo.resolve()/'ner'))
    import extract_entities as upstream
    a.out.mkdir(parents=True,exist_ok=True)
    target=a.out/f'node{rank}.jsonl'
    # Exclusive creation prevents accidental overwrites/concurrent writers.
    output=target.open('x',encoding='utf-8')
    started=time.monotonic()
    try:
        device,tokenizer,model,classifier=upstream.get_models()
        nlp=upstream.build_cleaning_nlp()
        ac_labels={'LABEL_0':'ABSENT','LABEL_1':'FAMILY','LABEL_2':'HYPOTHETICAL','LABEL_3':'POSSIBLE','LABEL_4':'PRESENT'}
        if set(classifier.model.config.id2label.values())!=set(ac_labels):
            raise ValueError('AC config differs from upstream LABEL_0..4 mapping; inspect before running')
        meta={'rank':rank,'world':world,'device':torch.cuda.get_device_name(0),'ner_dtype':str(next(model.parameters()).dtype),
            'ac_dtype':str(next(classifier.model.parameters()).dtype),'assertion_map':ac_labels,
            'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','spacy','medspacy')},
            'upstream_sha256':sha(a.repo/'ner/extract_entities.py'),'adapter_sha256':sha(Path(__file__)),
            'input_sha256':sha(a.input) if a.input else None,'offset_unit':'Python Unicode code points; end exclusive',
            'differences':['sentence-wise NER rather than sentence packing','offset-safe recursive character split for long sentences',
                'no upstream entity/assertion/section filtering','batched AC; all raw predictions retained'],
            'interpretation':'Candidate mentions, not clinically validated facts. FAMILY retains upstream meaning; no inferred temporality or experiencer.'}
        if rank==0:
            meta['model_files']={str(f.relative_to(a.repo)):sha(f) for folder in ('ner_model','ac_model')
                for f in sorted((a.repo/'data/models'/folder).iterdir()) if f.is_file()}
        write(a.out/f'identity{rank}.json',meta)
        docs=[{'document_id':f'synthetic_{rank}','patient_id':'synthetic','source_field':'synthetic',
            'text':'History of Present Illness:\nThe patient reports abdominal pain. No vomiting.\nPhysical Examination:\nThe abdomen is soft.'}]
        if not a.smoke_only:
            if not a.input:raise ValueError('--input required unless --smoke-only')
            docs=[json.loads(line) for i,line in enumerate(a.input.read_text(encoding='utf-8-sig').splitlines()) if line.strip() and i%world==rank]
        counts={};entities_total=0
        for record in docs:
            result={**record,'entities':[],'errors':[],'rejected_ner_spans':[],'status':'COMPLETE'}
            try:
                raw=record['text'];clean,sents,starts=upstream.clean_document_sentences(raw,nlp=nlp)
                result['cleaned_text']=clean; mapping=alignment(raw,clean);doc=nlp(clean)
                candidates=[];contexts=[];context_entities=[];num_chunks=0
                for sent,base in zip(sents,starts):
                    parts=chunks(sent,tokenizer);num_chunks+=len(parts)
                    for delta,text in parts:
                        for entity in upstream.infer_sentences(text):
                            local_start=int(entity['start']);local_end=int(entity['end'])
                            start=base+delta+local_start;end=base+delta+local_end
                            issue=span_issue(text,local_start,local_end)
                            if issue:
                                result['rejected_ner_spans'].append({'reason':issue,'entity_type':entity['label'],
                                    'clean_start':start,'clean_end':end,'clean_quote':clean[start:end],
                                    'chunk_clean_start':base+delta,'local_start':local_start,'local_end':local_end})
                                continue
                            rs,re=raw_span(mapping,start,end)
                            item={'mention_id':record['document_id']+f':m{len(candidates)}','entity_type':entity['label'],
                                'clean_start':start,'clean_end':end,'clean_quote':clean[start:end],
                                'raw_start':rs,'raw_end':re,'raw_quote':raw[rs:re],
                                'assertion':None,'assertion_score':None,'status':'PENDING',
                                'concept_id':None,'normalized_value':None,'temporality':None,'experiencer':None}
                            candidates.append(item)
                            span=doc.char_span(start,end,alignment_mode='expand')
                            if span is None:
                                item['status']='UNALIGNED_FOR_AC';continue
                            sentence=span.sent
                            left=doc[max(sentence.start,span.start-100):span.start].text
                            right=doc[span.end:min(sentence.end,span.end+25)].text
                            context=f'{left} <entity> {span.text} <entity> {right}'
                            item['ac_context']=context;item['ac_span']=[span.start_char,span.end_char]
                            item['ac_span_expanded']=(start,end)!=(span.start_char,span.end_char)
                            limit=min(512,int(classifier.tokenizer.model_max_length))
                            if len(classifier.tokenizer.encode(context))>limit:
                                item['status']='AC_CONTEXT_TOO_LONG';continue
                            contexts.append(context);context_entities.append(item)
                for offset in range(0,len(contexts),a.ac_batch_size):
                    items=context_entities[offset:offset+a.ac_batch_size]
                    try:
                        predictions=classifier(contexts[offset:offset+a.ac_batch_size],batch_size=a.ac_batch_size,truncation=False)
                        if len(predictions)!=len(items):raise ValueError('AC output count mismatch')
                        for item,pred in zip(items,predictions):
                            item.update(assertion=ac_labels[pred['label']],assertion_score=float(pred['score']),status='PREDICTED')
                    except Exception as e:
                        for item in items:item.update(status='AC_FAILED',error=repr(e))
                result['entities']=candidates;result['ner_chunks']=num_chunks
                if any(x['status']!='PREDICTED' for x in candidates):result['status']='PARTIAL'
                result['quality_note']='Engineering completion is not semantic correctness. No entity detection can also be a false negative.'
                entities_total+=len(candidates)
            except Exception as e:
                result['status']='FAILED';result['errors'].append(repr(e));result['traceback']=traceback.format_exc()
            output.write(json.dumps(result,ensure_ascii=False)+'\n');output.flush()
            counts[result['status']]=counts.get(result['status'],0)+1
            print(f'node={rank} documents={sum(counts.values())}/{len(docs)} entities={entities_total} statuses={counts}',flush=True)
        report={'node':rank,'documents':len(docs),'entities':entities_total,'statuses':counts,'elapsed_seconds':time.monotonic()-started,
            'smoke_only':a.smoke_only,'output_sha256':sha(target)}
        write(a.out/f'report{rank}.json',report)
        if counts.get('FAILED') or counts.get('PARTIAL'):
            print(f'node={rank} finished with document failures; retained for acceptance',flush=True)
            if a.smoke_only:raise RuntimeError('Synthetic model check failed; inspect node output')
    finally:output.close()
if __name__=='__main__':main()
