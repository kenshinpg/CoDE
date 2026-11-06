"""Merge only after all eight workers exit; keep failed documents in export."""
import argparse,json,collections
from pathlib import Path
from common import write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    reports=[];records=[];identities=[]
    for n in range(8):
        reports.append(json.loads((a.run/f'report{n}.json').read_text(encoding='utf-8-sig')))
        identities.append(json.loads((a.run/f'identity{n}.json').read_text(encoding='utf-8-sig')))
        path=a.run/f'node{n}.jsonl'
        if sha(path)!=reports[-1]['output_sha256']:raise ValueError('Output changed since worker finished')
        local=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
        if len(local)!=reports[-1]['documents']:raise ValueError('Document count mismatch')
        records.extend(local)
    for key in ('adapter_sha256','upstream_sha256','input_sha256'):
        if len({x[key] for x in identities})!=1:raise ValueError(f'Inconsistent identity: {key}')
    ids=[x['document_id'] for x in records]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate document IDs')
    types=collections.Counter();assertions=collections.Counter();statuses=collections.Counter();expanded=0
    for r in records:
        for e in r['entities']:
            if r['text'][e['raw_start']:e['raw_end']]!=e['raw_quote']:raise ValueError('Raw span mismatch')
            if r['cleaned_text'][e['clean_start']:e['clean_end']]!=e['clean_quote']:raise ValueError('Clean span mismatch')
            types[e['entity_type']]+=1;assertions[e['assertion'] or 'UNRESOLVED']+=1;statuses[e['status']]+=1
            expanded+=bool(e.get('ac_span_expanded'))
    with (a.run/'mentions.jsonl').open('x',encoding='utf-8') as f:
        for r in sorted(records,key=lambda x:x['document_id']):f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report={'documents':len(records),'patients':len({x['patient_id'] for x in records}),
        'document_statuses':dict(collections.Counter(x['status'] for x in records)),
        'zero_entity_documents':sum(not x['entities'] for x in records),'entity_types':dict(types),
        'assertions':dict(assertions),'entity_statuses':dict(statuses),'ac_expanded_spans':expanded,
        'rejected_ner_spans':sum(len(r.get('rejected_ner_spans',[])) for r in records),'offset_check':'PASS','semantic_quality':'NOT_YET_REVIEWED','clinical_values_or_ontology_normalized':False,
        'note':'No reference labels are passed to NER or assertion classification; absence of a mention is not a negative finding.'}
    write(a.run/'acceptance.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
