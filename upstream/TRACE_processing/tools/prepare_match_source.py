"""Compute-platform adapter to the historical full match input; no NER/linking or text edits."""
import argparse, hashlib, json, shutil
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig') as f:
        for l in f:
            if l.strip():yield json.loads(l)
def main():
    p=argparse.ArgumentParser();p.add_argument('--generation',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    g=read(a.generation/'manifest.json');im=read(a.input/'manifest.json')
    if g['input_manifest_sha256']!=sha(a.input/'manifest.json'):raise ValueError('Frozen input drift')
    if sha(a.generation/'requests_PRIVATE.jsonl')!=g['requests_sha256']:raise ValueError('Requests changed')
    for name in ('patients_model_input_PRIVATE.jsonl','generation_tasks_PRIVATE.jsonl','split_without_outcomes_PRIVATE.jsonl'):
        if sha(a.input/name)!=im['output_sha256'][name]:raise ValueError('Input changed: '+name)
    tasks=list(rows(a.generation/'requests_PRIVATE.jsonl'));generations={};patients={r['patient_id']:r for r in rows(a.input/'patients_model_input_PRIVATE.jsonl')};split={r['patient_id']:r for r in rows(a.input/'split_without_outcomes_PRIVATE.jsonl')}
    for r in rows(a.generation/'results_PRIVATE.jsonl'):
        if r['task_id'] in generations:raise ValueError('Multiple result records: resolve the intended invocation explicitly before export')
        generations[r['task_id']]=r
    if len(tasks)!=9600 or len(patients)!=2400 or set(generations)!={t['task_id'] for t in tasks}:raise ValueError('Fixed census differs')
    a.out.mkdir(parents=True,exist_ok=False)
    with (a.out/'explanations_PRIVATE.jsonl').open('x',encoding='utf-8') as ledger,(a.out/'documents_PRIVATE.jsonl').open('x',encoding='utf-8') as docs:
        for t in tasks:
            r=generations[t['task_id']];pid=t['patient_id'];text=r.get('parsed_explanation')
            if r['patient_id']!=pid or r['messages']!=t['messages']:raise ValueError('Generation/task mismatch')
            if json.loads(t['messages'][1]['content'])['patient_facts']!=[f['display_text'] for f in patients[pid]['facts']]:raise ValueError('Rendering differs')
            if r['status']=='complete':
                if not text or json.loads(r['raw_response']['choices'][0]['message']['content'])['explanation']!=text:raise ValueError('Explanation changed')
                docs.write(json.dumps(dict(task_id=t['task_id'],patient_id=pid,input_view_id=t['input_view_id'],text=text),ensure_ascii=False)+'\n')
            elif r['status']!='failed':raise ValueError('Unknown generation status')
            ledger.write(json.dumps(dict(t,partition=split[pid]['partition'],generation_status=r['status']),ensure_ascii=False)+'\n')
    shutil.copyfile(a.generation/'results_PRIVATE.jsonl',a.out/'generation_records_PRIVATE.jsonl')
    (a.out/'frozen_input').mkdir()
    shutil.copyfile(a.input/'patients_model_input_PRIVATE.jsonl',a.out/'frozen_input/patients_model_input_PRIVATE.jsonl')
    names=['explanations_PRIVATE.jsonl','documents_PRIVATE.jsonl','generation_records_PRIVATE.jsonl','frozen_input/patients_model_input_PRIVATE.jsonl']
    report=dict(outputs={n:sha(a.out/n) for n in names},protocol='public_full_match_input_adapter',omitted_intermediate_records=True,note='No changes to explanation text/F; reproduces final whole-text matching input, not the original release inventory.')
    (a.out/'release_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Prepared complete matching input; terminal generation failures retained.')
if __name__=='__main__':main()
