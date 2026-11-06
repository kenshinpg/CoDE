"""Join frozen tasks and latest audit records without rewriting clinical facts."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
from review import rows, read, write, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True);a=p.parse_args()
    m=read(a.run/'manifest.json')
    if sha(a.run/'tasks_PRIVATE.jsonl')!=m['tasks_sha256']:raise ValueError('Task fingerprint changed')
    latest={}; result_files={}
    for f in sorted(a.run.glob('node*_results_PRIVATE.jsonl')):
        result_files[str(f)]=sha(f)
        for r in rows(f):latest[r['id']]=r
    groups={}
    for part in ('cost_pilot','development','evaluation'):
        for r in rows(a.split/(part+'_PRIVATE.jsonl')):
            if r['patient_id'] in groups:raise ValueError('Review split overlap')
            groups[r['patient_id']]=part
    tasks=list(rows(a.run/'tasks_PRIVATE.jsonl'));ids={t['id'] for t in tasks}
    task_mids={t['id']:{x['mention_id'] for x in t['mentions']} for t in tasks}
    mention_doc={x['mention_id']:t['document_id'] for t in tasks for x in t['mentions']}
    chosen=set(m['patients'])
    if set(latest)-ids:raise ValueError('Unexpected result IDs')
    out=a.run/'export_v1';out.mkdir(exist_ok=True)
    docs=defaultdict(list);status=Counter();issues=Counter();states=Counter()
    with (out/'audit_tasks_PRIVATE.jsonl').open('w',encoding='utf-8') as f:
        for t in tasks:
            r=latest.get(t['id']);s=r['status'] if r else 'MISSING';status[s]+=1
            review=r.get('review') if r else None
            x={k:t[k] for k in ('id','patient_id','document_id','stage')}
            x.update(status=s,review=review,review_partition=groups.get(t['patient_id'],'remaining'))
            docs[t['document_id']].append(x)
            f.write(json.dumps(x,ensure_ascii=False)+'\n')
            if review:issues.update(i['kind'] for i in review['issues'])
    link=Path(m['link_directory'])/'linked_PRIVATE.jsonl'
    if sha(link)!=m['linked_sha256']:raise ValueError('Mapping source changed')
    mentions=0
    with (out/'mentions_reviewed_PRIVATE.jsonl').open('w',encoding='utf-8') as f:
        for r in rows(link):
            if r['patient_id'] not in chosen:continue
            reviews=docs.get(mention_doc[r['mention_id']],[])
            relevant=[x for x in reviews if x['stage']=='extraction' or r['mention_id'] in task_mids.get(x['id'],set())]
            flags=[i for x in relevant for i in (x.get('review') or {}).get('issues',[]) if i['mention_id']==r['mention_id']]
            omissions=[i for x in relevant for i in (x.get('review') or {}).get('issues',[]) if i['mention_id'] is None]
            complete=len(relevant)==2 and all(x['status']=='COMPLETE' for x in relevant)
            state='REVIEW_INCOMPLETE' if not complete else ('MODEL_FLAGGED' if flags else 'NO_MODEL_FLAG')
            states[state]+=1;mentions+=1
            # Preserve source clinical content; raw API traces remain in immutable source files.
            x={k:v for k,v in r.items() if k not in ('attempts','candidates','document_text')}
            x.update(audit_state=state,audit_flags=flags,document_omission_flags=omissions,
                     review_partition=groups.get(r['patient_id'],'remaining'),
                     audit_task_ids=[v['id'] for v in relevant],source_link_file=str(link),
                     clinical_correction_applied=False)
            f.write(json.dumps(x,ensure_ascii=False)+'\n')
    all_complete=status['COMPLETE']==len(tasks)
    errors=Counter(v.get('error','unknown') for r in latest.values() if r['status']=='FAILED' for v in r.get('attempts',[])[-1:])
    report={'dataset_version':'v1','review_complete':all_complete,'tasks':len(tasks),
            'task_statuses':dict(status),'mentions':mentions,'mention_states':dict(states),
            'model_issue_counts_not_gold_errors':dict(issues),
            'admissions':m['admissions'],'documents':m['documents'],'final_errors':dict(errors),
            'manifest_sha256':sha(a.run/'manifest.json'),'result_files':result_files,
            'source_link_sha256':m['linked_sha256'],
            'note':'No clinical corrections applied. Null retained, not negative. NO_MODEL_FLAG does not establish correctness or recall. Original documents and traces stay in source files. Review partition is not diagnostic train/test.'}
    write(out/'acceptance.json',report);print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if not all_complete:raise SystemExit(2)

if __name__=='__main__':main()
