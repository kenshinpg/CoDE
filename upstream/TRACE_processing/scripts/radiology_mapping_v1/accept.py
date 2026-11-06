"""Inventory raw mapping/review outputs; never apply review corrections implicitly."""
import argparse,collections,json
from pathlib import Path
from pipeline import rows,read,write,sha,validate,dump
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    m=read(a.root/'manifest.json');a.out.mkdir(parents=True,exist_ok=False);counts=collections.Counter();units=set();parents=set()
    with (a.out/'mapping_review_records_PRIVATE.jsonl').open('x',encoding='utf-8') as f:
        for shard in range(8):
            tf=a.root/f'tasks{shard}.jsonl'
            if sha(tf)!=m['task_hashes'][str(shard)]:raise ValueError('Task changed')
            stages={}
            for stage in ('map','review'):
                rp=a.root/stage/f'results{shard}.jsonl';dp=a.root/stage/f'done{shard}.json'
                if not dp.exists():raise ValueError(f'{stage} shard {shard} not complete')
                if sha(rp)!=read(dp)['results_sha256']:raise ValueError('Results changed since completion')
                stages[stage]={r['id']:r for r in rows(rp)}
            for t in rows(tf):
                mr=stages['map'].get(t['id']);rr=stages['review'].get(t['id']);mi={};ri={}
                if mr and mr['status']=='COMPLETE':mi={i['unit_id']:i for i in validate(mr['parsed'],t,'map')['items']}
                if rr and rr['status']=='COMPLETE':ri={i['unit_id']:i for i in validate(rr['parsed'],t,'review',mr['parsed'])['items']}
                for u in t['units']:
                    uid=u['unit_id']
                    if uid in units:raise ValueError('Duplicate unit')
                    units.add(uid)
                    if u['unit_kind']=='parent':parents.add(u['parent_mention_id'])
                    counts['units']+=1;counts['map_'+(mr['status'] if mr else 'MISSING')]+=1;counts['review_'+(rr['status'] if rr else 'MISSING')]+=1
                    if uid in ri:counts['verdict_'+ri[uid]['verdict']]+=1
                    dump(f,dict(unit=u,mapping=mi.get(uid),review=ri.get(uid),map_status=mr['status'] if mr else 'MISSING',review_status=rr['status'] if rr else 'MISSING',review_correction_applied=False,clinical_fact_ready=False))
    if len(units)!=m['units'] or len(parents)!=m['parents']:raise ValueError('Coverage mismatch')
    report=dict(counts=dict(counts),parents=len(parents),all_original_parents_retained=True,semantic_accuracy_established=False,scope='Candidate-conditioned same-model review; not full-document omission audit. Raw and proposed links separate; no implicit clinical merge.')
    write(a.out/'report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
