"""Compute-platform helpers for source metadata; no model inference."""
import argparse, csv, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='action',required=True)
    q=s.add_parser('lab-dictionary');q.add_argument('--mapping-csv',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q=s.add_parser('review-partitions');q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.action=='lab-dictionary':
        with a.mapping_csv.open(encoding='utf-8-sig',newline='') as f:
            rows=[{k:r.get(k) for k in ('itemid','label','fluid','category','corresponding_ids')} for r in csv.DictReader(f)]
        if not rows or any(not r['itemid'] or not r['label'] for r in rows):raise ValueError('Missing itemid/label')
        if len({r['itemid'] for r in rows})!=len(rows):raise ValueError('Duplicate native item IDs')
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open('x',encoding='utf-8') as f:json.dump(rows,f,ensure_ascii=False,indent=2)
    else:
        a.out.mkdir(parents=True,exist_ok=False)
        for part in ('cost_pilot','development','evaluation'):
            (a.out/(part+'_PRIVATE.jsonl')).touch(exist_ok=False)
        (a.out/'README.txt').write_text('Empty review-membership metadata for a fresh rebuild. Not historical membership and not diagnostic train/test partitions. Export assigns remaining.\n',encoding='utf-8')
    print('Prepared',a.out)
if __name__=='__main__':main()
