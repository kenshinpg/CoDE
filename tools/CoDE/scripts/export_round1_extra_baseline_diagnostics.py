"""Export existing subset-search diagnostics; no models, inference or raw text export."""
import argparse, json, math, statistics, tarfile, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def scan(folder):
    if read(folder/'run_manifest.json')['status'] != 'complete':
        raise ValueError('Incomplete shard: '+str(folder))
    rows=[]
    with (folder/'selections_PRIVATE.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['probe'] != 'encoder':
                continue
            detail=r['details']
            rows.append({k:r[k] for k in ['patient_id','method','fraction','both_full_correct','available','fact_count','group_count','full_group_count','correct','random_accuracy','search_status','unique_tree_queries']} | {
                'search_queries':detail.get('search_queries'),
                'verification_success':detail.get('verification_success'),
                'verification_n':detail.get('verification_n'),
                'search_precision':detail.get('search_precision'),
                'perturbation_coverage':detail.get('perturbation_coverage')})
    return folder.parent.name,rows


def distribution(values):
    xs=sorted(float(x) for x in values if x is not None)
    if not xs:return {'n':0}
    def quantile(q):
        i=(len(xs)-1)*q;lo=int(i);hi=math.ceil(i)
        return xs[lo]+(xs[hi]-xs[lo])*(i-lo)
    return dict(n=len(xs),mean=statistics.mean(xs),median=statistics.median(xs),
                p25=quantile(.25),p75=quantile(.75),min=xs[0],max=xs[-1],total=sum(xs))


def main(a):
    if read(a.source/'run_manifest.json')['status'] != 'complete':
        raise ValueError('Original run did not complete')
    folders=sorted(a.source.glob('*/shard_*')); allrows=defaultdict(list)
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=min(a.workers,len(folders))) as pool:
        futures={pool.submit(scan,f):f for f in folders}
        for i,f in enumerate(as_completed(futures),1):
            ds,rows=f.result();allrows[ds].extend(rows)
            print('READ',i,'/',len(folders),futures[f],f'{time.monotonic()-started:.1f}s',flush=True)
    if set(allrows) != {'ddx','trace'}:
        raise ValueError('Both dataset caches are required')
    result={'status':'complete','models_loaded':False,'inference_calls':0,'datasets':{}}
    for ds,rs in allrows.items():
        expected=1506 if ds=='ddx' else 359
        if len(rs)!=expected*11 or len({(r['patient_id'],r['method'],r['fraction']) for r in rs})!=len(rs):
            raise ValueError('Row census mismatch: '+ds)
        source=read(a.source/ds/'summary.json')
        original={r['name']:r for r in source['panels'] if r['kind']=='selection'}
        groups=defaultdict(list)
        for r in rs:groups[(r['method'],r['fraction'])].append(r)
        panels=[]
        for (method,fraction),pool in sorted(groups.items(),key=lambda x:str(x[0])):
            eligible=[r for r in pool if r['both_full_correct']]
            good=[r for r in eligible if r['available']]
            key=f'{ds}|{method}|encoder|{fraction}'
            old=original[key]
            assert old['records']==len(good),(key,'denominator')
            if good:
                accuracy=statistics.mean(r['correct'] for r in good)
                random_accuracy=statistics.mean(r['random_accuracy'] for r in good)
                assert abs(accuracy-old['accuracy'])<1e-10,(key,'accuracy')
                assert abs(random_accuracy-old['random_accuracy'])<1e-10,(key,'random accuracy')
            panel=dict(method=method,fraction=fraction,all_patients=len(pool),eligible_patients=len(eligible),
                       available_patients=len(good),eligible_status=dict(Counter(r['search_status'] for r in eligible)),
                       fact_count=distribution(r['fact_count'] for r in good),group_count=distribution(r['group_count'] for r in good),
                       retained_group_fraction=distribution(r['group_count']/r['full_group_count'] for r in good if r['full_group_count']),
                       search_queries_all=distribution(r['search_queries'] for r in pool))
            if method=='anchors_group_mask_tree':
                verify=[r for r in good if r['verification_n']]
                panel['independent_mask_verification']=dict(patients=len(verify),
                    draws=sum(r['verification_n'] for r in verify),success=sum(r['verification_success'] for r in verify),
                    patient_precision=distribution(r['verification_success']/r['verification_n'] for r in verify),
                    patients_empirical_precision_below_095=sum(r['verification_success']/r['verification_n']<.95 for r in verify),
                    perturbation_coverage=distribution(r['perturbation_coverage'] for r in good))
            panels.append(panel)
        # This counter spans all searches together and cannot be charged to one selector.
        queryrows=[r for r in rs if r['method']=='frequency' and r['fraction']==.25]
        result['datasets'][ds]=dict(panels=panels,selection_point_replay='all 11 Representation panels match original summary',
            unique_tree_queries_combined_per_patient=distribution(r['unique_tree_queries'] for r in queryrows))
        print('VERIFIED',ds,'11 selection panels; diagnostics exported',flush=True)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=Path(str(a.out)+'_reports.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:tar.add(a.out/'diagnostics.json',arcname='diagnostics.json')
    print('Return report:',archive,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('artifacts/round1_extra_baselines_v1'))
    p.add_argument('--out',type=Path,default=Path('artifacts/round1_extra_baselines_diagnostics'))
    p.add_argument('--workers',type=int,default=32)
    main(p.parse_args())
