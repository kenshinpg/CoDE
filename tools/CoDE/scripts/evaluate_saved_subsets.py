"""RQ1 development cross-probe evaluation; reuses saved models and selected sets."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import tarfile
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepare_lantern_cohort import read,rows,sha,dump


def matched_groups(groups,target,rng):
    """Uniform draw among group subsets with exactly target observed facts.

    Integer subset-count dynamic program: O(number_groups * target), no
    exponential enumeration. Groups partition observed facts.
    """
    keys=sorted(groups);n=len(keys)
    ways=[[0]*(target+1) for _ in range(n+1)];ways[n][0]=1
    for i in range(n-1,-1,-1):
        size=len(groups[keys[i]])
        for k in range(target+1):
            ways[i][k]=ways[i+1][k]+(ways[i+1][k-size] if k>=size else 0)
    if not ways[0][target]:return None
    remaining=target;chosen=[]
    for i,g in enumerate(keys):
        size=len(groups[g]);skip=ways[i+1][remaining]
        take=ways[i+1][remaining-size] if remaining>=size else 0
        if rng.randrange(skip+take)>=skip:
            chosen.append(g);remaining-=size
    assert remaining==0
    return sorted(f for g in chosen for f in groups[g])


def run(a):
    import torch
    from code_eval.context_encoder import ContextEncoder
    torch.set_num_threads(a.threads)
    tree=a.source/'tree';cohort=a.source/'cohort';tm=read(tree/'run_manifest.json');pm=read(a.probe/'run_manifest.json')
    if tm['status']!='complete' or pm['status']!='complete':raise ValueError('Incomplete upstream')
    if sha(tree/'run_manifest.json')!=pm['source_tree_manifest_sha256']:raise ValueError('Probe/tree mismatch')
    for name in ['encoder.pt']:
        if sha(a.probe/name)!=pm['output_sha256'][name]:raise ValueError('Checkpoint mismatch')
    for name in ['subset_curves.jsonl']:
        if sha(tree/name)!=tm['output_sha256'][name]:raise ValueError('Subset source mismatch')
    for name in ['development_records.jsonl']:
        expected=[h for p,h in tm['source_sha256'].items() if Path(p).name==name]
        if expected!=[sha(cohort/name)]:raise ValueError('Record source mismatch')
    if sha(cohort/'development_outcomes.jsonl')!=pm['outcome_sidecar_sha256']:raise ValueError('Outcome sidecar drift')
    ck=torch.load(a.probe/'encoder.pt',map_location='cpu',weights_only=True)
    vocab=ck['vocabulary'];labels=ck['labels']
    model=ContextEncoder(len(vocab),len(labels),ck['hidden']);model.load_state_dict(ck['state']);model.eval()
    patients={r['patient_id']:r for r in rows(cohort/'development_records.jsonl')}
    truth={r['patient_id']:r for r in rows(cohort/'development_outcomes.jsonl')}
    queries={}; tasks=[]
    def query(pid,fs):
        key=(pid,tuple(sorted(fs)))
        if key not in queries:queries[key]=len(queries)
        return queries[key]
    full={pid:query(pid,[f['fact_id'] for f in r['facts']]) for pid,r in patients.items()}
    curves=list(rows(tree/'subset_curves.jsonl'))
    for r in curves:
        pid=r['patient_id'];selected=r['fact_ids'];factset={f['fact_id'] for f in patients[pid]['facts']}
        if not set(selected)<=factset or len(selected)!=r['fact_count']:raise ValueError('Invalid saved subset')
        controls=[]
        if r['selector']!='random':
            for repeat in range(a.repeats):
                rng=random.Random(f"{a.seed}|{pid}|{r['fraction']}|{len(selected)}|{repeat}")
                fs=matched_groups(patients[pid]['groups'],len(selected),rng)
                if fs is None:raise ValueError('Saved whole-group set should have feasible size')
                controls.append(query(pid,fs))
        tasks.append(dict(patient_id=pid,selector=r['selector'],fraction=r['fraction'],
            fact_count=len(selected),query=query(pid,selected),controls=controls))
    outputs=[None]*len(queries);items=list(queries)
    print('Unique independent-probe queries',len(items),flush=True)
    for start in range(0,len(items),256):
        valid=[]
        for offset,(pid,fs) in enumerate(items[start:start+256]):
            rec=patients[pid];tokens={f['fact_id']:f['provenance']['source_token'] for f in rec['facts']}
            ts=[tokens[f] for f in fs]
            if all(t in vocab for t in ts):valid.append((start+offset,rec,ts))
        if valid:
            x=torch.zeros((len(valid),max(1,max(len(ts) for _,_,ts in valid))),dtype=torch.long)
            for i,(_,rec,ts) in enumerate(valid):x[i,:len(ts)]=torch.tensor([vocab[t] for t in ts],dtype=torch.long)
            h=torch.tensor([[(float(r['demographics']['age'])-ck['age_mean'])/ck['age_std'],float(r['demographics']['sex']=='M')] for _,r,ts in valid],dtype=torch.float32)
            with torch.no_grad():ps=model(x,h).softmax(1).numpy()
            for (idx,rec,_),p in zip(valid,ps):outputs[idx]=dict(prediction=labels[int(p.argmax())],reference_support=float(p[labels.index(truth[rec['patient_id']]['reference_diagnosis'])]))
        if start%10240==0:print('Predicted',min(start+256,len(items)),'/',len(items),flush=True)
    results=[]
    for t in tasks:
        pid=t['patient_id'];p=outputs[t['query']];f=outputs[full[pid]];controls=[outputs[i] for i in t['controls']]
        label=truth[pid]['reference_diagnosis'];available=p is not None and f is not None and all(c is not None for c in controls)
        results.append(dict(patient_id=pid,selector=t['selector'],fraction=t['fraction'],fact_count=t['fact_count'],
            component=truth[pid]['component'],reference_diagnosis=label,available=available,
            full_correct=f['prediction']==label if f else None,
            correct=p['prediction']==label if p else None,
            reference_support=p['reference_support'] if p else None,
            random_matched_accuracy=float(np.mean([c['prediction']==label for c in controls])) if available and controls else None))
    with (a.out/'subset_probe_records.jsonl').open('w',encoding='utf8') as f:
        for r in results:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary=[]
    for component in ['all','coverage','challenge']:
        for condition in ['full_correct','all_available']:
            for method in sorted({r['selector'] for r in results}):
                for fraction in [.25,.5,.75,1]:
                    population=[r for r in results if r['selector']==method and r['fraction']==fraction and (component=='all' or r['component']==component)]
                    rs=[r for r in population if r['available'] and (condition=='all_available' or r['full_correct'])]
                    if not rs:continue
                    acc=np.array([r['correct'] for r in rs],dtype=float)
                    differences=np.array([r['correct']-r['random_matched_accuracy'] for r in rs]) if method!='random' else None
                    rng=np.random.default_rng(a.seed);bs=[];ds=[]
                    for _ in range(500):
                        ix=rng.integers(0,len(rs),len(rs));bs.append(float(acc[ix].mean()))
                        if differences is not None:ds.append(float(differences[ix].mean()))
                    summary.append(dict(component=component,condition=condition,selector=method,fraction=fraction,
                        patients=len(rs),unavailable=sum(not r['available'] for r in population),accuracy=float(acc.mean()),
                        ci95=np.quantile(bs,[.025,.975]).tolist(),mean_facts=float(np.mean([r['fact_count'] for r in rs])),
                        gain_over_fact_matched_random=float(differences.mean()) if differences is not None else None,
                        gain_ci95=np.quantile(ds,[.025,.975]).tolist() if ds else None))
    dump(a.out/'rq1_development.json',summary)
    return dict(status='complete',scope='LightGBM selectors against frozen Encoder; development only',
        repeats=a.repeats,seed=a.seed,api_calls=0,models_trained=0,evaluation_scoring=False,
        matching='exact observed fact count; whole dependency groups; uniform feasible subset sampling',
        interpretation='not clinical necessity, no selection stability or four-source comparison yet',
        upstream_sha256={'tree':sha(tree/'run_manifest.json'),'probe':sha(a.probe/'run_manifest.json')},code_sha256=sha(Path(__file__)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('artifacts/lantern_development_v1'))
    p.add_argument('--probe',type=Path,default=Path('artifacts/encoder_probe_v1'))
    p.add_argument('--out',type=Path,default=Path('artifacts/rq1_cross_probe_v1'))
    p.add_argument('--repeats',type=int,default=10);p.add_argument('--threads',type=int,default=8);p.add_argument('--seed',type=int,default=261006)
    a=p.parse_args()
    if a.repeats<1:p.error('repeats must be positive')
    a.out.mkdir(parents=True,exist_ok=False);report={}
    try:report=run(a)
    except Exception:report=dict(status='failed',error=traceback.format_exc());raise
    finally:
        report['output_sha256']={f.name:sha(f) for f in a.out.iterdir() if f.is_file()}
        dump(a.out/'run_manifest.json',report)
        archive=a.out.with_name(a.out.name+'_reports.tar.gz')
        with tarfile.open(archive,'w:gz') as t:
            for f in a.out.iterdir():t.add(f,arcname=f.name)
        print('Return report:',archive,flush=True)


if __name__=='__main__':main()
