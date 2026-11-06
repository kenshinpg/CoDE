"""Post hoc continuous-support analysis of frozen natural and controlled scores."""
import json,itertools,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
METHODS=['fact_coverage','group_coverage','cohd_coverage','tree_positive_shap','tree_class_shap','cohd_absolute','lightgbm_absolute']
BASELINES=['group_coverage','cohd_absolute','lightgbm_absolute']
EPS=1e-10

def make_pairs(records,labels):
    buckets=defaultdict(dict);audit=Counter();q=[]
    for r in records:
        audit['input_records']+=1
        if not (r['mapping_available'] and r['target_supported'] and r['probe_available'] and r['both_full_correct']):continue
        audit['eligible_records']+=1;d=labels.index(r['diagnosis']);y=labels.index(r['reference_diagnosis']);p=r['cited_probabilities']
        v=dict(r,q=float(p[d]),qy=float(p[y]),margin=float(p[d])-max(float(x) for i,x in enumerate(p) if i!=d))
        q.append(v['q']);key=(r['patient_id'],r['diagnosis'],r['cited_fact_count']);tokens=tuple(sorted(r['cited_tokens']))
        if tokens in buckets[key]:
            old=buckets[key][tokens]
            if old['q']!=v['q'] or old['qy']!=v['qy'] or any(old['scores'].get(m)!=v['scores'].get(m) for m in METHODS):raise ValueError('Duplicate set has different scores')
            audit['duplicate_sets_removed']+=1
        else:buckets[key][tokens]=v
    pairs=[]
    for (pid,d,count),values in sorted(buckets.items()):
        for a,b in itertools.combinations(values.values(),2):
            finite=all(a['scores'].get(m) is not None and b['scores'].get(m) is not None and np.isfinite(a['scores'][m]) and np.isfinite(b['scores'][m]) for m in METHODS)
            pairs.append(dict(patient_id=pid,diagnosis=d,fact_count=count,common=finite,both_correct=not a['reference_failure'] and not b['reference_failure'],
                q=a['q']-b['q'],margin=a['margin']-b['margin'],qy=a['qy']-b['qy'],
                reference_recovery=int(not a['reference_failure'])-int(not b['reference_failure']),
                scores=[float(a['scores'][m])-float(b['scores'][m]) for m in METHODS] if finite else None))
    audit['distinct_pairs']=len(pairs);audit['common_pairs']=sum(r['common'] for r in pairs)
    return pairs,dict(counts=dict(audit),eligible_patients=len({k[0] for k in buckets}),q_quantiles=np.quantile(q,[0,.25,.5,.75,.9,.99,1]).tolist() if q else [],q_ge_099=sum(x>=.99 for x in q))

def summarize(pairs,endpoint,gap,subset,bootstrap=1000):
    possible=[r for r in pairs if subset=='all' or r['both_correct']]
    nontrivial=[r for r in possible if abs(r[endpoint])>EPS and abs(r[endpoint])>=gap]
    eligible=[r for r in nontrivial if r['common']];by=defaultdict(list);ties=Counter()
    for r in eligible:
        ds=np.asarray(r['scores']);dt=r[endpoint]
        values=np.where(np.abs(ds)<=EPS,.5,(ds*dt>0).astype(float))
        if values[0]!=.5:raise ValueError('Fact coverage must tie at equal count')
        for i,m in enumerate(METHODS):ties[m]+=int(abs(ds[i])<=EPS)
        # Orient reference readouts toward the higher target endpoint version.
        orient=float(np.sign(dt))
        by[r['patient_id']].append(np.r_[values,r['qy']*orient,r['reference_recovery']*orient,float(r['qy']*orient < -EPS)])
    pids=sorted(by);x=np.array([np.mean(by[pid],axis=0) for pid in pids]);rng=np.random.default_rng(26100644)
    boot=np.array([x[rng.integers(0,len(x),len(x))].mean(axis=0) for _ in range(bootstrap)]) if len(x)>1 and bootstrap else None
    def stat(i,j=None):
        vals=x[:,i] if j is None else x[:,i]-x[:,j]
        valsboot=(boot[:,i] if j is None else boot[:,i]-boot[:,j]) if boot is not None else None
        return dict(mean=float(vals.mean()) if len(x) else None,ci95=np.quantile(valsboot,[.025,.975]).tolist() if valsboot is not None else None)
    # Empty x must retain the column dimension for NA results.
    if not len(x):x=np.empty((0,len(METHODS)+3))
    metrics={m:dict(**stat(i),score_tied_pairs=ties[m]) for i,m in enumerate(METHODS)}
    contrasts={a+' minus '+b:stat(METHODS.index(a),METHODS.index(b)) for a in METHODS for b in BASELINES if a!=b}
    return dict(endpoint=endpoint,gap=gap,subset=subset,possible_pairs=len(possible),endpoint_eligible_pairs=len(nontrivial),common_pairs=len(eligible),patients=len(pids),
        methods=metrics,paired_differences=contrasts,reference_alignment=dict(q_y_toward_higher_endpoint=stat(len(METHODS)),recovery_toward_higher_endpoint=stat(len(METHODS)+1),fraction_higher_endpoint_lower_qy=stat(len(METHODS)+2)),
        patient_rows=[dict(patient_id=pid,pairs=len(by[pid]),values=x[i].tolist()) for i,pid in enumerate(pids)])

def main():
    out=ROOT/'docs/continuous_support_results_20261007';out.mkdir(exist_ok=False)
    files={k:ROOT/'docs'/v for k,v in [('natural','code10k_rq2_return_20261007'),('controlled','code10k_rq3_return_20261007')]}
    cfg=dict(stage='post hoc supplement after AP and feasibility counts; not original preregistration',primary_readout='q(d)',secondary_readout='margin',gaps=[EPS,.01],primary_probe='encoder',secondary_probe='lightgbm',bootstrap=1000,seed=26100644,methods=METHODS,baselines=BASELINES,subsets=['all','both_reference_correct'],pairing='same patient,target,actual fact count; identical sets deduplicated',quantity_scope='fact count matched; group count and semantic redundancy not matched',aggregation='pairs averaged within patient, patients equally; common finite seven-method pairs',endpoint_ties='exclude <=1e-10; gap .01 additionally applied',score_ties='<=1e-10 receives .5',source_sha256={k:hashlib.sha256((v/'scores.jsonl').read_bytes()).hexdigest() for k,v in files.items()})
    (out/'protocol.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    panels=[];audits={};patient_rows=[]
    for kind,p in files.items():
        labels=json.loads((p/'evaluation_lock.json').read_text())['labels'];records=[json.loads(l) for l in (p/'scores.jsonl').read_text().splitlines()]
        for probe in ['encoder','lightgbm']:
            for fraction in ([None] if kind=='natural' else [.25,.5,.75]):
                for stratum in (['correct','incorrect'] if kind=='natural' else ['correct']):
                    rs=[r for r in records if r['probe']==probe and (fraction is None or r['fraction']==fraction) and r['diagnosis_correct']==(stratum=='correct')]
                    key=f'{kind}|{probe}|{fraction}|{stratum}';pairs,audit=make_pairs(rs,labels);audits[key]=audit
                    for endpoint in ['q','margin']:
                        for gap in [EPS,.01]:
                            for subset in ['all','both_correct']:
                                r=summarize(pairs,endpoint,gap,subset);pidkey=f'{key}|{endpoint}|{gap}|{subset}'
                                patient_rows.extend(dict(panel=pidkey,**v) for v in r.pop('patient_rows'))
                                r.update(kind=kind,probe=probe,fraction=fraction,stratum=stratum,panel=pidkey,
                                    definition_identity=['lightgbm_absolute'] if probe=='lightgbm' and endpoint=='q' else [],
                                    self_source=['tree_positive_shap','tree_class_shap','lightgbm_absolute'] if probe=='lightgbm' else [],
                                    interpretation='Cross-model predictive support, not clinical sufficiency. Higher wrong-d support is not diagnostic improvement. Unadjusted intervals.')
                                panels.append(r)
                    print('Completed',key,flush=True)
    for name,data in [('summary.json',panels),('availability.json',audits),('patient_aggregates.json',patient_rows)]:
        (out/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(dict(status='complete',panels=len(panels),models_trained=False,api_calls=0,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}),indent=2),encoding='utf-8')
if __name__=='__main__':main()
