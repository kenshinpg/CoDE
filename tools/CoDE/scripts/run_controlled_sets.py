"""Fixed-budget random evidence sets; development-only diagnostic proxy study."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import tarfile
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepare_lantern_cohort import read,rows,sha,dump
from evaluate_saved_subsets import matched_groups
from run_coverage_controls import coverage


def sampled_sets(groups, seed, pid, fraction, repeats):
    # Budget anchored to a label- and source-independent random group ordering.
    order=sorted(groups);random.Random(f'{seed}|budget|{pid}|{fraction}').shuffle(order)
    count=sum(len(groups[g]) for g in order[:math.ceil(len(order)*fraction)])
    unique=set()
    for i in range(repeats):
        fs=matched_groups(groups,count,random.Random(f'{seed}|sets|{pid}|{fraction}|{i}'))
        if fs is None:raise ValueError('Feasible anchor count could not be sampled')
        unique.add(tuple(fs))
    return count,sorted(unique)


def run(a):
    import numpy as np
    import torch
    import lightgbm as lgb
    from code_eval.context_encoder import ContextEncoder
    from code_eval.tree_subset import Encoding,checked_contributions
    from code_eval.proxy_metrics import summarize
    torch.set_num_threads(a.threads)
    tree=a.source/'tree';cohort=a.source/'cohort'
    tm=read(tree/'run_manifest.json');pm=read(a.probe/'run_manifest.json');cm=read(a.controls/'run_manifest.json')
    if any(m['status']!='complete' for m in [tm,pm,cm]):raise ValueError('Incomplete upstream')
    if pm['source_tree_manifest_sha256']!=sha(tree/'run_manifest.json') or cm['source_tree_sha256']!=sha(tree/'run_manifest.json') or cm['source_probe_sha256']!=sha(a.probe/'run_manifest.json'):
        raise ValueError('Upstream mismatch')
    for directory,m,names in [(tree,tm,['mask_augmented.txt','patient_attributions.jsonl','encoding.json']),
                              (a.probe,pm,['encoder.pt']), (a.controls,cm,['class_shap.json'])]:
        for name in names:
            if sha(directory/name)!=m['output_sha256'][name]:raise ValueError('Artifact drift: '+name)
    for name in ['development_records.jsonl','evidence_dictionary.json']:
        if [h for p,h in tm['source_sha256'].items() if Path(p).name==name]!=[sha(cohort/name)]:raise ValueError('Cohort drift')
    if sha(cohort/'development_outcomes.jsonl')!=pm['outcome_sidecar_sha256']:raise ValueError('Outcome drift')
    encoding=Encoding(read(cohort/'evidence_dictionary.json'))
    if encoding.export()!=read(tree/'encoding.json'):raise ValueError('Encoding drift')
    labels=tm['labels']; classdata=read(a.controls/'class_shap.json')
    if classdata['labels']!=labels or classdata['columns']!=encoding.columns:raise ValueError('Class SHAP axes drift')
    means=np.asarray(classdata['mean_absolute']);model=lgb.Booster(model_file=str(tree/'mask_augmented.txt'))
    ck=torch.load(a.probe/'encoder.pt',map_location='cpu',weights_only=True)
    if ck['labels']!=labels:raise ValueError('Probe labels drift')
    vocab=ck['vocabulary'];probe=ContextEncoder(len(vocab),len(labels),ck['hidden']);probe.load_state_dict(ck['state']);probe.eval()
    def predict(inputs,context):
        result=[None]*len(inputs);valid=[(i,ts) for i,ts in enumerate(inputs) if all(t in vocab for t in ts)]
        if valid:
            x=torch.zeros((len(valid),max(1,max(len(ts) for _,ts in valid))),dtype=torch.long)
            for n,(_,ts) in enumerate(valid):x[n,:len(ts)]=torch.tensor([vocab[t] for t in ts],dtype=torch.long)
            h=torch.tensor([[(float(context['age'])-ck['age_mean'])/ck['age_std'],float(context['sex']=='M')]]*len(valid),dtype=torch.float32)
            with torch.no_grad():ps=probe(x,h).softmax(1).numpy()
            if not np.isfinite(ps).all():raise ValueError('Nonfinite probe')
            for (i,_),p in zip(valid,ps):result[i]=labels[int(p.argmax())]
        return result
    patients={}
    for r in rows(cohort/'development_records.jsonl'):
        pid=r['patient_id']
        if pid in patients and any(r[k]!=patients[pid][k] for k in ['facts','groups','demographics','pattern_id']):raise ValueError('Patient drift')
        patients[pid]=r
    outcomes={r['patient_id']:r for r in rows(cohort/'development_outcomes.jsonl')}
    attrs={r['patient_id']:r for r in rows(tree/'patient_attributions.jsonl')}
    if set(attrs)!=set(patients) or set(outcomes)!=set(patients):raise ValueError('Incomplete patient join')
    output=[];audits=[]
    print('Stage 1: frozen random sets, source scores and independent probe',flush=True)
    for n,(pid,r) in enumerate(patients.items(),1):
        tokens={f['fact_id']:f['provenance']['source_token'] for f in r['facts']};groups=r['groups'];ctx=r['demographics']
        full_tokens=[tokens[f] for f in sorted(tokens)];x=encoding.encode(full_tokens,ctx)[None,:]
        full=model.predict(x,num_threads=1)[0];d=int(full.argmax());diagnosis=labels[d]
        if diagnosis!=attrs[pid]['source_target']:raise ValueError('Source target drift')
        v=checked_contributions(model,x,len(labels))[0,d]
        cols={g:sorted({encoding.token(tokens[f])[0] for f in fs}) for g,fs in groups.items()}
        weights=dict(global_gain=attrs[pid]['group_weights']['global_gain'],
            local_positive_group_shap={g:max(0.,float(v[cs].sum())) for g,cs in cols.items()},
            local_absolute_shap={g:float(np.abs(v[cs]).sum()) for g,cs in cols.items()},
            class_mean_absolute_shap={g:float(means[d,cs].sum()) for g,cs in cols.items()},
            positive_leave_group_out={g:max(0.,e) for g,e in attrs[pid]['signed_leave_group_out'].items()})
        tasks=[]
        for fraction in [.25,.5,.75]:
            count,sets=sampled_sets(groups,a.seed,pid,fraction,a.repeats)
            audits.append(dict(patient_id=pid,fraction=fraction,fact_count=count,requested=a.repeats,unique=len(sets)))
            tasks.extend((fraction,count,fs) for fs in sets)
        inputs=[[tokens[f] for f in fs] for _,_,fs in tasks]
        ps=model.predict(np.stack([encoding.encode(ts,ctx) for ts in inputs]),num_threads=1)
        predictions=predict([full_tokens]+inputs,ctx);truth=outcomes[pid]['reference_diagnosis']
        for j,(fraction,count,fs) in enumerate(tasks):
            cited=set(fs);scores={k:coverage(w,groups,cited) for k,w in weights.items()}
            scores.update(fact_coverage=count/len(tokens),group_coverage=sum(len(set(gs)&cited)/len(gs) for gs in groups.values())/len(groups),
                          absolute_support=float(ps[j,d]),retained_support=float(ps[j,d]-full[d]))
            output.append(dict(explanation_id=f'{pid}|{fraction}|{j}',patient_id=pid,pattern_id=r['pattern_id'],
                component=outcomes[pid]['component'],reference_diagnosis=truth,diagnosis=diagnosis,diagnosis_correct=diagnosis==truth,
                fraction=fraction,cited_fact_count=count,fact_ids=list(fs),probe_available=predictions[0] is not None and predictions[j+1] is not None,
                full_correct=predictions[0]==truth,failure=predictions[j+1]!=truth,scores=scores,
                record_type='CONTROLLED_STRUCTURED_SUBSET_NOT_LLM_EXPLANATION'))
        if n%25==0:print('Patients',n,'/',len(patients),'sets',len(output),flush=True)
    for name,data in [('controlled_proxy_records.jsonl',output),('sampling_audit.jsonl',audits)]:
        with (a.out/name).open('w',encoding='utf8') as f:
            for r in data:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    methods=list(output[0]['scores']);panels={}
    print('Stage 2: within-budget paired metrics',flush=True)
    for fraction in [.25,.5,.75]:
        for target in ['correct_proposal','all_proposals']:
            rs=[r for r in output if r['fraction']==fraction and (target=='all_proposals' or r['diagnosis_correct'])]
            panels[f'{fraction}|{target}']=summarize(rs,methods,a.bootstrap,a.seed)
            print(fraction,target,'pairs',panels[f'{fraction}|{target}']['discordant_pairs'],flush=True)
    dump(a.out/'controlled_comparison.json',panels)
    return dict(status='complete',scope='development controlled structured subsets; no edited explanations',patients=len(patients),subsets=len(output),
        models_trained=0,api_calls=0,evaluation_scoring=False,repeats=a.repeats,bootstrap=a.bootstrap,seed=a.seed,
        upstream_sha256={'tree':sha(tree/'run_manifest.json'),'probe':sha(a.probe/'run_manifest.json'),'controls':sha(a.controls/'run_manifest.json')},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/evaluate_saved_subsets.py',ROOT/'scripts/run_coverage_controls.py',ROOT/'src/code_eval/proxy_metrics.py']},
        notes=['No prediction-conditioned subset sampling','Deduplicated within patient/budget','Class target is full LightGBM prediction, not gold label',
               'Fact coverage ties within matched pairs by design','Absolute and retained support tie in pair ordering by construction'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('artifacts/lantern_development_v1'))
    p.add_argument('--probe',type=Path,default=Path('artifacts/encoder_probe_v1'))
    p.add_argument('--controls',type=Path,default=Path('artifacts/coverage_controls_v1'))
    p.add_argument('--out',type=Path,default=Path('artifacts/controlled_sets_v1'))
    p.add_argument('--repeats',type=int,default=10);p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--seed',type=int,default=261006)
    a=p.parse_args()
    if min(a.repeats,a.bootstrap,a.threads)<1:p.error('Positive counts required')
    a.out.mkdir(parents=True,exist_ok=False);result={};started=time.time()
    try:result=run(a)
    except Exception:result=dict(status='failed',error=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.time()-started;result['output_sha256']={f.name:sha(f) for f in a.out.iterdir() if f.is_file()}
        dump(a.out/'run_manifest.json',result)
        archive=a.out.with_name(a.out.name+'_reports.tar.gz')
        with tarfile.open(archive,'w:gz') as t:
            for f in a.out.iterdir():t.add(f,arcname=f.name)
        print('Return report:',archive,flush=True)


if __name__=='__main__':main()
