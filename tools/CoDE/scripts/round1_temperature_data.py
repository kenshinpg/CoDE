"""Verified development-only inputs; reuse caches, replay only missing DDX Encoder."""
from pathlib import Path
import json, hashlib
import numpy as np

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f: return [json.loads(s) for s in f if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def checked(p,h,audit):
    actual=sha(p)
    if actual!=h: raise ValueError('Frozen input changed: '+str(p))
    audit[str(p)]=actual

def validate_cache(p, ids, targets, expected_ids, expected_y, classes):
    p=np.asarray(p,float); ids=list(map(str,ids)); expected_ids=list(map(str,expected_ids))
    if ids!=expected_ids or len(set(ids))!=len(ids): raise ValueError('Development ID axis mismatch')
    if not np.array_equal(targets,expected_y): raise ValueError('Development label axis mismatch')
    if p.shape!=(len(ids),classes): raise ValueError('Probability shape mismatch')
    finite=np.isfinite(p).all(1)
    if not np.isnan(p[~finite]).all(): raise ValueError('Partial/nonfinite probability row')
    q=p[finite]
    if not len(q) or (q<0).any() or not np.allclose(q.sum(1),1,atol=1e-5): raise ValueError('Invalid development probabilities')
    return finite

def partition_check(dev_ids,test_ids,train_ids):
    d,t,r=map(set,(dev_ids,test_ids,train_ids))
    if len(d)!=len(dev_ids) or d&t or d&r or t&r: raise ValueError('Partition overlap/duplicate development ID')

def ddx_encoder(dev,labels,modelpath,dictionary,fractions,threads,private):
    import torch
    from code_eval.context_encoder import ContextEncoder
    from code_eval.tree_subset import Encoding,metrics
    torch.set_num_threads(threads)
    ck=torch.load(modelpath,map_location='cpu',weights_only=True)
    if ck['labels']!=labels: raise ValueError('Encoder label mismatch')
    model=ContextEncoder(len(ck['vocabulary']),len(labels),ck['hidden'])
    model.load_state_dict(ck['state']); model.eval(); enc=Encoding(dictionary); vocab=ck['vocabulary']
    result={}; original=read(modelpath.parent/'sparse_validation.json')
    for frac in fractions:
        sets=[enc.masked(r['tokens'],frac,f"261006|development|{r['input_group_id']}") for r in dev]
        valid=[i for i,ts in enumerate(sets) if all(t in vocab for t in ts)]
        p=np.full((len(dev),len(labels)),np.nan)
        with torch.no_grad():
            for start in range(0,len(valid),128):
                ix=valid[start:start+128]; x=torch.zeros((len(ix),max(1,max(len(sets[i]) for i in ix))),dtype=torch.long)
                for j,i in enumerate(ix):
                    if sets[i]: x[j,:len(sets[i])]=torch.tensor([vocab[t] for t in sets[i]])
                h=torch.tensor([[(float(dev[i]['context']['age'])-ck['age_mean'])/ck['age_std'],float(dev[i]['context']['sex']=='M')] for i in ix],dtype=torch.float32)
                p[ix]=model(x,h).softmax(1).numpy()
                print('DDX frozen Encoder development',frac,min(start+128,len(valid)),'/',len(valid),flush=True)
        y=np.array([labels.index(r['label']) for r in dev]); m=metrics(p[valid],y[valid]); old=original[str(frac)]
        if len(valid)!=old['available'] or abs(m['accuracy']-old['accuracy'])>1e-10 or abs(m['nll']-old['nll'])>2e-4:
            raise ValueError('DDX development replay differs from accepted diagnostics')
        np.savez_compressed(private/f'DDX_encoder_keep{frac}.npz',probabilities=p,targets=y,ids=np.array([r['input_group_id'] for r in dev]))
        result[frac]=p
    return result

def load_development(root,cfg,out,threads):
    fractions=[0,.25,.5,.75,1]; audit={}; all_data={}; profiles={}; private=out/'private'; private.mkdir()
    for ds in ['DDX','TRACE']:
        c=cfg['datasets'][ds]; models=root/c['models']; fm=read(models/'fitted_sources.json')
        if ds=='DDX':
            rqcfg=read(root/'configs/code10k_rq2_20261007.json')
            checked(models/'fitted_sources.json',rqcfg['fitted_manifest_sha256'],audit)
            dc=read(models/'data_contract.json'); labels=fm['labels']; path=models/'source/tree/fusion_selected_private.jsonl'
            checked(path,dc['development_records_sha256'],audit); dev=rows(path)
            splitpath=root/c['protocol']/'patient_split_PRIVATE.jsonl'; checked(splitpath,dc['split_sha256'],audit); split=rows(splitpath)
            ids=[r['input_group_id'] for r in dev]; test=[r['patient_id'] for r in split if r['partition']=='test']; train=[r['patient_id'] for r in split if r['partition']=='train']
            if set(ids)!={r['patient_id'] for r in split if r['partition']=='development'} or len(ids)!=1506: raise ValueError('DDX dev census')
            sm={r['patient_id']:r for r in split}
            if any(r['label']!=sm[r['input_group_id']]['disease'] or r['pattern_id']!=sm[r['input_group_id']]['pattern_id'] for r in dev): raise ValueError('DDX development record mismatch')
            if {r['pattern_id'] for r in split if r['partition']=='development'} & {r['pattern_id'] for r in split if r['partition']!='development'}: raise ValueError('Pattern overlap')
            tree=models/'source/tree'; tm=read(tree/'run_manifest.json')
            for rel in ['encoder/encoder.pt','source/tree/mask_augmented.txt']: checked(models/rel,fm['models'][rel],audit)
            dpath=models/'source/cohort/evidence_dictionary.json'; checked(dpath,dc['dictionary_sha256'],audit)
            origin=tree
        else:
            view=root/c['view']; vm=read(view/'manifest.json')
            for n in ['split_PRIVATE.jsonl','development_records_PRIVATE.jsonl','train_records_PRIVATE.jsonl']:
                checked(view/n,vm['output_sha256'][n],audit)
                if fm['input_sha256'][n]!=vm['output_sha256'][n]: raise ValueError('TRACE model/view mismatch')
            split=rows(view/'split_PRIVATE.jsonl'); dev=rows(view/'development_records_PRIVATE.jsonl'); sm={r['admission_id']:r for r in split}
            ids=[r['patient_id'] for r in dev]; test=[r['admission_id'] for r in split if r['partition']=='test']; train=[r['admission_id'] for r in split if r['partition']=='train']
            labels=sorted({r['label'] for r in rows(view/'train_records_PRIVATE.jsonl')})
            if len(ids)!=361 or set(ids)!={r['admission_id'] for r in split if r['partition']=='development'}: raise ValueError('TRACE dev census')
            if any(r['label']!=sm[r['patient_id']]['reference_class'] for r in dev): raise ValueError('TRACE dev labels')
            groups={}
            for r in split:
                groups.setdefault(r['subject_group'],set()).add(r['partition'])
            if any(len(v)>1 for v in groups.values()): raise ValueError('TRACE subject leakage')
            modelpaths={k:Path(fm['models'][k]['path']) for k in ['encoder','lightgbm']}
            for k,p in modelpaths.items(): checked(p,fm['models'][k]['sha256'],audit)
            first=modelpaths['encoder'].parent; origin=first/'development_predictions_PRIVATE'
            tm=read(first/'run_manifest.json'); summary=read(first/'development_summary.json')
            if summary['labels']!=labels: raise ValueError('TRACE labels differ')
        partition_check(ids,test,train)
        if (len(train),len(ids),len(test),len(labels))!=((6988,1506,1506,46) if ds=='DDX' else (1680,361,359,4)):
            raise ValueError('Frozen population/class census mismatch')
        y=np.array([labels.index(r['label']) for r in dev]); data={}
        for probe in ['encoder','lightgbm']:
            ps={}
            if ds=='DDX' and probe=='encoder':
                print('DDX Encoder probability cache was not saved by original fitter; replay frozen checkpoint on DEVELOPMENT only',flush=True)
                ps=ddx_encoder(dev,labels,models/'encoder/encoder.pt',read(dpath),fractions,threads,private)
            else:
                for frac in fractions:
                    # DDX fitter used integer 1, TRACE source used float 1.0 in its cache names.
                    suffix=str(float(frac)) if ds=='TRACE' and frac==1 else str(frac)
                    name=(f'mask_augmented_keep{suffix}.npz' if ds=='DDX' else f"{'encoder' if probe=='encoder' else 'tree_mask_augmented'}_keep{suffix}.npz")
                    path=origin/name
                    expected=tm['output_sha256'][name] if ds=='DDX' else tm['artifacts_sha256'][str(path.relative_to(first))]
                    checked(path,expected,audit)
                    with np.load(path,allow_pickle=False) as z:
                        p=z['probabilities'].copy()
                        if 'labels' in z and z['labels'].tolist()!=labels: raise ValueError('Cached class axis mismatch')
                        validate_cache(p,z['ids'],z['targets'] if 'targets' in z else y,ids,y,len(labels))
                    ps[frac]=p
                    print('REUSE',ds,probe,frac,flush=True)
            for frac,p in ps.items():
                ok=validate_cache(p,ids,y,ids,y,len(labels)); data[probe,frac]=(p[ok],y[ok])
            profiles[ds,probe]={'development_records':len(ids),'cache_policy':'frozen_cpu_development_replay' if ds=='DDX' and probe=='encoder' else 'original_development_probability_cache','note':'TRACE Encoder development cache uses archived inference math; test probabilities retain accepted strict-math cache. No test refit.' if ds=='TRACE' and probe=='encoder' else ''}
        all_data[ds]={'probabilities':data,'labels':labels,'test_ids':set(test),'dev_ids':set(ids)}
    return all_data,profiles,audit
