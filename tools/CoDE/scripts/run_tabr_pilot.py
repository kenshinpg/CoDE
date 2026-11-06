"""C550-compatible TabR-S / matched no-retrieval development feasibility pilot."""
import argparse, hashlib, json, os, subprocess, sys, tarfile, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def worker(a):
    import numpy as np
    import torch
    from code_eval.tabr_adapter import DenseEncoding, make_model
    from code_eval.case_retrieval import CaseRetrieval, pattern
    from code_eval.tree_subset import metrics
    torch.set_num_threads(a.threads)
    cfg=read(a.config); seed=cfg['seed']; torch.manual_seed(seed)
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('Existing torch does not expose a usable GPU; do not install NVIDIA wheels over C550 torch')
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    from smoke_tabr import smoke
    checks=smoke(device)
    dump(a.out/'mechanism_smoke.json',checks)
    # Reset after smoke so both conditions have identical initialization.
    torch.manual_seed(seed)
    tree=a.source/'tree'; cohort=a.source/'cohort'; tm=read(tree/'run_manifest.json')
    if tm['status']!='complete': raise ValueError('Incomplete tree source')
    names=['teacher_fit_selected_private.jsonl','fusion_selected_private.jsonl']
    for name in names:
        if sha(tree/name)!=tm['output_sha256'][name]: raise ValueError('Fit/development artifact hash mismatch')
    for name in ['evidence_dictionary.json','excluded_fit_patterns.txt']:
        expected=[v for k,v in tm['source_sha256'].items() if Path(k).name==name]
        if len(expected)!=1 or sha(cohort/name)!=expected[0]: raise ValueError('Cohort source hash mismatch: '+name)
    fit,dev=[rows(tree/n) for n in names]; labels=tm['labels']
    forbidden=set((cohort/'excluded_fit_patterns.txt').read_text().splitlines())
    if any(pattern(r['tokens'])!=r['pattern_id'] or r['pattern_id'] in forbidden for r in fit+dev): raise ValueError('Pattern or isolation violation')
    if {r['pattern_id'] for r in fit}&{r['pattern_id'] for r in dev}: raise ValueError('Fit/development overlap')
    if {r['input_group_id'] for r in fit}&{r['input_group_id'] for r in dev}: raise ValueError('Identity overlap')
    enc=DenseEncoding(read(cohort/'evidence_dictionary.json'),fit)
    cases=CaseRetrieval(fit,labels,k=5)
    if cases.conflicting_patterns: raise ValueError('Conflicting bank patterns require explicit protocol decision')
    pats={p:i for i,p in enumerate(sorted({r['pattern_id'] for r in fit+dev}))}
    def tensor(records, inputs=None):
        return torch.tensor(np.stack([enc.encode(ts,r['context']) for r,ts in zip(records,inputs if inputs is not None else [r['tokens'] for r in records])]),device=device)
    masked=[enc.base.masked(r['tokens'],[0,.25,.5,.75][i%4],f"{tm['seed']}|fit|{r['input_group_id']}") for i,r in enumerate(fit)]
    x=torch.cat([tensor(fit),tensor(fit,masked)])
    y=torch.tensor([labels.index(r['label']) for r in fit]*2,device=device)
    p=torch.tensor([pats[r['pattern_id']] for r in fit]*2,device=device)
    bx=tensor(cases.rows); by=torch.tensor([labels.index(r['label']) for r in cases.rows],device=device)
    bp=torch.tensor([pats[r['pattern_id']] for r in cases.rows],device=device)
    model=make_model(enc.width,len(labels),cfg['model']).to(device)
    use=a.condition=='tabr'
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'])
    epochs=a.epochs or cfg['epochs']; batch=a.batch_size or cfg['batch_size']
    def predict(rs,ts=None,**kw):
        inputs=tensor(rs,ts); pp=torch.tensor([pats[r['pattern_id']] for r in rs],device=device)
        out=[]
        with torch.no_grad():
            for j in range(0,len(rs),batch):
                # Explicit flags remain constant across batches.
                flags=dict(use_retrieval=use); flags.update(kw)
                out.append(model(inputs[j:j+batch],bx,by,pp[j:j+batch],bp,cfg['context_size'],**flags).softmax(1).cpu().numpy())
        return np.concatenate(out)
    start=time.time(); history=[]
    print(a.condition,'fit',len(fit),'bank',len(cases.rows),'width',enc.width,'device',device,flush=True)
    for epoch in range(epochs):
        model.train(); order=np.random.default_rng(seed+epoch).permutation(len(y)); total=0
        for j in range(0,len(y),batch):
            ix=torch.tensor(order[j:j+batch],device=device); optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x[ix],bx,by,p[ix],bp,cfg['context_size'],use_retrieval=use),y[ix])
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward(); optimizer.step(); total+=float(loss.detach())*len(ix)
            if (j//batch)%50==0: print(a.condition,'epoch',epoch+1,'batch',j//batch,'loss',float(loss.detach()),flush=True)
        model.eval(); val=predict(dev)
        h=dict(epoch=epoch+1,loss=total/len(y),validation=metrics(val,[labels.index(r['label']) for r in dev]),elapsed_seconds=time.time()-start)
        history.append(h);dump(a.out/'history.json',history);print(a.condition,h,flush=True)
        torch.save(dict(model=model.state_dict(),config=cfg,epoch=epoch+1,encoding=dict(tokens=enc.tokens,features=enc.features,age_mean=enc.mean,age_std=enc.std),labels=labels),a.out/'checkpoint.pt')
    sparse={}; records=[]
    for frac in [0,.25,.5,.75,1]:
        ts=[enc.base.masked(r['tokens'],frac,f"{tm['seed']}|development|{r['input_group_id']}") for r in dev]
        probs=predict(dev,ts); sparse[str(frac)]=metrics(probs,[labels.index(r['label']) for r in dev])
        for r,pr in zip(dev,probs): records.append(dict(input_group_id=r['input_group_id'],fraction=frac,probabilities=pr.tolist()))
    dump(a.out/'sparse_validation.json',sparse)
    with (a.out/'diagnostic_predictions.jsonl').open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    selected=sorted(dev,key=lambda r:hashlib.sha256(f"{seed}|{r['input_group_id']}".encode()).hexdigest())[:100]
    lofo=[]
    for i,r in enumerate(selected):
        groups=enc.base.groups(r['tokens']); keys=sorted(groups)
        ts=[r['tokens']]+[sorted(set(r['tokens'])-set(groups[g])) for g in keys]
        ps=predict([r]*len(ts),ts); target=int(ps[0].argmax()); changes=ps[0,target]-ps[1:,target]
        old=[cases.query(t,r['context'],excluded_pattern=r['pattern_id'],excluded_identity=r['input_group_id']) for t in ts]
        old_ok=all(z['available'] for z in old)
        old_changes=None
        if old_ok:
            op=np.array([z['probabilities'] for z in old]); od=int(op[0].argmax());old_changes=(op[0,od]-op[1:,od]).tolist()
        lofo.append(dict(input_group_id=r['input_group_id'],target=labels[target],groups=keys,full_support=float(ps[0,target]),effects=changes.tolist(),positive_mass=float(np.maximum(changes,0).sum()),nonzero_1e_6=int((np.abs(changes)>1e-6).sum()),nonzero_0_01=int((np.abs(changes)>=.01).sum()),simple_available=old_ok,simple_own_target_effects=old_changes))
        if i%10==0:print(a.condition,'LOFO',i+1,'/100',flush=True)
    dump(a.out/'lofo_feasibility.json',lofo)
    diagnostics={}
    if use:
        original=predict(selected); off=predict(selected,use_retrieval=False); no_t=predict(selected,use_difference=False)
        saved=by.clone(); by=by[torch.randperm(len(by),device=device)]; shuffled=predict(selected); by=saved
        for name,pr in [('retrieval_off',off),('difference_off',no_t),('bank_labels_shuffled',shuffled)]:
            diagnostics[name]=dict(mean_absolute_probability_change=float(np.abs(pr-original).mean()),prediction_flip_rate=float((pr.argmax(1)!=original.argmax(1)).mean()))
    dump(a.out/'branch_diagnostics.json',diagnostics)
    return dict(status='complete',condition=a.condition,fit=len(fit),bank=cases.audit,diagnostic_dev=len(dev),epochs=epochs,batch_size=batch,seed=seed,torch=torch.__version__,device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                selection='fixed final epoch; no held-out explanation selection',scope='development feasibility, not RQ1/RQ2/RQ3 final results',elapsed_seconds=time.time()-start,
                input_sha256={str(q):sha(q) for q in [a.config,tree/names[0],tree/names[1],cohort/'evidence_dictionary.json']})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('artifacts/lantern_development_v1'))
    p.add_argument('--out',type=Path,default=Path('artifacts/tabr_pilot_v1'))
    p.add_argument('--config',type=Path,default=ROOT/'configs/tabr_pilot_20261007.json')
    p.add_argument('--gpus',default='0,1');p.add_argument('--device',default='cuda')
    p.add_argument('--threads',type=int,default=8);p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int)
    p.add_argument('--condition',choices=['tabr','no_retrieval']);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False); state={}; start=time.time()
    try:
        if a.condition: state=worker(a)
        else:
            ids=a.gpus.split(',')
            if len(ids)<2:raise ValueError('Provide two GPU IDs for matched concurrent conditions')
            processes=[]
            for condition,gpu in zip(['tabr','no_retrieval'],ids):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
                cmd=[sys.executable,'-u',__file__,'--source',str(a.source),'--out',str(a.out/condition),'--config',str(a.config),'--condition',condition,'--device',a.device,'--threads',str(a.threads)]
                for key in ['epochs','batch_size']:
                    if getattr(a,key) is not None:cmd+=['--'+key.replace('_','-'),str(getattr(a,key))]
                log=(a.out/(condition+'.log')).open('w');processes.append((condition,subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
                print('START',condition,'GPU',gpu,flush=True)
            errors=[]
            for name,proc,log in processes:
                code=proc.wait(); log.close();print('DONE',name,'exit',code,flush=True)
                if code: errors.append(name)
            if errors:raise RuntimeError('Conditions failed: '+','.join(errors)+'; return report contains logs')
            state=dict(status='complete',scope='mechanism and development feasibility only; natural explanation evaluation not run',api_calls=0)
    except Exception:
        state=dict(status='failed',error=traceback.format_exc());raise
    finally:
        state['wall_seconds']=time.time()-start
        state['code_sha256']={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'scripts/smoke_tabr.py',ROOT/'src/code_eval/tabr_adapter.py',ROOT/'src/code_eval/tabr_official_core.py']}
        dump(a.out/'run_manifest.json',state)
        archive=a.out.with_name(a.out.name+'_reports.tar.gz')
        with tarfile.open(archive,'w:gz') as t:
            for f in a.out.rglob('*'):
                if f.is_file() and f.suffix in ('.json','.jsonl','.log'):t.add(f,arcname=str(f.relative_to(a.out)))
        print('Return report:',archive,flush=True)
if __name__=='__main__':main()
