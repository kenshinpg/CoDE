"""Refit all sources on the accepted 10k split; development checks only.

Reuses verified model implementations, never old checkpoints. The materialized
tree/cohort layout is an explicit compatibility interface for existing workers.
"""
import argparse, json, os, shutil, subprocess, sys, tarfile, time, traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepare_code10k import rows,sha,dump,unique
SPLIT='b18d834dd7032bc4e6571b541c421d7b045a390662b28ed82cebecd311c776c8'
LEDGER='cdebeccc408a9397e545680b57a71e63e70ad1de2a47497219bbbc2241482b5b'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def emit(p,rs):
    with p.open('w',encoding='utf-8') as f:
        for r in rs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def structural_record(p,s):
    from code_eval.tree_subset import pattern
    if p['patient_id']!=s['patient_id']:raise ValueError('Patient identity mismatch')
    facts=[f for f in p['facts'] if f.get('visible',True) and f['coverage_scope']=='symptoms_antecedents']
    tokens=sorted({f['provenance']['source_token'] for f in facts})
    if not tokens or pattern(tokens)!=s['pattern_id'] or p['pattern_id']!=s['pattern_id']:raise ValueError('Pattern changed')
    ctx={k:p['demographics'][k] for k in ['age','sex']}
    return dict(input_group_id=p['patient_id'],patient_id=p['patient_id'],pattern_id=s['pattern_id'],
                tokens=tokens,context=ctx,label=s['disease'],component=s['component'])

def materialize(a):
    from code_eval.tree_subset import Encoding
    report=read(a.protocol/'cohort_report.json')
    if sha(a.protocol/'patient_split_PRIVATE.jsonl')!=SPLIT or sha(a.protocol/'explanation_ledger_PRIVATE.jsonl')!=LEDGER:raise ValueError('Accepted split/ledger changed')
    # Locate by basename only after requiring exactly one frozen source entry.
    for name in ['patients_public.jsonl','evidence_dictionary.json']:
        p=a.sample/'data'/name
        expected=[v for k,v in report['input_sha256'].items() if Path(k).name==name]
        if expected!=[sha(p)]:raise ValueError('Raw source drift: '+name)
    split=unique(a.protocol/'patient_split_PRIVATE.jsonl','patient_id')
    patients=unique(a.sample/'data/patients_public.jsonl','patient_id')
    if set(split)!=set(patients) or len(split)!=10000:raise ValueError('Fixed patient universe changed')
    cohort=a.out/'source/cohort';tree=a.out/'source/tree'
    cohort.mkdir(parents=True);tree.mkdir(parents=True)
    shutil.copy2(a.sample/'data/evidence_dictionary.json',cohort/'evidence_dictionary.json')
    enc=Encoding(read(cohort/'evidence_dictionary.json'))
    parts={k:[] for k in ['train','development','test']}
    for pid,s in sorted(split.items()):
        r=structural_record(patients[pid],s);enc.encode(r['tokens'],r['context'])
        parts[s['partition']].append(r)
    if {k:len(v) for k,v in parts.items()}!={'train':6988,'development':1506,'test':1506}:raise ValueError('Partition counts changed')
    patterns={k:{r['pattern_id'] for r in v} for k,v in parts.items()}
    for x,y in [('train','development'),('train','test'),('development','test')]:
        if patterns[x]&patterns[y]:raise ValueError('Pattern overlap')
    labels=sorted({r['label'] for r in parts['train']})
    if len(labels)!=46 or any(r['label'] not in labels for r in parts['development']+parts['test']):raise ValueError('Label support changed')
    emit(tree/'teacher_fit_selected_private.jsonl',parts['train'])
    emit(tree/'fusion_selected_private.jsonl',parts['development'])
    # Test values are not passed to workers. Only pattern hashes are retained as exclusions.
    (cohort/'excluded_fit_patterns.txt').write_text('\n'.join(sorted(patterns['test']))+'\n',encoding='utf-8')
    dump(tree/'encoding.json',enc.export())
    audit=dict(status='complete',split_sha256=SPLIT,ledger_sha256=LEDGER,labels=labels,
        label_policy='46 train-supported diagnoses for every source; unsupported LLM proposals stay in the evaluation ledger and are explicitly unavailable, not assigned zero support',
        fit_patients=len(parts['train']),development_patients=len(parts['development']),test_patients=len(parts['test']),
        mapping_used_for_training_selection=False,cross_partition_patterns=0,
        dictionary_sha256=sha(cohort/'evidence_dictionary.json'),
        by_disease={k:dict(Counter(r['label'] for r in v)) for k,v in parts.items()},
        training_records_sha256=sha(tree/'teacher_fit_selected_private.jsonl'),
        development_records_sha256=sha(tree/'fusion_selected_private.jsonl'),
        test_features_exported_to_workers=False,legacy_checkpoint_loaded=False)
    dump(a.out/'data_contract.json',audit)
    return parts['train'],parts['development'],labels,enc

def fit_tree(a,train,dev,labels,enc):
    from run_tree_subset_pilot import fit_models
    tree=a.out/'source/tree';cohort=a.out/'source/cohort'
    _,params=fit_models(enc,labels,{'teacher_fit':train,'fusion':dev},SimpleNamespace(seed=261006,threads=a.threads,rounds=120,out=tree))
    # Existing workers consume this manifest, pointing exclusively to this run.
    manifest=dict(status='complete',seed=261006,labels=labels,rounds=120,params=params,split_sha256=SPLIT,
       source_sha256={str(cohort/n):sha(cohort/n) for n in ['evidence_dictionary.json','excluded_fit_patterns.txt']},
       output_sha256={f.name:sha(f) for f in tree.iterdir() if f.is_file()})
    dump(tree/'run_manifest.json',manifest)

def fit_encoder(a,train,dev,labels,enc):
    import numpy as np
    import torch
    from code_eval.context_encoder import ContextEncoder,smoke
    from code_eval.tree_subset import metrics
    out=a.out/'encoder';out.mkdir();torch.set_num_threads(a.threads)
    dump(out/'smoke.json',smoke());torch.manual_seed(261006);torch.use_deterministic_algorithms(True)
    vocab={t:i+1 for i,t in enumerate(sorted({t for r in train for t in r['tokens']}))}
    ages=np.array([r['context']['age'] for r in train],dtype=float);mean=float(ages.mean());std=max(float(ages.std()),1.)
    def tensors(inputs):
        x=torch.zeros((len(inputs),max(1,max(len(t) for t,c in inputs))),dtype=torch.long)
        for i,(ts,c) in enumerate(inputs):x[i,:len(ts)]=torch.tensor([vocab[t] for t in ts],dtype=torch.long)
        h=torch.tensor([[(float(c['age'])-mean)/std,float(c['sex']=='M')] for ts,c in inputs],dtype=torch.float32)
        return x,h
    inputs=[(r['tokens'],r['context']) for r in train]+[(enc.masked(r['tokens'],[0,.25,.5,.75][i%4],f"261006|fit|{r['input_group_id']}"),r['context']) for i,r in enumerate(train)]
    x,h=tensors(inputs);y=torch.tensor([labels.index(r['label']) for r in train]*2)
    model=ContextEncoder(len(vocab),len(labels),128);opt=torch.optim.Adam(model.parameters(),lr=.001);history=[]
    for epoch in range(30):
        model.train();total=0.;order=np.random.default_rng(261006+epoch).permutation(len(y))
        for start in range(0,len(y),256):
            ix=torch.tensor(order[start:start+256]);opt.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(x[ix],h[ix]),y[ix])
            if not torch.isfinite(loss):raise ValueError('Nonfinite Encoder loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=float(loss.detach())*len(ix)
        history.append(dict(epoch=epoch+1,loss=total/len(y)));dump(out/'history.json',history);print('Encoder',history[-1],flush=True)
    model.eval();torch.save(dict(state=model.state_dict(),vocabulary=vocab,labels=labels,age_mean=mean,age_std=std,hidden=128),out/'encoder.pt')
    checks={}
    for frac in [0,.25,.5,.75,1]:
        ins=[(enc.masked(r['tokens'],frac,f"261006|development|{r['input_group_id']}"),r['context']) for r in dev]
        ids=[i for i,(ts,c) in enumerate(ins) if all(t in vocab for t in ts)]
        probs=[]
        with torch.no_grad():
            for k in range(0,len(ids),256):
                bx,bh=tensors([ins[i] for i in ids[k:k+256]]);probs.extend(model(bx,bh).softmax(1).numpy())
        checks[str(frac)]=dict(available=len(ids),unseen_token_unavailable=len(dev)-len(ids),**(metrics(probs,[labels.index(dev[i]['label']) for i in ids]) if ids else {}))
    dump(out/'sparse_validation.json',checks)
    dump(out/'run_manifest.json',dict(status='complete',split_sha256=SPLIT,fit_patients=len(train),seed=261006,epochs=30,hidden=128,batch_size=256,learning_rate=.001,precision='float32',device='cpu',checkpoint='fixed final epoch',source_tree_manifest_sha256=sha(a.out/'source/tree/run_manifest.json'),output_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file()}))

def run(a):
    # Fail before expensive work if the existing server runtime is incomplete.
    import numpy,lightgbm,torch
    ids=a.gpus.split(',')
    if len(ids)<2 or len(set(ids))!=len(ids):raise ValueError('Two distinct GPU IDs required')
    if not torch.cuda.is_available():raise RuntimeError('Use existing C550 torch environment; do not overwrite it with NVIDIA wheels')
    dump(a.out/'runtime.json',dict(torch=torch.__version__,numpy=numpy.__version__,lightgbm=lightgbm.__version__,cpu_threads=a.threads,gpus=ids))
    print('Preflight: data, tree and neural mechanism tests',flush=True)
    for pattern in ['test_code10k*.py','test_tree_subset.py','test_formal_sources.py']:
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p',pattern,'-v'],cwd=ROOT,check=True)
    print('Stage 1: verify frozen raw inputs and materialize train/development',flush=True)
    train,dev,labels,enc=materialize(a)
    print('Stage 2: refit LightGBM and Encoder',flush=True)
    fit_tree(a,train,dev,labels,enc);fit_encoder(a,train,dev,labels,enc)
    cfg=read(ROOT/'configs/formal_result_protocol_20261006.json')
    cfg['fit_source']='accepted 6988 train patients; no external fitting bank';cfg['evaluation']={'status':'not run by source fitting','partition':'test','patients':1506}
    cfg['nonblocking_supplements']=[];cfg['limitations']=['Retrospective synthetic cohort','Machine mapping status is not semantic validation']
    dump(a.out/'formal_fit_protocol.json',cfg)
    tc=read(ROOT/'configs/tabr_pilot_20261007.json');tc['scope']='new 6988-patient fit; development diagnostics only'
    dump(a.out/'tabr_fit_protocol.json',tc)
    jobs=[('formal',[sys.executable,'-u',str(ROOT/'scripts/freeze_formal_sources.py'),'--source',str(a.out/'source'),'--probe',str(a.out/'encoder'),'--protocol',str(a.out/'formal_fit_protocol.json'),'--out',str(a.out/'formal'),'--threads',str(a.threads)]),
          ('retrieval',[sys.executable,'-u',str(ROOT/'scripts/run_tabr_pilot.py'),'--source',str(a.out/'source'),'--out',str(a.out/'retrieval'),'--config',str(a.out/'tabr_fit_protocol.json'),'--gpus',','.join(ids[:2]),'--threads',str(a.threads)])]
    print('Stage 3: association/GRASP on CPU; TabR and No-retrieval on two GPUs',flush=True)
    processes=[]
    try:
        for name,cmd in jobs:
            log=(a.out/(name+'.log')).open('w',encoding='utf-8')
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);processes.append((name,p,log));print('START',name,flush=True)
        while any(p.poll() is None for _,p,_ in processes):
            print('Workers:',{n:('running' if p.poll() is None else p.returncode) for n,p,_ in processes},flush=True);time.sleep(30)
        errors=[n for n,p,_ in processes if p.returncode]
        if errors:raise RuntimeError('Worker failure: '+','.join(errors)+'; inspect included logs')
    finally:
        for _,p,log in processes:
            if p.poll() is None:p.terminate();p.wait()
            log.close()
    model_paths=[a.out/'source/tree/mask_augmented.txt',a.out/'encoder/encoder.pt',a.out/'formal/cohd_counts.json',a.out/'formal/grasp.pt',a.out/'retrieval/tabr/checkpoint.pt',a.out/'retrieval/no_retrieval/checkpoint.pt']
    dump(a.out/'fitted_sources.json',dict(status='complete',split_sha256=SPLIT,labels=labels,models={str(p.relative_to(a.out)):sha(p) for p in model_paths},voting_bank_sha256=sha(a.out/'source/tree/teacher_fit_selected_private.jsonl'),test_predictions_computed=False))
    return dict(status='complete',fit_patients=6988,development_patients=1506,classes=46,newly_fitted_models=6,test_evaluation_run=False,api_calls=0,next='Accept source mechanism checks, then execute RQ1/RQ2/RQ3 with these frozen models')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample',type=Path,default=Path('inputs/ddxplus_sample'))
    p.add_argument('--protocol',type=Path,default=Path('artifacts/code10k_protocol_v1'))
    p.add_argument('--out',type=Path,default=Path('artifacts/code10k_sources_v1'))
    p.add_argument('--gpus',default='0,1');p.add_argument('--threads',type=int,default=16)
    p.add_argument('--prepare-only',action='store_true',help='Verify data contract without fitting; use a distinct output directory')
    a=p.parse_args();a.sample=a.sample.resolve();a.protocol=a.protocol.resolve();a.out=a.out.resolve()
    a.out.mkdir(parents=True,exist_ok=False);state={};start=time.time()
    try:
        if a.threads<1:raise ValueError('Positive thread count required')
        if a.prepare_only:materialize(a);state=dict(status='prepared_only',training_complete=False)
        else:state=run(a)
    except Exception:state=dict(status='failed',error=traceback.format_exc());raise
    finally:
        state['elapsed_seconds']=time.time()-start
        state['code_sha256']={str(f.relative_to(ROOT)):sha(f) for folder in ['scripts','src','configs'] for f in (ROOT/folder).rglob('*') if f.is_file() and f.suffix in ('.py','.json')}
        dump(a.out/'run_manifest.json',state)
        archive=a.out.with_name(a.out.name+'_reports.tar.gz')
        with tarfile.open(archive,'w:gz') as tar:
            for f in a.out.rglob('*'):
                if f.is_file() and f.suffix in ('.json','.jsonl','.log') and 'selected_private' not in f.name and f.name!='cohd_counts.json':tar.add(f,arcname=str(f.relative_to(a.out)))
        print('Return report:',archive,flush=True)
if __name__=='__main__':main()
