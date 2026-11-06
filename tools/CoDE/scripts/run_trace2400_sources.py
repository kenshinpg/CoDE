"""TRACE2400 first source fitting: reuse Counts, LightGBM fit_models and ContextEncoder.
Training and development only. Test records are never opened.
"""
import argparse,hashlib,json,sys,tarfile,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_inputs(view,previous=None,production=True):
    from code_eval.trace2400.encoding import TraceEncoding
    manifest=read(view/'manifest.json');report=read(view/'view_report.json');checked={}
    for n in ['view_report.json','view_config.json','split_PRIVATE.jsonl','evidence_dictionary.json','train_records_PRIVATE.jsonl','development_records_PRIVATE.jsonl']:
        checked[n]=sha(view/n)
        if checked[n]!=manifest['output_sha256'].get(n):raise ValueError('Frozen input identity mismatch: '+n)
    if report['view_id']!='trace2400_record_evidence_v2_names' or report['missing_display_names'] or not report['common_llm_input_ready']:raise ValueError('Wrong/unready common view')
    split=rows(view/'split_PRIVATE.jsonl');by_id={r['admission_id']:r for r in split}
    if len(split)!=len(by_id):raise ValueError('Duplicate split identity')
    groups=defaultdict(set)
    for r in split:groups[r['subject_group']].add(r['partition'])
    if any(len(v)!=1 for v in groups.values()):raise ValueError('Subject crosses partitions')
    parts=Counter(r['partition'] for r in split)
    if production and dict(parts)!={'train':1680,'development':361,'test':359}:raise ValueError('Accepted partition counts changed')
    previous_match=None
    if previous:
        previous_match=sha(previous/'split_PRIVATE.jsonl')==checked['split_PRIVATE.jsonl']
        if not previous_match:raise ValueError('Patient split changed after name repair')
    dictionary=read(view/'evidence_dictionary.json')
    if {'HP_0001082','HP_0001733','HP_0001735','HP_6000143'} & set(dictionary['features']):raise ValueError('Direct target concept survived')
    enc=TraceEncoding(dictionary);data={}
    for p in ['train','development']:
        records=rows(view/(p+'_records_PRIVATE.jsonl'));ids=set()
        for r in records:
            aid=r['patient_id'];s=by_id.get(aid)
            if not s or s['partition']!=p or r['partition']!=p or r['label']!=s['reference_class'] or aid in ids:raise ValueError('Record/split mismatch')
            if r['context']!={} or r['input_group_id']!=aid or not r['tokens']:raise ValueError('Invalid TRACE model input')
            if len(set(r['tokens']))!=len(r['tokens']):raise ValueError('Duplicate tokens')
            enc.encode(r['tokens'],r['context']);ids.add(aid)
        if ids!={a for a,s in by_id.items() if s['partition']==p}:raise ValueError('Missing partition records')
        data[p]=records
    labels=sorted({r['label'] for r in data['train']})
    if production and labels!=['appendicitis','cholecystitis','diverticulitis','pancreatitis']:raise ValueError('Unexpected disease support')
    if any(r['label'] not in labels for r in data['development']):raise ValueError('Unsupported development label')
    train_features={t.split('_@_',1)[0] for r in data['train'] for t in r['tokens']}
    if train_features!=set(dictionary['features']):raise ValueError('Dictionary contains nontraining features')
    return data,labels,enc,dict(dataset='TRACE2400-v1',view_id=report['view_id'],partitions=dict(parts),features=len(enc.columns),
      groups=len(groups),previous_split_identical=previous_match,input_sha256=checked,test_feature_records_opened=False,
      fixed_context=[],clinical_leakage_cleared=False,development_exposure_cleared=False)

def run(a):
    import numpy as np,torch,lightgbm as lgb
    from code_eval.cohort_stats import Counts
    from code_eval.context_encoder import ContextEncoder
    from code_eval.tree_subset import metrics,checked_contributions
    from run_tree_subset_pilot import fit_models
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA/MetaX torch unavailable; keep vendor torch or select --device cpu')
    torch.set_num_threads(a.threads)
    data,labels,enc,contract=load_inputs(a.view,a.previous_view,not a.synthetic_smoke)
    dump(a.out/'data_contract.json',contract);dump(a.out/'encoding.json',enc.export())
    dump(a.out/'runtime.json',{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'lightgbm':lgb.__version__,'device':a.device,'threads':a.threads})
    train,dev=data['train'],data['development'];seed=a.seed
    fractions=[0,.25,.5,.75,1.]
    dev_sets={str(f):[enc.masked(r['tokens'],f,f"{seed}|development|{r['input_group_id']}") for r in dev] for f in fractions}
    y=np.array([labels.index(r['label']) for r in dev]);prior=np.array([sum(r['label']==d for r in train)/len(train) for d in labels])
    print('Stage 1: existing LightGBM fitting (full_only + mask_augmented)',flush=True)
    tree=a.out/'tree';tree.mkdir()
    models,params=fit_models(enc,labels,{'teacher_fit':train,'fusion':dev},SimpleNamespace(seed=seed,threads=a.threads,rounds=a.rounds,out=tree))
    checkx=np.stack([enc.encode(r['tokens'],{}) for r in dev[:8]])
    for model in models.values():checked_contributions(model,checkx,len(labels))
    print('Stage 2: existing cohort Counts fit',flush=True)
    counts=Counts(labels)
    for r in train:counts.add(r['tokens'],r['label'])
    counts.validate();dump(a.out/'association.json',{'counts':counts.export(),'alpha':5.,'context':[]})
    print('Stage 3: existing ContextEncoder with disabled context branch',flush=True)
    vocab={t:i+1 for i,t in enumerate(sorted({t for r in train for t in r['tokens']}))}
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model=ContextEncoder(len(vocab),len(labels),128).to(a.device)
    model.context.weight.requires_grad_(False) # zero input removes demographic term exactly
    def tensor(ts):
        x=torch.zeros((len(ts),max(1,max(map(len,ts)))),dtype=torch.long,device=a.device)
        for i,t in enumerate(ts):
            if t:x[i,:len(t)]=torch.tensor([vocab[v] for v in t],device=a.device)
        return x,torch.zeros((len(ts),2),dtype=torch.float32,device=a.device)
    ins=[r['tokens'] for r in train]+[enc.masked(r['tokens'],[0,.25,.5,.75][i%4],f"{seed}|fit|{r['input_group_id']}") for i,r in enumerate(train)]
    target=[labels.index(r['label']) for r in train]*2
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001);history=[]
    for epoch in range(a.epochs):
        model.train();total=0.;order=np.random.default_rng(seed+epoch).permutation(len(ins))
        for start in range(0,len(order),256):
            ids=order[start:start+256];x,h=tensor([ins[i] for i in ids]);by=torch.tensor([target[i] for i in ids],device=a.device)
            opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x,h),by)
            if not torch.isfinite(loss):raise ValueError('Nonfinite encoder loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=float(loss.detach())*len(ids)
        history.append({'epoch':epoch+1,'loss':total/len(ins)});dump(a.out/'encoder_history.json',history);print('Encoder',history[-1],flush=True)
    torch.save({'state':{k:v.cpu() for k,v in model.state_dict().items()},'vocabulary':vocab,'labels':labels,'hidden':128,'context_policy':'zero vector; frozen demographic branch'},a.out/'encoder.pt');model.eval()
    summary={};prediction_dir=a.out/'development_predictions_PRIVATE';prediction_dir.mkdir()
    print('Stage 4: development-only full and sparse checks',flush=True)
    for frac,sets in dev_sets.items():
        available=np.array([all(t in vocab for t in ts) for ts in sets]);idx=np.flatnonzero(available)
        predictions={'prior':np.tile(prior,(len(dev),1))}
        for name,m in models.items():predictions['tree_'+name]=m.predict(np.stack([enc.encode(ts,{}) for ts in sets]),num_threads=a.threads)
        for name in ['association','encoder']:predictions[name]=np.full((len(dev),len(labels)),np.nan)
        for i in idx:predictions['association'][i]=[r['support'] for r in counts.score(sets[i],alpha=5.)['scores']]
        with torch.no_grad():
            for start in range(0,len(idx),256):
                ids=idx[start:start+256];x,h=tensor([sets[i] for i in ids]);predictions['encoder'][ids]=model(x,h).softmax(1).cpu().numpy()
        result={}
        for name,probs in predictions.items():
            ok=np.isfinite(probs).all(1);result[name]={'available':int(ok.sum()),'unavailable':int((~ok).sum()),'all_available':metrics(probs[ok],y[ok]) if ok.any() else None,
              'common_population':metrics(probs[idx],y[idx]) if len(idx) else None}
            np.savez_compressed(prediction_dir/f'{name}_keep{frac}.npz',probabilities=probs,targets=y,ids=np.array([r['patient_id'] for r in dev]),available=ok)
        summary[frac]={'common_population_n':len(idx),'methods':result};print('Validated fraction',frac,'common n',len(idx),flush=True)
    output={'status':'complete','dataset':'TRACE2400-v1','partition':'development','seed':seed,'training_n':len(train),'development_n':len(dev),'features':len(enc.columns),'labels':labels,'results':summary,
      'configuration':{'tree_rounds':a.rounds,'tree_params':params,'encoder_epochs':a.epochs,'encoder_hidden':128,'batch_size':256,'learning_rate':.001,'association_alpha':5.},
      'unseen_policy':'Same strict available-record policy as DDXPlus neural/association comparison. No unknown-token removal. Tree and prior retain all records; common population also reported.',
      'comparability':'Reuse DDXPlus algorithms, augmentation and metrics; TRACE omits unverified demographics and splits by inferred subject. Distinct dataset and task; no direct 4-vs-46 accuracy comparison.',
      'not_in_this_batch':['GRASP-inspired','TabR','RQ1 selection effects','RQ2 rationale evaluation','RQ3 controlled sets','test performance'],
      'limitations':['single seed development check','same-scale lab assumption','retrospective text; incomplete semantic leakage audit','historical development exposure unresolved']}
    dump(a.out/'development_summary.json',output)
    lines=['# TRACE2400-v1 training and development results','',
      'Dataset: TRACE2400-v1. Split: development only. DDXPlus results are not included.',
      f'Training admissions: {len(train)}; development admissions: {len(dev)}; features: {len(enc.columns)}.',
      'Patient representation uses ContextEncoder with its demographic term disabled. Association uses the original Counts scorer without demographic tokens. LightGBM reuses fit_models.',
      'Unseen states retain unavailable status for association/Encoder. All-available and common-population estimates must be distinguished. No clinical validation or independent-test claim is made.','',
      '| Retained fraction | Method | Available n | Accuracy | Macro accuracy | NLL | Common n | Common accuracy |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for frac,item in summary.items():
        for name,val in item['methods'].items():
            av=val['all_available'];cm=val['common_population']
            def fmt(v,k):return f'{v[k]:.6f}' if v else 'NA'
            lines.append(f"| {frac} | {name} | {val['available']} | {fmt(av,'accuracy')} | {fmt(av,'macro_accuracy')} | {fmt(av,'nll')} | {item['common_population_n']} | {fmt(cm,'accuracy')} |")
    lines.extend(['','Single-seed development estimates. This is a fitting/availability check, not RQ1, RQ2 or RQ3 evidence.','Historical development exposure and retrospective-source limitations remain.','No reference label, raw lab value or demographic placeholder enters the common patient feature view.'])
    (a.out/'TRACE2400_development_experiment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name,h in contract['input_sha256'].items():
        if sha(a.view/name)!=h:raise ValueError('Input changed during training')
    dump(a.out/'run_manifest.json',{'status':'complete','dataset':'TRACE2400-v1','input_sha256':contract['input_sha256'],
      'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/run_tree_subset_pilot.py',ROOT/'src/code_eval/cohort_stats.py',ROOT/'src/code_eval/context_encoder.py',ROOT/'src/code_eval/tree_subset.py',ROOT/'src/code_eval/trace2400/encoding.py']},
      'artifacts_sha256':{str(p.relative_to(a.out)):sha(p) for p in a.out.rglob('*') if p.is_file()}})
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--view',type=Path,default=Path('artifacts/trace2400_view_v2_names'));p.add_argument('--previous-view',type=Path,default=Path('artifacts/trace2400_view_v1'))
    p.add_argument('--out',type=Path,default=Path('artifacts/trace2400_sources_v1'));p.add_argument('--threads',type=int,default=16);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=261013)
    p.add_argument('--rounds',type=int,default=120);p.add_argument('--epochs',type=int,default=30);p.add_argument('--synthetic-smoke',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if a.out.exists() or a.out.resolve().is_relative_to(a.view.resolve()):p.error('Use a new output directory outside the input view')
    if min(a.threads,a.rounds,a.epochs)<=0:p.error('Threads, rounds and epochs must be positive')
    a.out.mkdir(parents=True);start=time.time()
    try:
        import numpy,torch,lightgbm
        import subprocess
        print('Preflight: input contracts and tiny reused-model fit',flush=True)
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_trace_sources.py','-v'],cwd=ROOT,check=True)
        r=run(a)
    except Exception as e:
        r={'status':'failed','exception_type':type(e).__name__,'locations':[{'module':Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]}
        if isinstance(e,ModuleNotFoundError):r['missing_module']=e.name
        dump(a.out/'failure.json',r);print(json.dumps(r),flush=True)
    with tarfile.open(str(a.out)+'_reports.tar.gz','w:gz') as t:
        for n in ['data_contract.json','runtime.json','development_summary.json','encoder_history.json','TRACE2400_development_experiment.md','run_manifest.json','failure.json']:
            if (a.out/n).exists():t.add(a.out/n,arcname=n)
    print('Elapsed seconds',round(time.time()-start,1),'Return report:',str(a.out)+'_reports.tar.gz',flush=True)
    if r['status']!='complete':sys.exit(1)
if __name__=='__main__':main()
