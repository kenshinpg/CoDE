"""Parallel existing GRASP-inspired / TabR / no-retrieval fits on TRACE."""
import argparse,json,os,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_trace2400_sources import read,rows,sha,dump,load_inputs

def verify_first(a,contract):
    m=read(a.first/'run_manifest.json')
    if m['status']!='complete' or m['input_sha256']!=contract['input_sha256']:raise ValueError('First source input mismatch')
    for n in ['encoder.pt','association.json','tree/mask_augmented.txt']:
        if sha(a.first/n)!=m['artifacts_sha256'][n]:raise ValueError('First source checkpoint changed')

def worker(a):
    import numpy as np,torch
    from code_eval.tree_subset import metrics
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Use the existing C550 torch environment')
    device='cuda:0';data,labels,enc,contract=load_inputs(a.view,a.previous_view)
    verify_first(a,contract);train,dev=data['train'],data['development'];seed=261013
    a.out.mkdir(parents=True,exist_ok=False);dump(a.out/'data_contract.json',contract)
    fractions=[0,.25,.5,.75,1.];y=np.array([labels.index(r['label']) for r in dev]);result={};audit={};history=[]
    predictions=a.out/'predictions_PRIVATE';predictions.mkdir()
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    if a.condition=='grasp':
        from code_eval.context_grasp import ContextGRASP
        g=read(ROOT/'configs/formal_result_protocol_20261006.json')['grasp'];g['seed']=seed
        ck=torch.load(a.first/'encoder.pt',map_location='cpu',weights_only=True);vocab=ck['vocabulary']
        if ck['labels']!=labels:raise ValueError('Class axes mismatch')
        model=ContextGRASP(len(vocab),len(labels),g['hidden'],g['centers']).to(device);model.context.weight.requires_grad_(False)
        def tensor(ts):
            x=torch.zeros((len(ts),max(1,max(map(len,ts)))),dtype=torch.long,device=device)
            for i,t in enumerate(ts):
                if t:x[i,:len(t)]=torch.tensor([vocab[k] for k in t],device=device)
            return x,torch.zeros((len(ts),2),device=device)
        ts=[r['tokens'] for r in train]+[enc.masked(r['tokens'],[0,.25,.5,.75][i%4],f"{seed}|fit|{r['input_group_id']}") for i,r in enumerate(train)]
        x,h=tensor(ts);target=torch.tensor([labels.index(r['label']) for r in train]*2,device=device)
        opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=g['learning_rate']);gradmax={k:0. for k in ['graph1','graph2','gate_graph','gate_patient']}
        for epoch in range(g['epochs']):
            warm=epoch<g['warmup']
            if not warm:
                model.eval()
                with torch.no_grad():emb=torch.cat([model.encode(x[i:min(i+256,len(train))],h[i:min(i+256,len(train))]) for i in range(0,len(train),256)])
                model.fit_centers(emb,seed=seed+epoch,neighbors=g['neighbors'],identity=epoch==g['warmup'])
            model.train();total=0.;order=np.random.default_rng(seed+epoch).permutation(len(target))
            for start in range(0,len(order),256):
                ix=torch.tensor(order[start:start+256],device=device);opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(x[ix],h[ix],use_graph=not warm),target[ix])
                if not torch.isfinite(loss):raise ValueError('Nonfinite GRASP loss')
                loss.backward()
                for name in gradmax:
                    grad=getattr(model,name).weight.grad
                    if grad is not None:gradmax[name]=max(gradmax[name],float(grad.norm()))
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=float(loss.detach())*len(ix)
            history.append({'epoch':epoch+1,'warmup':warm,'loss':total/len(target)});dump(a.out/'history.json',history);print(a.condition,history[-1],flush=True)
        model.eval();torch.save({'state':{k:v.cpu() for k,v in model.state_dict().items()},'vocabulary':vocab,'labels':labels,'hidden':g['hidden'],'centers':g['centers'],'context_policy':'disabled','protocol':g},a.out/'checkpoint.pt')
        for f in fractions:
            sets=[enc.masked(r['tokens'],f,f"{seed}|development|{r['input_group_id']}") for r in dev]
            idx=np.array([i for i,t in enumerate(sets) if all(k in vocab for k in t)],dtype=int);p=np.full((len(dev),len(labels)),np.nan)
            with torch.no_grad():
                for start in range(0,len(idx),128):
                    ix=idx[start:start+128];tx,th=tensor([sets[i] for i in ix]);p[ix]=model(tx,th).softmax(1).cpu().numpy()
                if f==1 and len(idx):
                    tx,th=tensor([sets[i] for i in idx]);normal,d=model(tx,th,return_audit=True);off=model(tx,th,use_graph=False);shift=model(tx,th,shift_assignment=True)
                    audit={'gradient_max':gradmax,'graph_off_l1':float((normal.softmax(1)-off.softmax(1)).abs().sum(1).mean()),'assignment_shift_l1':float((normal.softmax(1)-shift.softmax(1)).abs().sum(1).mean()),'used_centers':int(d['assignment'].unique().numel()),'propagated_variance':float(model.propagated().var(0,unbiased=False).mean()),'gate_mean':float(d['gate'].mean()),'representation_change_mean':float(d['representation_change'].mean()),'adjacency_row_sum_error':float((model.adj.sum(1)-1).abs().max()),'deterministic_forward_max_error':float((normal-model(tx,th)).abs().max())}
            result[str(f)]={'available':len(idx),'unavailable':len(dev)-len(idx),'metrics':metrics(p[idx],y[idx]) if len(idx) else None}
            np.savez_compressed(predictions/f'keep{f}.npz',probabilities=p,targets=y,ids=np.array([r['patient_id'] for r in dev]))
        audit['passed']=bool(audit and all(v>0 for v in gradmax.values()) and audit['graph_off_l1']>1e-6 and audit['assignment_shift_l1']>1e-6 and audit['used_centers']>=2 and audit['propagated_variance']>1e-10 and audit['adjacency_row_sum_error']<1e-6 and audit['deterministic_forward_max_error']<1e-6)
        cfg=g
    else:
        from code_eval.tabr_adapter import make_model
        from code_eval.trace2400.retrieval import TraceDenseEncoding,retrieval_groups
        from smoke_tabr import smoke
        dump(a.out/'core_equivalence_smoke.json',smoke(torch.device(device)))
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        cfg=read(ROOT/'configs/tabr_pilot_20261007.json');cfg['seed']=seed
        dense=TraceDenseEncoding(read(a.view/'evidence_dictionary.json'))
        subjects={r['admission_id']:r['subject_group'] for r in rows(a.view/'split_PRIVATE.jsonl')}
        excludes=retrieval_groups(train+dev,subjects);patterns=defaultdict(list)
        for r in train:patterns[r['pattern_id']].append(r)
        if any(len({r['label'] for r in rs})>1 for rs in patterns.values()):raise ValueError('Conflicting identical training patterns need explicit handling')
        bank=sorted([min(rs,key=lambda r:r['input_group_id']) for rs in patterns.values()],key=lambda r:r['input_group_id'])
        def tensor(rs,ts=None):return torch.tensor(np.stack([dense.encode(t,r['context']) for r,t in zip(rs,ts if ts is not None else [r['tokens'] for r in rs])]),device=device)
        bx=tensor(bank);by=torch.tensor([labels.index(r['label']) for r in bank],device=device);bp=torch.tensor([excludes[r['patient_id']] for r in bank],device=device)
        masked=[enc.masked(r['tokens'],[0,.25,.5,.75][i%4],f"{seed}|fit|{r['input_group_id']}") for i,r in enumerate(train)]
        x=torch.cat([tensor(train),tensor(train,masked)]);target=torch.tensor([labels.index(r['label']) for r in train]*2,device=device);qp=torch.tensor([excludes[r['patient_id']] for r in train]*2,device=device)
        minimum=int(min((bp!=q).sum().item() for q in qp.unique()))
        if minimum<cfg['context_size']:raise ValueError('Insufficient retrieval bank after subject exclusion')
        model=make_model(dense.width,len(labels),cfg['model']).to(device);use=a.condition=='tabr'
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay']);batch=cfg['batch_size']
        def predict(rs,ts=None,**flags):
            xx=tensor(rs,ts);pp=torch.tensor([excludes[r['patient_id']] for r in rs],device=device);pr=[]
            options={'use_retrieval':use};options.update(flags)
            with torch.no_grad():
                for i in range(0,len(rs),batch):pr.append(model(xx[i:i+batch],bx,by,pp[i:i+batch],bp,cfg['context_size'],**options).softmax(1).cpu().numpy())
            return np.concatenate(pr)
        for epoch in range(cfg['epochs']):
            model.train();total=0.;order=np.random.default_rng(seed+epoch).permutation(len(target))
            for start in range(0,len(order),batch):
                ix=torch.tensor(order[start:start+batch],device=device);opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(model(x[ix],bx,by,qp[ix],bp,cfg['context_size'],use_retrieval=use),target[ix])
                if not torch.isfinite(loss):raise ValueError('Nonfinite TabR loss')
                loss.backward();opt.step();total+=float(loss.detach())*len(ix)
            model.eval();history.append({'epoch':epoch+1,'loss':total/len(target),'validation':metrics(predict(dev),y)});dump(a.out/'history.json',history);print(a.condition,history[-1],flush=True)
        torch.save({'model':{k:v.cpu() for k,v in model.state_dict().items()},'config':cfg,'labels':labels,'encoding':{'tokens':dense.tokens,'features':dense.features,'context':[]},'bank_admission_ids':[r['patient_id'] for r in bank],'exclusion_components':excludes},a.out/'checkpoint.pt')
        for f in fractions:
            ts=[enc.masked(r['tokens'],f,f"{seed}|development|{r['input_group_id']}") for r in dev];pr=predict(dev,ts)
            result[str(f)]={'available':len(dev),'unavailable':0,'metrics':metrics(pr,y)}
            np.savez_compressed(predictions/f'keep{f}.npz',probabilities=pr,targets=y,ids=np.array([r['patient_id'] for r in dev]))
        audit={'bank_rows':len(bank),'training_rows':len(train),'minimum_allowed_bank':minimum,'same_subject_and_pattern_excluded':True,'passed':True}
        if use:
            selected=sorted(dev,key=lambda r:sha_string(f"{seed}|{r['patient_id']}"))[:100];original=predict(selected)
            off=predict(selected,use_retrieval=False);no_t=predict(selected,use_difference=False)
            saved=by.clone();by=by[torch.randperm(len(by),device=device)];shuffled=predict(selected);by=saved
            for name,pr in [('retrieval_off',off),('difference_off',no_t),('bank_labels_shuffled',shuffled)]:audit[name]={'mean_absolute_probability_change':float(np.abs(pr-original).mean()),'prediction_flip_rate':float((pr.argmax(1)!=original.argmax(1)).mean())}
            audit['passed']=all(audit[n]['mean_absolute_probability_change']>1e-8 for n in ['retrieval_off','difference_off','bank_labels_shuffled'])
    dump(a.out/'mechanism_audit.json',audit);dump(a.out/'sparse_validation.json',result)
    output={'status':'complete' if audit['passed'] else 'mechanism_check_failed','dataset':'TRACE2400-v1','condition':a.condition,'partition':'development','train_n':len(train),'development_n':len(dev),'labels':labels,'seed':seed,'configuration':cfg,'mechanism':audit,'test_features_opened':False}
    dump(a.out/'run_manifest.json',dict(output,checkpoint_sha256=sha(a.out/'checkpoint.pt'),input_sha256=contract['input_sha256']))
    return output

def combined_report(a):
    import numpy as np
    from code_eval.tree_subset import metrics
    bank={'encoder':a.first/'encoder.pt','cohd':a.first/'association.json','lightgbm':a.first/'tree/mask_augmented.txt',
          'grasp':a.out/'grasp/checkpoint.pt','tabr':a.out/'tabr/checkpoint.pt','no_retrieval':a.out/'no_retrieval/checkpoint.pt'}
    summary={};lines=['# TRACE2400-v1 complete source development comparison','',
      'All numbers use TRACE2400-v1 development admissions. No test evaluation or DDXPlus result is included.',
      'Original DDXPlus source implementations are reused; demographic context is disabled and retrieval excludes same-subject components.',
      '', '| Fraction | Method | All-available n | Accuracy | Common n | Common accuracy |','|---|---|---:|---:|---:|---:|']
    for frac in ['0','0.25','0.5','0.75','1.0']:
        allp={};base_ids=None;targets=None
        for name in ['prior','tree_full_only','tree_mask_augmented','association','encoder','grasp','tabr','no_retrieval']:
            path=a.first/'development_predictions_PRIVATE'/f'{name}_keep{frac}.npz' if name not in ['grasp','tabr','no_retrieval'] else a.out/name/'predictions_PRIVATE'/f'keep{frac}.npz'
            with np.load(path,allow_pickle=False) as z:
                ids=z['ids'];y=z['targets'];pr=z['probabilities']
                if base_ids is None:base_ids=ids.copy();targets=y.copy()
                if not np.array_equal(ids,base_ids) or not np.array_equal(y,targets):raise ValueError('Prediction axes differ')
                allp[name]=pr.copy()
        common=np.logical_and.reduce([np.isfinite(p).all(1) for p in allp.values()]);summary[frac]={'common_n':int(common.sum()),'methods':{}}
        for name,pr in allp.items():
            ok=np.isfinite(pr).all(1);av=metrics(pr[ok],targets[ok]) if ok.any() else None;cm=metrics(pr[common],targets[common]) if common.any() else None
            summary[frac]['methods'][name]={'available':int(ok.sum()),'unavailable':int((~ok).sum()),'all_available':av,'common_population':cm}
            lines.append(f"| {frac} | {name} | {int(ok.sum())} | {av['accuracy']:.6f} | {int(common.sum())} | {cm['accuracy']:.6f} |" if cm and av else f'| {frac} | {name} | {int(ok.sum())} | NA | 0 | NA |')
    dump(a.out/'combined_development.json',{'dataset':'TRACE2400-v1','partition':'development','results':summary,'test_evaluated':False})
    (a.out/'TRACE2400_extended_development.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(a.out/'fitted_sources.json',{'dataset':'TRACE2400-v1','view_id':'trace2400_record_evidence_v2_names','status':'fitted_development_checked',
      'models':{k:{'path':str(p),'sha256':sha(p)} for k,p in bank.items()},'input_sha256':read(a.first/'data_contract.json')['input_sha256'],
      'test_evaluated':False,'rq2_status':'requires rationales and cited evidence from this same view'})

def sha_string(s):
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--view',type=Path,default=Path('artifacts/trace2400_view_v2_names'));p.add_argument('--previous-view',type=Path,default=Path('artifacts/trace2400_view_v1'));p.add_argument('--first',type=Path,default=Path('artifacts/trace2400_sources_v1'));p.add_argument('--out',type=Path,default=Path('artifacts/trace2400_extended_sources_v1'));p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--threads',type=int,default=8);p.add_argument('--condition',choices=['grasp','tabr','no_retrieval']);a=p.parse_args()
    for k in ['view','previous_view','first','out']:setattr(a,k,getattr(a,k).resolve())
    if a.out.exists() or any(a.out.is_relative_to(q) for q in [a.view,a.first]):p.error('Output must be new and separate')
    if a.condition:
        try:r=worker(a)
        except Exception as e:
            a.out.mkdir(parents=True,exist_ok=True);r={'status':'failed','exception_type':type(e).__name__,'locations':[{'module':Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]};dump(a.out/'failure.json',r)
        if r['status']!='complete':sys.exit(1)
        return
    ids=a.gpus.split(',')
    if len(set(ids))<3:p.error('Three distinct visible GPU IDs are required')
    a.out.mkdir(parents=True);jobs=[];state={}
    try:
        import torch,numpy
        if not torch.cuda.is_available():raise RuntimeError('C550 torch unavailable')
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_trace_retrieval.py','-v'],cwd=ROOT,check=True)
        for condition,gpu in zip(['grasp','tabr','no_retrieval'],ids):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            log=(a.out/(condition+'.log')).open('w',encoding='utf-8')
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--condition',condition,'--view',str(a.view),'--previous-view',str(a.previous_view),'--first',str(a.first),'--out',str(a.out/condition),'--threads',str(a.threads)]
            jobs.append((condition,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log));print('START',condition,'GPU',gpu,flush=True)
        while any(p.poll() is None for n,p,l in jobs):
            print('Workers',[(n,p.poll()) for n,p,l in jobs],flush=True);time.sleep(20)
        state={'status':'complete' if all(p.returncode==0 for n,p,l in jobs) else 'failed','worker_codes':{n:p.returncode for n,p,l in jobs},'dataset':'TRACE2400-v1','test_evaluated':False}
        if state['status']=='complete':combined_report(a)
    except Exception as e:state={'status':'failed','exception_type':type(e).__name__}
    finally:
        for n,p,l in jobs:
            if p.poll() is None:p.terminate();p.wait()
            l.close()
        state['code_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/code_eval/context_grasp.py',ROOT/'src/code_eval/tabr_adapter.py',ROOT/'src/code_eval/tabr_official_core.py',ROOT/'src/code_eval/trace2400/retrieval.py']}
        dump(a.out/'run_manifest.json',state)
        with tarfile.open(str(a.out)+'_reports.tar.gz','w:gz') as t:
            for f in a.out.rglob('*'):
                if f.is_file() and f.suffix in ('.json','.md'):t.add(f,arcname=str(f.relative_to(a.out)))
        print('Return report:',str(a.out)+'_reports.tar.gz',flush=True)
    if state['status']!='complete':sys.exit(1)
if __name__=='__main__':main()
