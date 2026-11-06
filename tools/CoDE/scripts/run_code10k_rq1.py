"""Frozen-model RQ1: train-only class SHAP, full test census, matched controls."""
import argparse,itertools,json,math,os,random,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from prepare_code10k import sha,dump,rows,unique
from run_code10k_sources import SPLIT,structural_record
from evaluate_saved_subsets import matched_groups
from code_eval.tree_subset import Encoding,checked_contributions
from code_eval.rq1_summary import BUDGETS,FAMILIES,summarize

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def emit(f,r):f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def verify(a):
    cfg=read(a.config);manifest=a.models/'fitted_sources.json'
    if sha(manifest)!=cfg['fitted_manifest_sha256']:raise ValueError('Accepted fitted source manifest drift')
    fm=read(manifest)
    if fm['status']!='complete' or fm['split_sha256']!=SPLIT:raise ValueError('Incomplete/unmatched fitted sources')
    for n,h in fm['models'].items():
        if sha(a.models/n)!=h:raise ValueError('Checkpoint drift: '+n)
    trainpath=a.models/'source/tree/teacher_fit_selected_private.jsonl'
    if sha(trainpath)!=fm['voting_bank_sha256']:raise ValueError('Training bank drift')
    if sha(a.protocol/'patient_split_PRIVATE.jsonl')!=SPLIT:raise ValueError('Patient split drift')
    report=read(a.protocol/'cohort_report.json')
    for name in ['patients_public.jsonl','evidence_dictionary.json']:
        expected=[v for k,v in report['input_sha256'].items() if Path(k).name==name]
        if expected!=[sha(a.sample/'data'/name)]:raise ValueError('Original patient/dictionary drift')
    dictionary=a.models/'source/cohort/evidence_dictionary.json'
    if sha(dictionary)!=sha(a.sample/'data/evidence_dictionary.json'):raise ValueError('Model dictionary differs')
    return fm,list(rows(trainpath)),read(dictionary)

def class_shap(a,fm,train,dictionary):
    import lightgbm as lgb
    enc=Encoding(dictionary);labels=fm['labels'];model=lgb.Booster(model_file=str(a.models/'source/tree/mask_augmented.txt'))
    sums=np.zeros((len(labels),len(enc.columns)));counts=np.zeros(len(labels),dtype=int)
    for start in range(0,len(train),16):
        batch=train[start:start+16];x=np.stack([enc.encode(r['tokens'],r['context']) for r in batch])
        values=checked_contributions(model,x,len(labels))
        for i,r in enumerate(batch):
            d=labels.index(r['label']);sums[d]+=np.abs(values[i,d,:-1]);counts[d]+=1
        if start%160==0:print('Class SHAP',min(start+16,len(train)),'/',len(train),flush=True)
    if counts.sum()!=6988 or (counts==0).any():raise ValueError('Class attribution fit population mismatch')
    dump(a.out/'class_shap.json',dict(labels=labels,columns=enc.columns,counts=counts.tolist(),mean_absolute=(sums/counts[:,None]).tolist(),
        train_sha256=fm['voting_bank_sha256'],tree_sha256=fm['models']['source/tree/mask_augmented.txt'],
        definition='Mean absolute class-d logit SHAP on full training patients with reference diagnosis d; raw-score reconstruction verified'))

class RetrievalSources:
    def __init__(self,a,fit,labels,dictionary):
        import torch
        from code_eval.tabr_adapter import DenseEncoding,make_model
        from code_eval.case_retrieval import CaseRetrieval
        self.torch=torch;self.labels=labels;self.device=torch.device('cuda');self.models={}
        self.cases=CaseRetrieval(fit,labels,k=5);self.enc=DenseEncoding(dictionary,fit)
        self.patterns={r['pattern_id'] for r in fit};self.ids={r['input_group_id'] for r in fit}
        self.bx=torch.tensor(np.stack([self.enc.encode(r['tokens'],r['context']) for r in self.cases.rows]),device=self.device)
        self.by=torch.tensor([labels.index(r['label']) for r in self.cases.rows],device=self.device)
        self.bp=torch.arange(len(self.cases.rows),device=self.device)
        for name in ['tabr','no_retrieval']:
            ck=torch.load(a.models/'retrieval'/name/'checkpoint.pt',map_location='cpu',weights_only=True)
            if ck['labels']!=labels or ck['epoch']!=30:raise ValueError('Retrieval class/checkpoint mismatch')
            if ck['encoding']!=dict(tokens=self.enc.tokens,features=self.enc.features,age_mean=self.enc.mean,age_std=self.enc.std):raise ValueError('Retrieval input encoding drift')
            model=make_model(self.enc.width,len(labels),ck['config']['model']).to(self.device);model.load_state_dict(ck['model'],strict=True);model.eval()
            self.models[name]=(model,ck['config']['context_size'])
    def predict(self,inputs,r):
        torch=self.torch
        if r['pattern_id'] in self.patterns or r['patient_id'] in self.ids:raise ValueError('Query patient/pattern in fitted bank')
        q=torch.tensor(np.stack([self.enc.encode(ts,r['context']) for ts in inputs]),device=self.device)
        qp=torch.full((len(inputs),),-1,dtype=torch.long,device=self.device);result={}
        with torch.no_grad():
            for name,(model,k) in self.models.items():
                result[name]=model(q,self.bx,self.by,qp,self.bp,k,use_retrieval=name=='tabr').softmax(1).cpu().numpy()
                if not np.isfinite(result[name]).all() or not np.allclose(result[name].sum(1),1,atol=1e-5):raise ValueError('Invalid retrieval probabilities')
        result['voting']=[]
        for ts in inputs:
            v=self.cases.query(ts,r['context'],excluded_pattern=r['pattern_id'],excluded_identity=r['patient_id'])
            result['voting'].append(np.asarray(v['probabilities']) if v['available'] else None)
        return result

def worker(a):
    import torch
    from code_eval.frozen_bank import FrozenBank
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fm,fit,dictionary=verify(a);labels=fm['labels'];enc=Encoding(dictionary)
    lock=read(a.out/'evaluation_lock.json')
    if lock['class_shap_sha256']!=sha(a.out/'class_shap.json'):raise ValueError('Class SHAP drift')
    cls=read(a.out/'class_shap.json');means=np.asarray(cls['mean_absolute'])
    if cls['labels']!=labels or cls['columns']!=enc.columns or cls['train_sha256']!=fm['voting_bank_sha256']:raise ValueError('Class attribution axes/provenance drift')
    paths={n:a.models/p for n,p in {'lightgbm':'source/tree/mask_augmented.txt','encoder':'encoder/encoder.pt','grasp':'formal/grasp.pt','cohd':'formal/cohd_counts.json'}.items()}
    bank=FrozenBank(paths,enc,labels,a.threads);retrieval=RetrievalSources(a,fit,labels,dictionary)
    gain=bank.tree.feature_importance(importance_type='gain')
    split=unique(a.protocol/'patient_split_PRIVATE.jsonl','patient_id');test=sorted(pid for pid,s in split.items() if s['partition']=='test')
    patients=unique(a.sample/'data/patients_public.jsonl','patient_id');selected=test[a.shard::a.shards]
    out=a.out/f'shard_{a.shard}';out.mkdir();started=time.time();state={}
    try:
        with (out/'selections.jsonl').open('w',encoding='utf-8') as dest,(out/'patient_audit.jsonl').open('w',encoding='utf-8') as audit:
            for i,pid in enumerate(selected):
                r=structural_record(patients[pid],split[pid]);groups=enc.groups(r['tokens']);full=set(r['tokens']);gs=sorted(groups)
                minus=[sorted(full-set(groups[g])) for g in gs];inputs=[sorted(full)]+minus
                ps=bank.predict(inputs,r['context']);ps.update(retrieval.predict(inputs,r));orders={};weight_audit={}
                for name,values in ps.items():
                    if all(v is not None for v in values):
                        d=int(values[0].argmax());w={g:abs(float(values[0][d]-values[j+1][d])) for j,g in enumerate(gs)}
                        orders[name]=sorted(gs,key=lambda g:(-w[g],g));weight_audit[name]=dict(target=labels[d],absolute_mass=sum(w.values()),nonzero_groups=sum(v>1e-12 for v in w.values()))
                shap=checked_contributions(bank.tree,enc.encode(r['tokens'],r['context'])[None],len(labels))[0]
                target=int(ps['lightgbm'][0].argmax());cols={g:sorted({enc.token(t)[0] for t in ts}) for g,ts in groups.items()}
                for name,w in {'tree_gain':{g:float(gain[c].sum()) for g,c in cols.items()},'tree_local_abs_shap':{g:float(np.abs(shap[target,c]).sum()) for g,c in cols.items()},'tree_class_abs_shap':{g:float(means[target,c].sum()) for g,c in cols.items()}}.items():
                    orders[name]=sorted(gs,key=lambda g:(-w[g],g))
                both=all(ps[n][0] is not None and labels[int(ps[n][0].argmax())]==r['label'] for n in ['encoder','lightgbm'])
                emit(audit,dict(patient_id=pid,pattern_id=r['pattern_id'],component=r['component'],reference=r['label'],both_full_correct=both,
                    full_predictions={n:labels[int(v[0].argmax())] if v[0] is not None else None for n,v in ps.items()},importance=weight_audit,available_selectors=sorted(orders)))
                tasks=[];queries=set();control_cache={}
                for name in sorted(set(itertools.chain.from_iterable(FAMILIES.values()))):
                    for frac in BUDGETS:
                        if name not in orders:
                            tasks.append((name,frac,None,None));continue
                        ts=tuple(sorted(t for g in orders[name][:math.ceil(frac*len(gs))] for t in groups[g]))
                        key=(frac,len(ts))
                        if key not in control_cache:control_cache[key]=[tuple(matched_groups(groups,len(ts),random.Random(f'261006|{pid}|{frac}|{len(ts)}|{j}'))) for j in range(10)]
                        controls=control_cache[key]
                        if any(len(c)!=len(ts) for c in controls):raise ValueError('Random count matching failed')
                        queries.add(ts);queries.update(controls);tasks.append((name,frac,ts,controls))
                keys=sorted(queries);pred=bank.predict([list(q) for q in keys],r['context'],sources=['encoder','lightgbm'])
                lookup={n:dict(zip(keys,v)) for n,v in pred.items()}
                for name,frac,ts,controls in tasks:
                    for probe in ['encoder','lightgbm']:
                        valid=ts is not None and lookup[probe][ts] is not None and all(lookup[probe][c] is not None for c in controls)
                        emit(dest,dict(patient_id=pid,selector=name,probe=probe,fraction=frac,both_full_correct=both,available=valid,
                            selected_tokens=list(ts) if ts is not None else None,fact_count=len(ts) if ts is not None else None,
                            correct=int(labels[int(lookup[probe][ts].argmax())]==r['label']) if valid else None,
                            random_accuracy=float(np.mean([labels[int(lookup[probe][c].argmax())]==r['label'] for c in controls])) if valid else None,
                            random_correct=[int(labels[int(lookup[probe][c].argmax())]==r['label']) for c in controls] if valid else None,
                            controls=[list(c) for c in controls] if controls is not None else None,
                            unavailable_reason=None if valid else ('source_importance_unavailable' if ts is None else 'probe_unavailable')))
                if (i+1)%10==0:print('Shard',a.shard,'patients',i+1,'/',len(selected),'seconds',round(time.time()-started,1),flush=True)
        state=dict(status='complete',patients=len(selected),patient_ids=selected)
    except Exception:state=dict(status='failed',error=traceback.format_exc());raise
    finally:dump(out/'run_manifest.json',state)

def main_run(a):
    fm,train,dictionary=verify(a)
    ids=a.gpus.split(',')
    if len(ids)!=len(set(ids)):raise ValueError('Distinct GPUs required')
    for test in ['test_code10k_rq1.py','test_reserved_evaluation.py']:
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p',test,'-v'],cwd=ROOT,check=True)
    print('Stage 1: training-only class SHAP',flush=True);class_shap(a,fm,train,dictionary)
    dump(a.out/'evaluation_lock.json',dict(config=read(a.config),split_sha256=SPLIT,fitted_manifest_sha256=sha(a.models/'fitted_sources.json'),class_shap_sha256=sha(a.out/'class_shap.json'),families=FAMILIES,
        budgets=BUDGETS,random_repeats=10,bootstrap=1000,seed=261006,patients=1506,
        primary_population='both full probes correct; common availability within each comparison family',sensitivity_population='all test patients; common availability within each comparison family',
        selection_target='source own full prediction; tree attribution rules use tree full prediction',no_mapping_filter=True,no_training=True))
    processes=[]
    try:
        for i,gpu in enumerate(ids):
            cmd=[sys.executable,'-u',__file__,'--models',str(a.models),'--protocol',str(a.protocol),'--sample',str(a.sample),'--config',str(a.config),'--out',str(a.out),'--shard',str(i),'--shards',str(len(ids)),'--threads',str(a.threads)]
            log=(a.out/f'worker_{i}.log').open('w',encoding='utf-8')
            p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);processes.append((p,log));print('START shard',i,'GPU',gpu,flush=True)
        while any(p.poll() is None for p,_ in processes):
            print('Workers',[(i,'running' if p.poll() is None else p.returncode) for i,(p,_) in enumerate(processes)],flush=True);time.sleep(30)
        if any(p.returncode for p,_ in processes):raise RuntimeError('Worker failed; return logs')
    finally:
        for p,log in processes:
            if p.poll() is None:p.terminate();p.wait()
            log.close()
    records=[];audits=[]
    for i in range(len(ids)):
        d=a.out/f'shard_{i}';assert read(d/'run_manifest.json')['status']=='complete'
        records.extend(rows(d/'selections.jsonl'));audits.extend(rows(d/'patient_audit.jsonl'))
    expected={r['patient_id'] for r in rows(a.protocol/'patient_split_PRIVATE.jsonl') if r['partition']=='test'}
    if len(audits)!=1506 or {r['patient_id'] for r in audits}!=expected:raise ValueError('Test census incomplete')
    if len(records)!=1506*10*3*2:raise ValueError('Incomplete selection rows')
    print('Stage 3: patient-paired intervals and overlap',flush=True);dump(a.out/'rq1_summary.json',summarize(records))
    dump(a.out/'population_flow.json',dict(starting_patients=1506,both_full_correct=sum(r['both_full_correct'] for r in audits),mapping_filter=False,
        full_probe_unavailable={n:sum(r['full_predictions'][n] is None for r in audits) for n in ['encoder','lightgbm']}))
    return dict(status='complete',patients=1506,selection_rows=len(records),model_training=False,api_calls=0,evaluation='RQ1 only; no natural-rationale or controlled-support claims')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,default=Path('artifacts/code10k_sources_v1'));p.add_argument('--protocol',type=Path,default=Path('artifacts/code10k_protocol_v1'))
    p.add_argument('--sample',type=Path,default=Path('inputs/ddxplus_sample'))
    p.add_argument('--config',type=Path,default=ROOT/'configs/code10k_rq1_20261007.json')
    p.add_argument('--out',type=Path,default=Path('artifacts/code10k_rq1_v1'));p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--shard',type=int);p.add_argument('--shards',type=int)
    a=p.parse_args()
    for n in ['models','protocol','sample','config','out']:setattr(a,n,getattr(a,n).resolve())
    if a.shard is not None:worker(a)
    else:
        a.out.mkdir(parents=True,exist_ok=False);state={};start=time.time()
        try:state=main_run(a)
        except Exception:state=dict(status='failed',error=traceback.format_exc());raise
        finally:
            state.update(seconds=time.time()-start,output_sha256={f.relative_to(a.out).as_posix():sha(f) for f in a.out.rglob('*') if f.is_file()},
                code_sha256={f.relative_to(ROOT).as_posix():sha(f) for directory in ['scripts','src'] for f in (ROOT/directory).rglob('*.py')})
            dump(a.out/'run_manifest.json',state);archive=a.out.with_name(a.out.name+'_reports.tar.gz')
            with tarfile.open(archive,'w:gz') as tar:
                for f in a.out.rglob('*'):
                    if f.is_file():tar.add(f,arcname=f.relative_to(a.out).as_posix())
            print('Return report:',archive,flush=True)
