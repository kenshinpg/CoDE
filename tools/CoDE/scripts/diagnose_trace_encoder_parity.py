"""Read-only Encoder parity diagnosis. Development inputs only; no training."""
import argparse,json,sys,tarfile,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_trace2400_sources import load_inputs,read,sha,dump

def compare(got,want):
    good=np.isfinite(got).all(1);old=np.isfinite(want).all(1);common=good&old
    delta=np.abs(got[common]-want[common]);ok=np.isclose(got[common],want[common],atol=2e-5,rtol=1e-5)
    return dict(availability_mismatch=int((good!=old).sum()),rows=int(common.sum()),max_abs_error=float(delta.max()) if delta.size else None,mean_abs_error=float(delta.mean()) if delta.size else None,mismatched_rows=int((~ok.all(1)).sum()),prediction_flips=int((got[common].argmax(1)!=want[common].argmax(1)).sum()),passed=bool(np.array_equal(good,old) and ok.all()))

def run(a):
    import torch
    from code_eval.context_encoder import ContextEncoder
    torch.set_num_threads(a.threads)
    defaults={'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_precision':torch.get_float32_matmul_precision()}
    data,labels,enc,contract=load_inputs(a.view);dev=data['development'];checkpoint=a.first/'encoder.pt';manifest=read(a.first/'run_manifest.json')
    if manifest['input_sha256']!=contract['input_sha256'] or sha(checkpoint)!=manifest['artifacts_sha256']['encoder.pt']:raise ValueError('Checkpoint or input drift')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck['labels']!=labels:raise ValueError('Label axis mismatch')
    vocab=ck['vocabulary'];device='cuda:0'
    model=ContextEncoder(len(vocab),len(labels),ck['hidden']).to(device);model.load_state_dict(ck['state'],strict=True);model.eval()
    report=dict(dataset='TRACE2400-v1',scope='development Encoder only',torch=torch.__version__,device=torch.cuda.get_device_name(0),startup_math=defaults,checkpoint_sha256=sha(checkpoint),parameter_dtypes=sorted({str(v.dtype) for v in model.parameters()}),results={},test_records_opened=False)
    cpu=None
    for f in [0,.25,.5,.75,1.]:
        ts=[enc.masked(r['tokens'],f,f"261013|development|{r['input_group_id']}") for r in dev]
        valid=[i for i,t in enumerate(ts) if all(v in vocab for v in t)]
        file=a.first/'development_predictions_PRIVATE'/f'encoder_keep{f}.npz'
        if sha(file)!=manifest['artifacts_sha256'][file.relative_to(a.first).as_posix()]:raise ValueError('Saved prediction drift')
        with np.load(file,allow_pickle=False) as z:
            if z['ids'].tolist()!=[r['patient_id'] for r in dev] or z['targets'].tolist()!=[labels.index(r['label']) for r in dev]:raise ValueError('Saved prediction axis mismatch')
            want=z['probabilities'].copy()
        def predict(m,devic,batch):
            pr=np.full(want.shape,np.nan)
            with torch.no_grad():
                for start in range(0,len(valid),batch):
                    ids=valid[start:start+batch];x=torch.zeros((len(ids),max(1,max(len(ts[i]) for i in ids))),dtype=torch.long,device=devic)
                    for j,i in enumerate(ids):
                        if ts[i]:x[j,:len(ts[i])]=torch.tensor([vocab[t] for t in ts[i]],device=devic)
                    pr[ids]=m(x,torch.zeros((len(ids),2),device=devic)).softmax(1).cpu().numpy()
            return pr
        outputs={}
        for mode,mt,ct,batch in [('startup_batch256',defaults['matmul_tf32'],defaults['cudnn_tf32'],256),('strict_batch256',False,False,256),('strict_batch128',False,False,128),('strict_single',False,False,1)]:
            torch.backends.cuda.matmul.allow_tf32=mt;torch.backends.cudnn.allow_tf32=ct
            pr=predict(model,device,batch);outputs[mode]=pr;report['results'][str(f)+'|'+mode]=compare(pr,want)
            dump(a.out/'encoder_parity_diagnostic.json',report);print(f,mode,report['results'][str(f)+'|'+mode],flush=True)
        if not all(compare(v,want)['passed'] for v in outputs.values()):
            if cpu is None:
                cpu=ContextEncoder(len(vocab),len(labels),ck['hidden']);cpu.load_state_dict(ck['state'],strict=True);cpu.eval()
            pr=predict(cpu,'cpu',256);report['results'][str(f)+'|cpu_batch256']=compare(pr,want)
            report['results'][str(f)+'|strict_single_vs_cpu']=compare(outputs['strict_single'],pr)
            print(f,'cpu_batch256',compare(pr,want),flush=True)
        report['results'][str(f)+'|batch256_vs_single']=compare(outputs['strict_batch256'],outputs['strict_single'])
    report['status']='diagnosis_complete';report['interpretation']='Comparisons diagnose batch/math sensitivity; no tolerance change or evaluation bypass applied.'
    dump(a.out/'encoder_parity_diagnostic.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--view',type=Path,default=Path('artifacts/trace2400_view_v2_names'));p.add_argument('--first',type=Path,default=Path('artifacts/trace2400_sources_v1'));p.add_argument('--out',type=Path,default=Path('artifacts/trace2400_encoder_parity_diagnostic_v1'));p.add_argument('--threads',type=int,default=4);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except Exception as e:
        dump(a.out/'failure.json',dict(exception_type=type(e).__name__,locations=[dict(module=Path(t.filename).name,line=t.lineno) for t in traceback.extract_tb(e.__traceback__)]));raise
    finally:
        with tarfile.open(str(a.out)+'_reports.tar.gz','w:gz') as tar:
            for file in a.out.glob('*.json'):tar.add(file,arcname=file.name)
        print('Return report:',str(a.out)+'_reports.tar.gz',flush=True)
