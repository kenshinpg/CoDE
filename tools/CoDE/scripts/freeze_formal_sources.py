"""Fit missing sources from frozen recipe; never load reserved records/outcomes."""
import argparse
from pathlib import Path
import json
import sys
import time
import tarfile
import traceback
import subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prepare_lantern_cohort import read,rows,sha,dump


def run(a):
    import numpy as np
    import torch
    from code_eval.tree_subset import Encoding,pattern,metrics
    from code_eval.context_grasp import ContextGRASP
    from code_eval.formal_sources import AssociationSource,context_tokens
    cfg=read(a.protocol);g=cfg['grasp'];torch.set_num_threads(a.threads)
    tree=a.source/'tree';cohort=a.source/'cohort';tm=read(tree/'run_manifest.json');pm=read(a.probe/'run_manifest.json')
    if tm['status']!='complete' or pm['status']!='complete' or pm['source_tree_manifest_sha256']!=sha(tree/'run_manifest.json'):raise ValueError('Incomplete or mismatched sources')
    for name in ['teacher_fit_selected_private.jsonl','fusion_selected_private.jsonl','mask_augmented.txt','encoding.json']:
        if sha(tree/name)!=tm['output_sha256'][name]:raise ValueError('Tree artifact drift: '+name)
    if sha(a.probe/'encoder.pt')!=pm['output_sha256']['encoder.pt']:raise ValueError('Encoder checkpoint drift')
    for name in ['excluded_fit_patterns.txt','evidence_dictionary.json']:
        if [h for p,h in tm['source_sha256'].items() if Path(p).name==name]!=[sha(cohort/name)]:raise ValueError('Cohort drift')
    train=list(rows(tree/'teacher_fit_selected_private.jsonl'));dev=list(rows(tree/'fusion_selected_private.jsonl'))
    excluded=set((cohort/'excluded_fit_patterns.txt').read_text().splitlines())
    for r in train+dev:
        if pattern(r['tokens'])!=r['pattern_id'] or r['pattern_id'] in excluded:raise ValueError('Fit isolation failed')
    if {r['pattern_id'] for r in train}&{r['pattern_id'] for r in dev}:raise ValueError('Fit/dev overlap')
    labels=tm['labels'];encoding=Encoding(read(cohort/'evidence_dictionary.json'))
    if encoding.export()!=read(tree/'encoding.json'):raise ValueError('Encoding drift')
    # Freeze provenance before fitting. No evaluation record is read by this program.
    lock=dict(protocol=cfg,protocol_sha256=sha(a.protocol),tree_manifest_sha256=sha(tree/'run_manifest.json'),
        encoder_manifest_sha256=sha(a.probe/'run_manifest.json'),fit_sha256=sha(tree/'teacher_fit_selected_private.jsonl'),
        exclusion_sha256=sha(cohort/'excluded_fit_patterns.txt'),evaluation_predictions_computed=False,
        checkpoint_existing_sha256={'lightgbm':sha(tree/'mask_augmented.txt'),'encoder':sha(a.probe/'encoder.pt')})
    dump(a.out/'recipe_lock.json',lock)
    print('Stage 1: association counts from',len(train),'patients',flush=True)
    assoc=AssociationSource(labels,cfg['cohd']['alpha']);assoc.fit(train);dump(a.out/'cohd_counts.json',assoc.export())
    ck=torch.load(a.probe/'encoder.pt',map_location='cpu',weights_only=True);vocab=ck['vocabulary']
    if ck['labels']!=labels or ck['hidden']!=g['hidden']:raise ValueError('Encoder axes/config mismatch')
    # Identical base initialization, train views, minibatch permutations and optimizer.
    torch.manual_seed(g['seed']);torch.use_deterministic_algorithms(True)
    model=ContextGRASP(len(vocab),len(labels),g['hidden'],g['centers'])
    mean=float(ck['age_mean']);std=float(ck['age_std'])
    def tensors(inputs):
        x=torch.zeros((len(inputs),max(1,max(len(ts) for ts,ctx in inputs))),dtype=torch.long)
        for i,(ts,ctx) in enumerate(inputs):x[i,:len(ts)]=torch.tensor([vocab[t] for t in ts],dtype=torch.long)
        h=torch.tensor([[(float(ctx['age'])-mean)/std,float(ctx['sex']=='M')] for ts,ctx in inputs],dtype=torch.float32)
        return x,h
    inputs=[(r['tokens'],r['context']) for r in train]
    inputs += [(encoding.masked(r['tokens'],[0,.25,.5,.75][i%4],f"{tm['seed']}|fit|{r['input_group_id']}"),r['context']) for i,r in enumerate(train)]
    x,h=tensors(inputs);y=torch.tensor([labels.index(r['label']) for r in train]*2)
    optimizer=torch.optim.Adam(model.parameters(),lr=g['learning_rate']);history=[];gradmax={k:0. for k in ['graph1','graph2','gate_graph','gate_patient']}
    print('Stage 2: matched context GRASP training; fixed final epoch',flush=True)
    for epoch in range(g['epochs']):
        warm=epoch<g['warmup']
        if not warm:
            model.eval()
            with torch.no_grad():emb=torch.cat([model.encode(x[i:min(i+g['batch_size'],len(train))],h[i:min(i+g['batch_size'],len(train))]) for i in range(0,len(train),g['batch_size'])])
            model.fit_centers(emb,seed=g['seed']+epoch,neighbors=g['neighbors'],identity=epoch==g['warmup'])
        model.train();order=np.random.default_rng(g['seed']+epoch).permutation(len(y));loss_sum=0.
        for start in range(0,len(y),g['batch_size']):
            ix=torch.tensor(order[start:start+g['batch_size']]);optimizer.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(x[ix],h[ix],use_graph=not warm),y[ix])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            for name in gradmax:
                grad=getattr(model,name).weight.grad
                if grad is not None:gradmax[name]=max(gradmax[name],float(grad.norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();loss_sum+=float(loss.detach())*len(ix)
        history.append(dict(epoch=epoch+1,warmup=warm,loss=loss_sum/len(y)));dump(a.out/'history.json',history)
        print('Epoch',epoch+1,'/',g['epochs'],'loss',history[-1]['loss'],flush=True)
    model.eval();torch.save(dict(state=model.state_dict(),vocabulary=vocab,labels=labels,age_mean=mean,age_std=std,hidden=g['hidden'],centers=g['centers'],protocol_sha256=sha(a.protocol)),a.out/'grasp.pt')
    print('Stage 3: structural development sparse validity and mechanism acceptance',flush=True)
    sparse={};audit=None
    for fraction in [0,.25,.5,.75,1]:
        inputs=[(encoding.masked(r['tokens'],fraction,f"{tm['seed']}|development|{r['input_group_id']}"),r['context']) for r in dev]
        cp=[assoc.distribution(ts,ctx) for ts,ctx in inputs];ci=[i for i,p in enumerate(cp) if p is not None]
        valid=[i for i,(ts,ctx) in enumerate(inputs) if all(t in vocab for t in ts)];gx,gh=tensors([inputs[i] for i in valid])
        with torch.no_grad():
            prob=model(gx,gh).softmax(1).numpy()
            if fraction==1:
                normal,details=model(gx,gh,return_audit=True);off=model(gx,gh,use_graph=False);shift=model(gx,gh,shift_assignment=True)
                propagated=model.propagated();p=normal.softmax(1)
                audit=dict(gradient_max=gradmax,graph_off_l1=float((p-off.softmax(1)).abs().sum(1).mean()),
                    assignment_shift_l1=float((p-shift.softmax(1)).abs().sum(1).mean()),
                    used_centers=int(details['assignment'].unique().numel()),propagated_variance=float(propagated.var(0,unbiased=False).mean()),
                    gate_mean=float(details['gate'].mean()),representation_change_mean=float(details['representation_change'].mean()),
                    adjacency_row_sum_error=float((model.adj.sum(1)-1).abs().max()),
                    deterministic_forward_max_error=float((normal-model(gx,gh)).abs().max()))
        sparse[str(fraction)]=dict(cohd=dict(available=len(ci),**metrics([cp[i] for i in ci],[labels.index(dev[i]['label']) for i in ci])),
            grasp=dict(available=len(valid),**metrics(prob,[labels.index(dev[i]['label']) for i in valid])),
            unseen_context_terms=sum(assoc.counts.feature[t]==0 for ts,ctx in inputs for t in context_tokens(ctx)))
        print('Sparse',fraction,json.dumps(sparse[str(fraction)]),flush=True)
    audit['passed']=bool(all(v>0 for v in gradmax.values()) and audit['graph_off_l1']>1e-6 and audit['assignment_shift_l1']>1e-6 and audit['used_centers']>=2 and audit['propagated_variance']>1e-10 and audit['adjacency_row_sum_error']<1e-6 and audit['deterministic_forward_max_error']<1e-6)
    audit['accuracy_is_not_acceptance_criterion']=True
    dump(a.out/'mechanism_audit.json',audit);dump(a.out/'sparse_validation.json',sparse)
    if not audit['passed']:raise ValueError('GRASP mechanism gate failed; reserved evaluation remains unopened; inspect mechanism_audit.json')
    bank=dict(status='frozen',recipe_sha256=sha(a.out/'recipe_lock.json'),protocol_sha256=sha(a.protocol),
        sources={'lightgbm':dict(path=str(tree/'mask_augmented.txt'),sha256=sha(tree/'mask_augmented.txt')),
                 'encoder':dict(path=str(a.probe/'encoder.pt'),sha256=sha(a.probe/'encoder.pt')),
                 'cohd':dict(path=str(a.out/'cohd_counts.json'),sha256=sha(a.out/'cohd_counts.json')),
                 'grasp':dict(path=str(a.out/'grasp.pt'),sha256=sha(a.out/'grasp.pt'))},
        source_input='same patient tokens and raw demographics; association age is decade-binned, neural/tree age continuous',
        fit_patients=len(train),structural_development_patients=len(dev),evaluation_opened=False)
    dump(a.out/'frozen_source_bank.json',bank)
    return dict(status='complete',sources=4,fit_patients=len(train),structural_development_patients=len(dev),
        reserved_evaluation_opened=False,models_newly_trained=['cohd_counts','context_grasp'],api_calls=0,
        protocol_sha256=sha(a.protocol),torch_version=torch.__version__,
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/code_eval/formal_sources.py',ROOT/'src/code_eval/context_grasp.py',ROOT/'src/code_eval/context_encoder.py',ROOT/'src/code_eval/grasp_adapter.py']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('artifacts/lantern_development_v1'))
    p.add_argument('--probe',type=Path,default=Path('artifacts/encoder_probe_v1'))
    p.add_argument('--protocol',type=Path,default=ROOT/'configs/formal_result_protocol_20261006.json')
    p.add_argument('--out',type=Path,default=Path('artifacts/formal_sources_v1'));p.add_argument('--threads',type=int,default=8)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);result={};start=time.time()
    try:
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-p','test_formal_sources.py','-v'],check=True)
        result=run(a)
    except Exception:result=dict(status='failed',error=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.time()-start;result['output_sha256']={f.name:sha(f) for f in a.out.iterdir() if f.is_file()}
        dump(a.out/'run_manifest.json',result)
        archive=a.out.with_name(a.out.name+'_reports.tar.gz')
        with tarfile.open(archive,'w:gz') as t:
            for f in a.out.glob('*.json'):
                if f.name!='cohd_counts.json':t.add(f,arcname=f.name)
        print('Return report:',archive,flush=True)


if __name__=='__main__':main()
