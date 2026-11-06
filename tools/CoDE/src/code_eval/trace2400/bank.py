"""Frozen TRACE input adapters for unchanged CoDE predictor implementations."""
import json
import os
from collections import Counter,defaultdict
import numpy as np
from .encoding import TraceEncoding
from .math_policy import source_math
from .retrieval import TraceDenseEncoding,retrieval_groups

class TraceBank:
    def __init__(self,paths,dictionary,labels,train,queries,subjects,threads=4):
        import torch,lightgbm as lgb
        from ..context_encoder import ContextEncoder
        from ..context_grasp import ContextGRASP
        from ..cohort_stats import Counts
        from ..tabr_adapter import make_model
        self.torch=torch;torch.set_num_threads(threads);self.device='cuda:0'
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.labels=labels;self.encoding=TraceEncoding(dictionary)
        self.tree=lgb.Booster(model_file=str(paths['lightgbm']))
        raw=json.loads(paths['cohd'].read_text());v=raw['counts'];self.alpha=raw['alpha']
        self.co=Counts(v['labels']);self.co.n=v['n'];self.co.disease=Counter(v['disease']);self.co.feature=Counter(v['feature']);self.co.pair=defaultdict(Counter,{k:Counter(x) for k,x in v['pair'].items()})
        if self.co.labels!=labels:raise ValueError('Association axes mismatch')
        self.neural={}
        for name,cls in [('encoder',ContextEncoder),('grasp',ContextGRASP)]:
            ck=torch.load(paths[name],map_location='cpu',weights_only=True)
            if ck['labels']!=labels:raise ValueError('Neural axes mismatch')
            model=cls(len(ck['vocabulary']),len(labels),ck['hidden'],**({'centers':ck['centers']} if name=='grasp' else {})).to(self.device)
            model.load_state_dict(ck['state']);model.eval();self.neural[name]=(model,ck['vocabulary'])
        self.dense=TraceDenseEncoding(dictionary);self.components=retrieval_groups(train+queries,subjects)
        self.retrieval={};byid={r['patient_id']:r for r in train}
        for name in ['tabr','no_retrieval']:
            ck=torch.load(paths[name],map_location='cpu',weights_only=True)
            if ck['labels']!=labels or ck['encoding']!={'tokens':self.dense.tokens,'features':self.dense.features,'context':[]}:raise ValueError('TabR axes mismatch')
            bank=[byid[i] for i in ck['bank_admission_ids']]
            model=make_model(self.dense.width,len(labels),ck['config']['model']).to(self.device);model.load_state_dict(ck['model']);model.eval()
            bx=torch.tensor(np.stack([self.dense.encode(r['tokens'],{}) for r in bank]),device=self.device)
            by=torch.tensor([labels.index(r['label']) for r in bank],device=self.device)
            bp=torch.tensor([self.components[r['patient_id']] for r in bank],device=self.device)
            self.retrieval[name]=(model,bx,by,bp,ck['config']['context_size'])
    def predict(self,inputs,context,sources=None,patient_id=None,archive=False):
        if context:raise ValueError('Unexpected demographic context')
        names=list(sources) if sources is not None else ['lightgbm','cohd','encoder','grasp']
        torch=self.torch;result={}
        for name in names:
            if name=='lightgbm':out=list(self.tree.predict(np.stack([self.encoding.encode(t,{}) for t in inputs]),num_threads=1))
            elif name=='cohd':
                out=[np.array([x['support'] for x in self.co.score(t,self.alpha)['scores']]) if all(self.co.feature[x]>0 for x in t) else None for t in inputs]
            elif name in self.neural:
                model,vocab=self.neural[name];out=[None]*len(inputs);valid=[i for i,t in enumerate(inputs) if all(x in vocab for x in t)]
                batch=256 if name=='encoder' else 128
                for start in range(0,len(valid),batch):
                    ix=valid[start:start+batch];x=torch.zeros((len(ix),max(1,max(len(inputs[i]) for i in ix))),dtype=torch.long,device=self.device)
                    for j,i in enumerate(ix):
                        if inputs[i]:x[j,:len(inputs[i])]=torch.tensor([vocab[t] for t in inputs[i]],device=self.device)
                    with source_math(torch,name,archive=archive),torch.no_grad():pr=model(x,torch.zeros((len(ix),2),device=self.device)).softmax(1).cpu().numpy()
                    for i,p in zip(ix,pr):out[i]=p
            elif name in self.retrieval:
                identities=patient_id if isinstance(patient_id,list) else [patient_id]*len(inputs)
                if len(identities)!=len(inputs) or any(p not in self.components for p in identities):raise ValueError('Query exclusion identity missing')
                model,bx,by,bp,k=self.retrieval[name];components=[self.components[p] for p in identities]
                if any(int((bp!=c).sum())<k for c in set(components)):raise ValueError('Insufficient eligible retrieval bank')
                out=[]
                mode=os.environ.get('CODE_TRACE_TABR_QUERY_MODE','legacy')
                if mode not in ['legacy','single']:raise ValueError('Unknown TabR query policy')
                batch=1 if name=='tabr' and mode=='single' else 128
                for start in range(0,len(inputs),batch):
                    ts=inputs[start:start+batch];xx=torch.tensor(np.stack([self.dense.encode(t,{}) for t in ts]),device=self.device);qp=torch.tensor(components[start:start+batch],dtype=torch.long,device=self.device)
                    with source_math(torch,name,archive=archive),torch.no_grad():out.extend(model(xx,bx,by,qp,bp,k,use_retrieval=name=='tabr').softmax(1).cpu().numpy())
            else:raise ValueError('Unknown source '+name)
            result[name]=[]
            for p in out:
                if p is not None:
                    p=np.asarray(p,dtype=float)
                    if p.shape!=(len(self.labels),) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-5):raise ValueError('Invalid source probabilities')
                result[name].append(p)
        return result

class RetrievalView:
    def __init__(self,bank):self.bank=bank
    def predict(self,inputs,r):return self.bank.predict(inputs,{},['tabr','no_retrieval'],r['patient_id'])
