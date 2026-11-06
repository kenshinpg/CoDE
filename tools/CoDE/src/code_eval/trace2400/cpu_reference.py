"""Frozen-parameter CPU float64 reference for review sensitivity only."""
import numpy as np
from .retrieval import TraceDenseEncoding,retrieval_groups
SOURCES=['grasp','tabr','no_retrieval']
METHODS=[m+'_'+s for m in SOURCES for s in ['coverage','absolute','retained']]
POLICY='CPU_float64_single_thread'
TOL=1e-10

def coverage_reference(probabilities,full,minus,groups,cited,target):
    from run_coverage_controls import coverage
    fp=probabilities[full];cp=probabilities[cited]
    weights=({g:max(0.,float(fp[target]-probabilities[k][target])) for g,k in minus.items()}
             if fp is not None and all(probabilities[k] is not None for k in minus.values()) else None)
    return dict(coverage=coverage(weights,groups,set(cited)) if weights is not None else None,
        absolute=float(cp[target]) if cp is not None else None,
        retained=float(cp[target]-fp[target]) if cp is not None and fp is not None else None),sum(weights.values()) if weights is not None else None

def probability_delta(a,b):
    if (a is None)!=(b is None):raise ValueError('CPU probability availability changed across repeats')
    return float(np.max(np.abs(a-b))) if a is not None else 0.

class CPUReference:
    def __init__(self,paths,dictionary,labels,train,queries,subjects):
        import torch
        from ..context_grasp import ContextGRASP
        from ..tabr_adapter import make_model
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.torch=torch;self.labels=labels;self.dense=TraceDenseEncoding(dictionary)
        self.components=retrieval_groups(train+queries,subjects)
        ck=torch.load(paths['grasp'],map_location='cpu',weights_only=True)
        if ck['labels']!=labels:raise ValueError('GRASP label axes changed')
        self.grasp=ContextGRASP(len(ck['vocabulary']),len(labels),ck['hidden'],centers=ck['centers'])
        self.grasp.load_state_dict(ck['state']);self.grasp.double().eval();self.vocabulary=ck['vocabulary']
        self.models={};self.bank_cached={};self.cache_enabled=True
        byid={r['patient_id']:r for r in train}
        for name in ['tabr','no_retrieval']:
            ck=torch.load(paths[name],map_location='cpu',weights_only=True)
            if ck['labels']!=labels or ck['encoding']!={'tokens':self.dense.tokens,'features':self.dense.features,'context':[]}:
                raise ValueError('Retrieval axes changed')
            bank=[byid[x] for x in ck['bank_admission_ids']]
            model=make_model(self.dense.width,len(labels),ck['config']['model'])
            model.load_state_dict(ck['model']);model.double().eval()
            bx=torch.tensor(np.stack([self.dense.encode(r['tokens'],{}) for r in bank]),dtype=torch.float64)
            by=torch.tensor([labels.index(r['label']) for r in bank],dtype=torch.long)
            bp=torch.tensor([self.components[r['patient_id']] for r in bank],dtype=torch.long)
            self.models[name]=(model,bx,by,bp,ck['config']['context_size'])
            # The frozen bank tensor and eval-mode parameters are constant across queries.
            # Check cached/uncached parity separately before accepting this optimization.
            if name=='tabr':
                native=model._encode
                with torch.no_grad():cached=native({'num':bx})
                def encode(x,native=native,bx=bx,cached=cached):
                    return cached if self.cache_enabled and x['num'] is bx else native(x)
                model._encode=encode

    def predict(self,tokens,pid):
        torch=self.torch;out={}
        with torch.no_grad():
            if all(t in self.vocabulary for t in tokens):
                x=torch.zeros((1,max(1,len(tokens))),dtype=torch.long)
                if tokens:x[0,:len(tokens)]=torch.tensor([self.vocabulary[t] for t in tokens])
                out['grasp']=self.grasp(x,torch.zeros((1,2),dtype=torch.float64)).softmax(1)[0].numpy()
            else:out['grasp']=None  # Same source-specific unavailability as the frozen TraceBank.
            xx=torch.tensor(self.dense.encode(tokens,{})[None],dtype=torch.float64)
            for name,(model,bx,by,bp,k) in self.models.items():
                qp=torch.tensor([self.components[pid]],dtype=torch.long)
                out[name]=model(xx,bx,by,qp,bp,k,use_retrieval=name=='tabr').softmax(1)[0].numpy()
        for p in out.values():
            if p is None:continue
            if p.shape!=(len(self.labels),) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-10,rtol=0):
                raise ValueError('Invalid CPU reference probabilities')
        return out

    def check_cache(self,tokens,pid):
        self.cache_enabled=False
        try:a=self.predict(tokens,pid)
        finally:self.cache_enabled=True
        b=self.predict(tokens,pid)
        delta=max(probability_delta(a[m],b[m]) for m in SOURCES)
        if delta>TOL:raise ValueError('CPU cached/uncached parity failed')
        return delta
