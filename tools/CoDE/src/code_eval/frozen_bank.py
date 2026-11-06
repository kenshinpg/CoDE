"""Read-only four-source prediction API; no fitting or query-time adaptation."""
import numpy as np
from .formal_sources import AssociationSource


class FrozenBank:
    def __init__(self,paths,encoding,labels,threads=8):
        import torch
        import lightgbm as lgb
        from .context_encoder import ContextEncoder
        from .context_grasp import ContextGRASP
        import json
        self.torch=torch;torch.set_num_threads(threads)
        self.labels=labels;self.encoding=encoding
        self.tree=lgb.Booster(model_file=str(paths['lightgbm']))
        self.co=AssociationSource.restore(json.loads(paths['cohd'].read_text(encoding='utf8')))
        if self.co.counts.labels!=labels:raise ValueError('Counts label axes mismatch')
        self.neural={}
        for name,cls in [('encoder',ContextEncoder),('grasp',ContextGRASP)]:
            ck=torch.load(paths[name],map_location='cpu',weights_only=True)
            if ck['labels']!=labels:raise ValueError('Neural label axes mismatch')
            kwargs={'centers':ck['centers']} if name=='grasp' else {}
            model=cls(len(ck['vocabulary']),len(labels),ck['hidden'],**kwargs)
            model.load_state_dict(ck['state']);model.eval();self.neural[name]=(model,ck)

    def predict(self,inputs,context,sources=None):
        torch=self.torch
        sources=set(sources) if sources is not None else {'lightgbm','cohd','encoder','grasp'}
        if not sources<={'lightgbm','cohd','encoder','grasp'}:raise ValueError('Unknown prediction source')
        result={}
        if 'lightgbm' in sources:result['lightgbm']=list(self.tree.predict(np.stack([self.encoding.encode(ts,context) for ts in inputs]),num_threads=1))
        if 'cohd' in sources:result['cohd']=[self.co.distribution(ts,context) for ts in inputs]
        for name,(model,ck) in self.neural.items():
            if name not in sources:continue
            vocab=ck['vocabulary'];out=[None]*len(inputs)
            valid=[i for i,ts in enumerate(inputs) if all(t in vocab for t in ts)]
            for start in range(0,len(valid),256):
                ids=valid[start:start+256]
                x=torch.zeros((len(ids),max(1,max(len(inputs[i]) for i in ids))),dtype=torch.long)
                for j,i in enumerate(ids):x[j,:len(inputs[i])]=torch.tensor([vocab[t] for t in inputs[i]],dtype=torch.long)
                h=torch.tensor([[(float(context['age'])-ck['age_mean'])/ck['age_std'],float(context['sex']=='M')]]*len(ids),dtype=torch.float32)
                with torch.no_grad():p=model(x,h).softmax(1).numpy()
                for i,v in zip(ids,p):out[i]=v
            result[name]=out
        for name,values in result.items():
            for i,p in enumerate(values):
                if p is not None:
                    p=np.asarray(p,dtype=float)
                    if len(p)!=len(self.labels) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-5):raise ValueError('Invalid probabilities: '+name)
                    values[i]=p
        return result


class PatientCache:
    def __init__(self,bank,tokens,context):
        self.bank=bank;self.tokens=tokens;self.context=context
        self.cache={name:{} for name in ['lightgbm','encoder','cohd','grasp']}

    def ensure(self,sets):
        new=sorted({tuple(sorted(fs)) for fs in sets}-self.cache['lightgbm'].keys())
        if not new:return
        if any(not set(fs)<=self.tokens.keys() for fs in new):raise ValueError('Nonpatient facts')
        values=self.bank.predict([[self.tokens[f] for f in fs] for fs in new],self.context)
        for name,ps in values.items():self.cache[name].update(zip(new,ps))

    def get(self,name,fs):return self.cache[name][tuple(sorted(fs))]
