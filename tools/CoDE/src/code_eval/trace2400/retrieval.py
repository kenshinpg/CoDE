"""TRACE input adaptation for the existing TabR core; no diagnostic model changes."""
import numpy as np
from .encoding import TraceEncoding
class TraceDenseEncoding:
    def __init__(self,dictionary):
        self.base=TraceEncoding(dictionary);self.features=sorted(self.base.features);self.tokens=[]
        for f in self.features:
            self.tokens.extend((f,v) for v in sorted(self.base.features[f]['values']))
        self.index={x:i for i,x in enumerate(self.tokens)}
        self.fi={f:len(self.tokens)+i for i,f in enumerate(self.features)}
        self.width=len(self.tokens)+len(self.features)
    def encode(self,tokens,context):
        self.base.encode(tokens,context);x=np.zeros(self.width,dtype=np.float32);x[len(self.tokens):]=1
        for t in set(tokens):
            f,v=t.split('_@_',1);x[self.index[(f,v)]]=1;x[self.fi[f]]=0
        return x

def retrieval_groups(records,subjects):
    """Exclude connected same-subject OR identical-pattern components, including masks."""
    parent={r['patient_id']:r['patient_id'] for r in records}
    def root(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    owners={}
    for r in sorted(records,key=lambda r:r['patient_id']):
        aid=r['patient_id']
        for key in [('subject',subjects[aid]),('pattern',r['pattern_id'])]:
            if key in owners:parent[root(aid)]=root(owners[key])
            else:owners[key]=aid
    names=sorted({root(a) for a in parent});mapping={g:i for i,g in enumerate(names)}
    return {a:mapping[root(a)] for a in parent}
