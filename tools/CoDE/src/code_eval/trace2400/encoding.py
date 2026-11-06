"""TRACE-only input adapter; reuse grouping, masking and token rules from CoDE."""
import numpy as np
from ..tree_subset import Encoding
class TraceEncoding(Encoding):
    def __init__(self,dictionary):
        super().__init__(dictionary)
        self.columns=self.columns[2:]
        self.categorical=[i-2 for i in self.categorical if i>=2]
        self.locations={k:v-2 for k,v in self.locations.items()}
    def encode(self,tokens,context=None):
        if context:raise ValueError('TRACE view has no verified demographic context')
        x=np.full(len(self.columns),np.nan,dtype=np.float32);used=set()
        for token in sorted(set(tokens)):
            col,val=self.token(token)
            if col in used and x[col]!=val:raise ValueError('Conflicting feature values')
            x[col]=val;used.add(col)
        return x
    def export(self):
        d=super().export();d['fixed_context']=[];d['adapter']='TraceEncoding';return d
