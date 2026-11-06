"""Minimal repair: row-normalize graph aggregation; retain every other v0.1 mechanism."""
from .grasp_adapter import Teacher

CONDITIONS=('encoder','prototype_identity','graph_legacy','graph_row_normalized')


class RepairTeacher(Teacher):
    def __init__(self,vocabulary,classes,hidden=128,centers=32,condition='graph_row_normalized'):
        if condition not in CONDITIONS:raise ValueError('Unknown condition')
        variant='encoder' if condition=='encoder' else 'identity' if condition=='prototype_identity' else 'grasp'
        super().__init__(vocabulary,classes,hidden,centers,variant)
        self.condition=condition

    def fit_centers(self,*args,**kwargs):
        super().fit_centers(*args,**kwargs)
        if self.condition=='graph_row_normalized':
            # Directed row mean, no added self-loops, no gate clamp or branch LayerNorm.
            self.adj.copy_(self.adj/self.adj.sum(1,keepdim=True).clamp_min(1))
