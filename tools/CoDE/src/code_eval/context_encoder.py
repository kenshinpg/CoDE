"""Evidence-set Encoder with observed tokens and exact fixed demographic context.

Same mean-embedding/MLP family as the historical baseline; demographics are
added before its encoder. This is a newly fitted model, not an old checkpoint.
"""
import torch
from torch import nn


class ContextEncoder(nn.Module):
    def __init__(self,vocabulary,classes,hidden=128):
        super().__init__()
        self.embedding=nn.Embedding(vocabulary+1,hidden,padding_idx=0)
        self.context=nn.Linear(2,hidden,bias=False)
        self.encoder=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.LayerNorm(hidden))
        self.output_norm=nn.LayerNorm(hidden)
        self.head=nn.Linear(hidden,classes)

    def forward(self,tokens,context):
        present=tokens!=0
        mean=(self.embedding(tokens)*present[:,:,None]).sum(1)/present.sum(1,keepdim=True).clamp_min(1)
        return self.head(self.output_norm(self.encoder(mean+self.context(context))))


def smoke():
    torch.manual_seed(5)
    model=ContextEncoder(6,3,16)
    x=torch.tensor([[1,2],[3,4],[5,6]]*8)
    context=torch.tensor([[-1.,0.],[0.,1.],[1.,0.]]*8)
    y=torch.tensor([0,1,2]*8)
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    before=float(nn.functional.cross_entropy(model(x,context),y).detach())
    for _ in range(25):
        opt.zero_grad();loss=nn.functional.cross_entropy(model(x,context),y);loss.backward()
        if not torch.isfinite(loss):raise ValueError('Nonfinite Encoder smoke loss')
        opt.step()
    after=float(nn.functional.cross_entropy(model(x,context),y).detach())
    context_gradient=float(model.context.weight.grad.norm())
    if not after<before or not context_gradient>0:raise ValueError('Encoder smoke failed')
    if not torch.isfinite(model(torch.zeros((3,1),dtype=torch.long),context[:3])).all():
        raise ValueError('Context-only forward failed')
    return dict(loss_before=before,loss_after=after,context_gradient=context_gradient)
