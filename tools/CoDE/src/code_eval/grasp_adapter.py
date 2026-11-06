"""Independent GRASP-inspired implementation for single-visit evidence sets.

Mechanism adaptation, not the original PhysioNet model. No upstream source copied.
Centers are fitted only on fitting-partition patients and frozen for inference.
"""
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def kmeans(x, k, iterations, seed):
    if not 1<k<=len(x):raise ValueError('Require at least K fitting embeddings')
    gen=torch.Generator(device=x.device).manual_seed(seed)
    centers=x[torch.randperm(len(x),generator=gen,device=x.device)[:k]].clone()
    for _ in range(iterations):
        distances=(x.square().sum(1,keepdim=True)+centers.square().sum(1)[None,:]-2*x@centers.T)
        assignments=distances.argmin(1)
        updated=torch.zeros_like(centers);updated.index_add_(0,assignments,x)
        counts=torch.bincount(assignments,minlength=k)
        # Retain previous center for empty clusters; never use query data.
        updated=torch.where(counts[:,None]>0,updated/counts.clamp_min(1)[:,None],centers)
        if torch.allclose(updated,centers,rtol=0,atol=1e-5):
            centers=updated;break
        centers=updated
    return centers


@torch.no_grad()
def adjacency(centers, neighbors, identity=False):
    k=len(centers)
    if identity:return torch.eye(k,device=centers.device)
    if not 0<neighbors<k:raise ValueError('neighbors must be less than centers')
    distances=torch.cdist(centers,centers);distances.fill_diagonal_(float('inf'))
    # Stable ties by center index; directed, no self edges, unnormalized as source.
    nearest=torch.argsort(distances,dim=1,stable=True)[:,:neighbors]
    return torch.zeros_like(distances).scatter_(1,nearest,1.)


class Teacher(nn.Module):
    def __init__(self, vocabulary, classes, hidden=128, centers=32, variant='grasp'):
        super().__init__()
        if variant not in ['encoder','grasp','identity']:raise ValueError('Unknown variant')
        self.variant=variant
        self.embedding=nn.Embedding(vocabulary+1,hidden,padding_idx=0)
        self.encoder=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.LayerNorm(hidden))
        self.output_norm=nn.LayerNorm(hidden)
        self.head=nn.Linear(hidden,classes)
        self.graph1=nn.Linear(hidden,hidden);self.graph2=nn.Linear(hidden,hidden)
        self.gate_graph=nn.Linear(hidden,1);self.gate_patient=nn.Linear(hidden,1)
        self.register_buffer('centers',torch.zeros(centers,hidden))
        self.register_buffer('adj',torch.eye(centers))
        self.register_buffer('centers_ready',torch.tensor(False))

    def encode(self, tokens):
        observed=(tokens!=0)
        mean=(self.embedding(tokens)*observed[:,:,None]).sum(1)/observed.sum(1,keepdim=True).clamp_min(1)
        return self.encoder(mean)

    def forward(self,tokens,use_graph=True):
        h=self.encode(tokens)
        if self.variant!='encoder' and use_graph:
            if not self.centers_ready.item():raise ValueError('Fit centers before graph inference')
            logits=F.relu(h@self.centers.T)
            weights=F.gumbel_softmax(logits,tau=1.,hard=True) if self.training else F.one_hot(logits.argmax(1),len(self.centers)).to(h.dtype)
            # Bias follows aggregation, matching A(XW)+b rather than A(XW+b).
            c=F.relu(self.adj@(self.centers@self.graph1.weight.T)+self.graph1.bias)
            c=F.relu(self.adj@(c@self.graph2.weight.T)+self.graph2.bias)
            graph=weights@c
            a=torch.sigmoid(self.gate_graph(graph));b=torch.sigmoid(self.gate_patient(h))
            gate=a/(a+b).clamp_min(1e-8)
            h=gate*graph+(1-gate)*h
        return self.head(self.output_norm(h))

    @torch.no_grad()
    def fit_centers(self, embeddings, neighbors=20, iterations=20, seed=260930, identity=False):
        centers=kmeans(embeddings,len(self.centers),iterations,seed)
        self.centers.copy_(centers)
        self.adj.copy_(adjacency(centers,neighbors,identity or self.variant=='identity'))
        self.centers_ready.fill_(True)
