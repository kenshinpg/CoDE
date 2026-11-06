"""GRASP-inspired static-set adaptation atop the identical context Encoder."""
import torch
from torch import nn
from torch.nn import functional as F
from .context_encoder import ContextEncoder
from .grasp_adapter import kmeans, adjacency


class ContextGRASP(ContextEncoder):
    def __init__(self,vocabulary,classes,hidden=128,centers=32):
        super().__init__(vocabulary,classes,hidden)
        self.graph1=nn.Linear(hidden,hidden);self.graph2=nn.Linear(hidden,hidden)
        self.gate_graph=nn.Linear(hidden,1);self.gate_patient=nn.Linear(hidden,1)
        self.register_buffer('centers',torch.zeros(centers,hidden))
        self.register_buffer('adj',torch.eye(centers))
        self.register_buffer('centers_ready',torch.tensor(False))

    def encode(self,tokens,context):
        observed=tokens!=0
        mean=(self.embedding(tokens)*observed[:,:,None]).sum(1)/observed.sum(1,keepdim=True).clamp_min(1)
        return self.encoder(mean+self.context(context))

    def propagated(self):
        c=F.relu(self.adj@(self.centers@self.graph1.weight.T)+self.graph1.bias)
        return F.relu(self.adj@(c@self.graph2.weight.T)+self.graph2.bias)

    def forward(self,tokens,context,use_graph=True,shift_assignment=False,return_audit=False):
        h=self.encode(tokens,context);patient=h
        if not use_graph:return self.head(self.output_norm(h))
        if not self.centers_ready.item():raise ValueError('Centers not fitted')
        logits=F.relu(h@self.centers.T)
        w=F.gumbel_softmax(logits,tau=1.,hard=True) if self.training else F.one_hot(logits.argmax(1),len(self.centers)).to(h.dtype)
        if shift_assignment:w=torch.roll(w,1,dims=1)
        graph=w@self.propagated()
        a=torch.sigmoid(self.gate_graph(graph));b=torch.sigmoid(self.gate_patient(h));gate=a/(a+b).clamp_min(1e-8)
        mixed=gate*graph+(1-gate)*patient;out=self.head(self.output_norm(mixed))
        if return_audit:return out,dict(assignment=w.argmax(1),gate=gate,representation_change=(mixed-patient).norm(dim=1))
        return out

    @torch.no_grad()
    def fit_centers(self,embeddings,seed,neighbors=20,identity=False):
        self.centers.copy_(kmeans(embeddings,len(self.centers),20,seed))
        adj=adjacency(self.centers,neighbors,identity)
        self.adj.copy_(adj/adj.sum(1,keepdim=True).clamp_min(1))
        self.centers_ready.fill_(True)
