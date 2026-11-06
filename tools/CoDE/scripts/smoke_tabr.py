"""Run CPU/GPU forward, gradient and isolation checks before any real training."""
import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def smoke(device):
    import torch
    import numpy as np
    from code_eval.tabr_adapter import make_model,DenseEncoding
    from code_eval.tabr_official_core import Model
    torch.manual_seed(17)
    cfg=dict(num_embeddings=None,d_main=12,d_multiplier=2.,encoder_n_blocks=0,
             predictor_n_blocks=1,mixer_normalization='auto',context_dropout=0.,
             dropout0=0.,dropout1=0.,normalization='LayerNorm',activation='ReLU')
    ref=Model(n_num_features=7,n_bin_features=0,cat_cardinalities=[],n_classes=3,**cfg).to(device)
    model=make_model(7,3,cfg).to(device);model.load_state_dict(ref.state_dict())
    ref.eval();model.eval()
    q=torch.randn(4,7,device=device); bank=torch.randn(13,7,device=device)
    labels=torch.arange(13,device=device)%3
    qp=torch.full((4,),-1,device=device);bp=torch.arange(13,device=device)
    expected=ref(x_={'num':q},y=None,candidate_x_={'num':bank},candidate_y=labels,context_size=5,is_train=False)
    actual=model(q,bank,labels,qp,bp,5)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    expected.square().sum().backward();actual.square().sum().backward()
    for (name,p),(name2,p2) in zip(ref.named_parameters(),model.named_parameters()):
        assert name==name2
        if p.grad is None:assert p2.grad is None
        else:torch.testing.assert_close(p.grad,p2.grad,rtol=2e-4,atol=2e-5)
    for name in ['K.weight','T.0.weight','label_encoder.0.weight']:
        assert dict(model.named_parameters())[name].grad.abs().sum()>0,name
    # Same-pattern bank labels, including duplicate views, cannot affect output.
    duplicate_bp=bp.clone();duplicate_bp[1]=0
    self_q=bank[:1];self_pattern=bp[:1]
    before=model(self_q,bank,labels,self_pattern,duplicate_bp,5)
    changed=labels.clone();changed[:2]=(changed[:2]+1)%3
    after=model(self_q,bank,changed,self_pattern,duplicate_bp,5)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    perm=torch.arange(12,-1,-1,device=device)
    torch.testing.assert_close(actual,model(q,bank[perm],labels[perm],qp,bp[perm],5),rtol=2e-5,atol=2e-6)
    dictionary={'features':{'b':{'type':'B','group_id':'g'},'c':{'type':'C','values':{'1':{},'2':{}},'group_id':'g'},'m':{'type':'M','values':{'a':{},'b':{}},'group_id':'h'}}}
    ctx={'age':40,'sex':'M'};enc=DenseEncoding(dictionary,[{'context':ctx}])
    zero=enc.encode([],ctx);negative=enc.encode(['b_@_false'],ctx);positive=enc.encode(['b'],ctx)
    assert not np.array_equal(zero,negative) and not np.array_equal(negative,positive)
    one=enc.encode(['m_@_a'],ctx);two=enc.encode(['m_@_a','m_@_b'],ctx)
    assert np.count_nonzero(one!=two)==1
    # Tiny backward/update is checked independently of official equivalence.
    opt=torch.optim.AdamW(model.parameters(),lr=.001);opt.zero_grad()
    loss=torch.nn.functional.cross_entropy(model(q,bank,labels,qp,bp,5),torch.tensor([0,1,2,0],device=device))
    loss.backward();opt.step(); assert torch.isfinite(loss)
    return dict(status='passed',device=str(device),torch=torch.__version__,
                official_forward_max_abs_difference=float((actual-expected).detach().abs().max()),
                checks=['official forward equality with exact torch reference search',
                        'official parameter-gradient equality','K/T/label embedding nonzero gradients',
                        'all same-pattern candidates excluded','candidate-order invariance without ties',
                        'unknown versus explicit false','multi-value preservation','backward optimizer step'],
                limitation='Does not verify FAISS runtime or clinical validity')

if __name__=='__main__':
    import torch
    device=torch.device(sys.argv[1] if len(sys.argv)>1 else 'cpu')
    print(json.dumps(smoke(device),indent=2))
