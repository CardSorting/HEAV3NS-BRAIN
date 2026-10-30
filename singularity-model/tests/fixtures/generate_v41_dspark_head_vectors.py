"""Independent PyTorch CPU DSpark draft-head vector."""
import json,torch
h=torch.tensor([[1.,2.],[-1.,3.]],dtype=torch.bfloat16)
norm=torch.tensor([1.,1.],dtype=torch.bfloat16)
head=torch.tensor([[1.,0.],[0.,1.],[-1.,-1.]],dtype=torch.bfloat16).float()
emb=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.bfloat16)
markov=torch.tensor([[0.,1.],[1.,0.],[-1.,-1.]],dtype=torch.bfloat16).float()
conf=torch.tensor([1.,2.,3.,4.],dtype=torch.bfloat16).float()
tokens=[0]; logits=[]; confidence=[]
for x in h:
    normalized=(x.float()*torch.rsqrt(x.float().square().mean()+1e-6)*norm.float()).to(torch.bfloat16).float()
    e=emb[tokens[-1]].float(); score=head@normalized+markov@e
    tokens.append(int(score.argmax())); logits.append(score.tolist()); confidence.append(float(torch.cat([x.float(),e])@conf))
print(json.dumps({'tokens':tokens,'logits':logits,'confidence':confidence}))
