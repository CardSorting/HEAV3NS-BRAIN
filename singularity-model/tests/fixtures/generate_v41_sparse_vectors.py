"""PyTorch CPU reference for V4.1 sparse attention/indexing and pooling."""
import json, torch
import torch.nn.functional as F
torch.set_num_threads(1)
q=torch.tensor([[(i*3+j-5)/8 for j in range(4)] for i in range(3)],dtype=torch.float32)
kv=torch.tensor([[(i*2-j-4)/8 for j in range(4)] for i in range(8)],dtype=torch.float32)
w=torch.tensor([.25,-.5,.75]); sink=torch.tensor([-.5,.25,1.])
index=(torch.einsum('hd,td->ht',q,kv).relu()*w[:,None]).sum(0)
ids=torch.tensor([0,3,5,7]); logits=(q@kv[ids].T)*.5
probs=torch.cat([logits,sink[:,None]],dim=-1).softmax(-1)[:,:-1]
attn=(probs@kv[ids]).to(torch.bfloat16).float()
# Pool two full groups and leave one pending token.
values=torch.tensor([[(i*3+j-5)/8 for j in range(4)] for i in range(5)])
gates=torch.tensor([[(i-j*2)/4 for j in range(4)] for i in range(5)])
pool=(values[:4].reshape(2,2,4)*gates[:4].reshape(2,2,4).softmax(dim=1)).sum(dim=1)
print(json.dumps(dict(index=index.tolist(),attention=attn.flatten().tolist(),pool=pool.tolist())))
