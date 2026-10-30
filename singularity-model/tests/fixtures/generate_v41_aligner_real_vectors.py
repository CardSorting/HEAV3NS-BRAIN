"""PyTorch reference for the pinned checkpoint aligner using synthetic features."""
import hashlib,json,sys,torch
import torch.nn.functional as F

torch.set_num_threads(1)
raw=open(sys.argv[1],'rb').read()
digest=hashlib.sha256(raw).hexdigest()
assert digest=='5929ee263ea80174c86266f366ef2d5b3a935b4ee313fe5daa19afa7f453b2e4'
def view(a,b,shape): return torch.frombuffer(bytearray(raw[a:b]),dtype=torch.bfloat16).reshape(shape)
b1=view(0,10240,(5120,));w1=view(10240,94382080,(5120,9216))
b2=view(94382080,94392320,(5120,));w2=view(94392320,146821120,(5120,5120))
features=torch.tensor([((i*17)%97-48)/64 for i in range(3*3*1024)],dtype=torch.bfloat16).reshape(3,3,1024)
unfold=F.unfold(features.permute(2,0,1).unsqueeze(0),3,stride=3).squeeze(0).transpose(0,1)
out=F.linear(F.gelu(F.linear(unfold,w1,b1)),w2,b2).float().flatten()
print(json.dumps({'sha256':digest,'output':out.tolist()}))
