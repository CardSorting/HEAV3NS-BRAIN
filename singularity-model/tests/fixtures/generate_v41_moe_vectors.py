"""PyTorch CPU routing and unquantized-activation FP4 expert reference vectors.
Args: raw layer 6 gate BF16 weights and the consecutive F32 text/VL biases.
"""
import hashlib,json,sys,torch
import torch.nn.functional as F
torch.set_num_threads(1)
wbytes=open(sys.argv[1],'rb').read(); bbytes=open(sys.argv[2],'rb').read()
assert len(wbytes)==384*5120*2 and len(bbytes)==384*2*4
w=torch.frombuffer(bytearray(wbytes),dtype=torch.bfloat16).reshape(384,5120).float()
b=torch.frombuffer(bytearray(bbytes),dtype=torch.float32).reshape(2,384)
x=torch.tensor([(i%97-48)/64 for i in range(5120)],dtype=torch.float32)
scores=F.softplus(w@x).sqrt()
def routes(bias):
    idx=(scores+bias).topk(6).indices
    selected=scores[idx]; weights=selected/(selected.sum()+1e-20)*1.5
    return dict(ids=idx.tolist(),weights=weights.tolist())
# Small expert uses each packed byte's low nibble first and scale 2^-2.
table=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
def matrix(seed):
    packed=torch.tensor([(i*19+seed)%256 for i in range(32*16)],dtype=torch.uint8).reshape(32,16)
    return torch.stack([table[(packed&15).long()],table[(packed>>4).long()]],-1).reshape(32,32)*.25
w1,w2,w3=[matrix(i) for i in (7,13,23)]
xx=torch.tensor([(i%9-4)/8 for i in range(32)],dtype=torch.bfloat16)
gate=(w1@xx.float()).to(torch.bfloat16).float().clamp(max=10)
up=(w3@xx.float()).to(torch.bfloat16).float().clamp(-10,10)
hidden=(F.silu(gate)*up*.7).to(torch.bfloat16)
result=(w2@hidden.float()).to(torch.bfloat16).float()
# Two identical routed experts selected with weights 0.75; shared expert is zero.
hidden_two=(F.silu(gate)*up*.75).to(torch.bfloat16)
each=(w2@hidden_two.float()).to(torch.bfloat16).float()
combined=(each+each).to(torch.bfloat16).float()
print(json.dumps(dict(weight_sha256=hashlib.sha256(wbytes).hexdigest(),bias_sha256=hashlib.sha256(bbytes).hexdigest(),text=routes(b[0]),image=routes(b[1]),expert=result.tolist(),combined=combined.tolist())))
