"""PyTorch CPU reference for a real layer-6 expert with unquantized activations.
Arguments: packed expert weights, scales; input and route weight match gate vector.
"""
import hashlib,json,sys,torch
import torch.nn.functional as F
torch.set_num_threads(1)
wraw=open(sys.argv[1],'rb').read(); sraw=open(sys.argv[2],'rb').read()
assert len(wraw)==3*5898240 and len(sraw)==3*368640
fp4=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
def matrix(index,rows,cols):
    wb=bytearray(wraw[index*5898240:(index+1)*5898240])
    sb=bytearray(sraw[index*368640:(index+1)*368640])
    packed=torch.frombuffer(wb,dtype=torch.uint8).reshape(rows,cols//2)
    values=torch.stack((fp4[(packed&15).long()],fp4[(packed>>4).long()]),-1).reshape(rows,cols)
    scales=torch.frombuffer(sb,dtype=torch.uint8).view(torch.float8_e8m0fnu).float().reshape(rows,cols//32)
    return values*scales.repeat_interleave(32,dim=1)
x=torch.tensor([(i%97-48)/64 for i in range(5120)],dtype=torch.bfloat16)
weight=0.2759096622467041
w1=matrix(0,2304,5120); w2=matrix(1,5120,2304); w3=matrix(2,2304,5120)
g=(w1@x.float()).to(torch.bfloat16).float().clamp(max=10)
u=(w3@x.float()).to(torch.bfloat16).float().clamp(-10,10)
h=(F.silu(g)*u*weight).to(torch.bfloat16)
y=(w2@h.float()).to(torch.bfloat16).float()
def quantized(x):
    x=x.to(torch.bfloat16).float().reshape(-1,32)
    scale=(x.abs().amax(dim=-1).clamp_min(1e-4)/448).log2().ceil().exp2()
    quant=(x/scale[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).float()
    return (quant*scale[:,None]).flatten()
qx=quantized(x)
qg=(w1@qx).to(torch.bfloat16).float().clamp(max=10)
qu=(w3@qx).to(torch.bfloat16).float().clamp(-10,10)
qh=(F.silu(qg)*qu*weight).to(torch.bfloat16)
qy=(w2@quantized(qh)).to(torch.bfloat16).float()
print(json.dumps(dict(weights_sha256=hashlib.sha256(wraw).hexdigest(),scales_sha256=hashlib.sha256(sraw).hexdigest(),route_weight=weight,output=y.tolist(),quantized_output=qy.tolist())))
