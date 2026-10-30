"""PyTorch transcription of a tiny DSpark attention block's checkpoint formulas."""
import json, math, torch

torch.set_num_threads(1)
def quant(x):
    x=x.to(torch.bfloat16).float().reshape(-1,32)
    scale=2**torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(1e-4)/448))
    q=(x/scale[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).float()
    return (q*scale[:,None]).to(torch.bfloat16).flatten()
def linear(x):
    xq=quant(x).float(); return (xq@torch.full((32,32),.125)).to(torch.bfloat16)
def norm(x):
    x=x.float(); return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-20)).to(torch.bfloat16)
def rope(x,pos,inverse=False):
    y=x.float().clone(); angle=torch.tensor(float(pos)); s=angle.sin()*(-1 if inverse else 1); c=angle.cos(); a=y[-2].clone(); b=y[-1].clone(); y[-2]=a*c-b*s; y[-1]=a*s+b*c; return y.to(torch.bfloat16)
def kv(x,pos): return quant(rope(norm(linear(x)),pos)).reshape(32)
main=torch.ones(32,dtype=torch.bfloat16); drafts=[main,torch.zeros_like(main)]
cache=torch.stack([kv(main,0),kv(main,1)]+[kv(d,2+i) for i,d in enumerate(drafts)])
outputs=[]
for i,x in enumerate(drafts):
    q=rope(linear(norm(linear(x))),2+i).float(); scores=(cache.float()@q)/math.sqrt(32)
    maximum=scores.max(); p=(scores-maximum).exp(); denom=p.sum()+torch.exp(torch.tensor(22.)-maximum)
    out=((p.to(torch.bfloat16).float()@cache.float())/denom).to(torch.bfloat16)
    rotated=rope(out,2+i,True)
    grouped=(quant(rotated).float()@torch.full((32,32),.125)).to(torch.bfloat16)
    outputs.append(linear(grouped).float().flatten().tolist())
print(json.dumps({'outputs':outputs}))
