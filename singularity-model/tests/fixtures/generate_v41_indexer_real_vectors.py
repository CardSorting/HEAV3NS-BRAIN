"""PyTorch CPU indexer oracle using pinned layer-2 checkpoint byte ranges."""
import hashlib, json, math, sys
import torch

torch.set_num_threads(1)
paths = sys.argv[1:]
assert len(paths) == 3
raw = [open(p,'rb').read() for p in paths]
checksums = [hashlib.sha256(x).hexdigest() for x in raw]
assert checksums == ['28acb1dd040d4cf68be46b735e2e84ab14bccf6d1ad0bfeeb4011cb1e6fa20fc','63f7c8035fc839273c2db3daa8ba94fcad1f6e321ff8e5b4d82b6e3a8d682631','316fe279230fdfde74c744cbaf7416f34ba913a3c7aeaa72a2f57ae8a4d189c5']
def tensor(data, dtype): return torch.frombuffer(bytearray(data), dtype=dtype)
k_norm = tensor(raw[0][:256],torch.bfloat16).float()
wproj = tensor(raw[0][256:327936],torch.bfloat16).reshape(32,5120)
wk = tensor(raw[0][327936:],torch.bfloat16).reshape(128,512)
scales = tensor(raw[1],torch.uint8).view(torch.float8_e8m0fnu).float().reshape(128,40)
weights = tensor(raw[2],torch.uint8).view(torch.float8_e4m3fn).float().reshape(4096,1280)
rot_dim, base, orig = 64, 160000.0, 65536
freq = 1.0 / base ** (torch.arange(0,rot_dim,2,dtype=torch.float32)/rot_dim)
def correction(rot): return rot_dim*math.log(orig/(rot*2*math.pi))/(2*math.log(base))
low=max(math.floor(correction(32)),0); high=min(math.ceil(correction(1)),rot_dim-1)
ramp=((torch.arange(rot_dim//2)-low)/max(high-low,1e-3)).clamp(0,1)
freq = freq/16*ramp+freq*(1-ramp)
def rope(x,pos):
    z=x.float().clone(); tail=z[..., -64:].reshape(*z.shape[:-1],32,2); a=pos*freq
    rotated=torch.stack((tail[...,0]*a.cos()-tail[...,1]*a.sin(),tail[...,0]*a.sin()+tail[...,1]*a.cos()),-1)
    z[..., -64:]=rotated.flatten(-2); return z.to(torch.bfloat16)
def fp4(x):
    x=x.float().reshape(-1,32); scale=torch.pow(2.0,torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(6*2**-126)/6)))
    values=(x/scale[:,None]).abs().clamp(max=6)
    levels=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.]); dist=(values[...,None]-levels).abs()
    # equal-distance ties choose the even encoded nibble
    parity=torch.tensor([0,1,0,1,0,1,0,1]); best=dist.min(-1,keepdim=True).values
    choice=torch.where(dist==best, parity, 2).argmin(-1)
    return (levels[choice]*torch.sign(x)*scale[:,None]).to(torch.bfloat16).reshape(-1,128)
keys=[]
for t in range(4):
    latent=torch.tensor([((i*17+t*19)%113-56)/64 for i in range(512)],dtype=torch.bfloat16)
    key=(latent.float() @ wk.float().T).to(torch.bfloat16).float()
    key=(key*torch.rsqrt(key.square().mean()+1e-20)*k_norm).to(torch.bfloat16)
    keys.append(fp4(rope(key,2*t)).flatten())
keys=torch.stack(keys)
qr=torch.tensor([((i*7)%103-51)/64 for i in range(1280)],dtype=torch.bfloat16)
x=qr.float().reshape(40,32); amax=x.abs().amax(-1).clamp_min(1e-4); asc=torch.pow(2.,torch.ceil(torch.log2(amax/448)))
qx=(x/asc[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).float()
w=weights.reshape(128,32,40,32); proj=torch.einsum('rbkc,kc->rbk',w,qx)*scales[:,None,:]*asc[None,None,:]
query=proj.sum(-1).reshape(32,128).to(torch.bfloat16)
query=fp4(rope(query,7))
input=torch.tensor([((i*11)%89-44)/64 for i in range(5120)],dtype=torch.bfloat16)
weights_proj=(input.float()@wproj.float().T).to(torch.bfloat16)*(128*32)**-0.5
scores=((query @ keys.T).relu()*weights_proj[:,None]).sum(0)
selected=scores.topk(2,sorted=False).indices.sort().values.tolist()
print(json.dumps({'sha256':checksums,'keys':keys.float().tolist(),'scores':scores.float().tolist(),'selected':selected}))
