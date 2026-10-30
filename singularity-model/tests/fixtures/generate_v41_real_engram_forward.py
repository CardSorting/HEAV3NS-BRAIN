"""CPU reference for layer-14 Engram using four real rows repeated to 24 columns.
Args: row bytes, row scales, q/k weights, projection bytes, projection scales.
Uses synthetic residual, repeated IDs and FP8 activation emulation; not a real prompt.
"""
import hashlib,json,sys,torch

torch.set_num_threads(1)
paths=sys.argv[1:]
assert len(paths)==5
raw=[open(p,'rb').read() for p in paths]
rows, row_scales, qk, weights, scales = raw
assert list(map(len,raw))==[1024,32,81920,157286400,153600]
emb=torch.frombuffer(bytearray(rows),dtype=torch.uint8).view(torch.float8_e4m3fn).float().reshape(4,256)
es=torch.frombuffer(bytearray(row_scales),dtype=torch.uint8).view(torch.float8_e8m0fnu).float().reshape(4,8)
emb=(emb*es.repeat_interleave(32,1)).to(torch.bfloat16).float().repeat(6,1).flatten()
a=emb.reshape(192,32)
act_scale=(a.abs().amax(-1).clamp_min(1e-4)/448).log2().ceil().exp2()
act=(a/act_scale[:,None]).clamp(-448,448).to(torch.float8_e4m3fn).float()*act_scale[:,None]
w=torch.frombuffer(bytearray(weights),dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(800,32,192,32)
s=torch.frombuffer(bytearray(scales),dtype=torch.uint8).view(torch.float8_e8m0fnu).float().reshape(800,192)
parts=[]
for i in range(800):
    parts.append(((w[i].float()*act[None,:,:]).sum(-1)*s[i][None,:]).sum(-1))
p=torch.cat(parts).to(torch.bfloat16).float(); key=p[:20480].reshape(4,5120); value=p[20480:]
qk_t=torch.frombuffer(bytearray(qk),dtype=torch.bfloat16).float().reshape(2,4,5120)
h=torch.tensor([(i%97-48)/64 for i in range(20480)],dtype=torch.bfloat16).float().reshape(4,5120)
rstd=torch.rsqrt(h.square().mean(-1)+1e-20)*torch.rsqrt(key.square().mean(-1)+1e-20)
dot=(h*(qk_t[0]*qk_t[1])*key).sum(-1)*rstd*5120**-.5
gate=torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(),dot))
out=(h+gate[:,None]*value).to(torch.bfloat16).float()
print(json.dumps(dict(hashes=[hashlib.sha256(b).hexdigest() for b in raw],output=out.flatten().tolist())))
