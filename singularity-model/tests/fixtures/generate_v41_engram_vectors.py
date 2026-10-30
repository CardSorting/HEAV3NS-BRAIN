"""PyTorch CPU Engram lookup and gate vectors from pinned shard-48 byte ranges."""
import hashlib,json,sys,torch
torch.set_num_threads(1)
wr=open(sys.argv[1],'rb').read(); sr=open(sys.argv[2],'rb').read(); qr=open(sys.argv[3],'rb').read()
assert (len(wr),len(sr),len(qr))==(1024,32,81920)
w=torch.frombuffer(bytearray(wr),dtype=torch.uint8).view(torch.float8_e4m3fn).float().reshape(4,256)
s=torch.frombuffer(bytearray(sr),dtype=torch.uint8).view(torch.float8_e8m0fnu).float().reshape(4,8)
embed=(w*s.repeat_interleave(32,1)).to(torch.bfloat16).float()
qk=torch.frombuffer(bytearray(qr),dtype=torch.bfloat16).float().reshape(2,4,5120)
h=torch.tensor([(i%97-48)/64 for i in range(4*5120)],dtype=torch.bfloat16).float().reshape(4,5120)
k=torch.tensor([(i%89-44)/64 for i in range(4*5120)],dtype=torch.bfloat16).float().reshape(4,5120)
v=torch.tensor([(i%37-18)/32 for i in range(5120)],dtype=torch.bfloat16).float()
rstd=torch.rsqrt(h.square().mean(-1)+1e-20)*torch.rsqrt(k.square().mean(-1)+1e-20)
dot=(h*(qk[0]*qk[1])*k).sum(-1)*rstd*5120**-.5
gate=torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(),dot))
out=(h+gate[:,None]*v).to(torch.bfloat16).float()
print(json.dumps(dict(weight_sha=hashlib.sha256(wr).hexdigest(),scale_sha=hashlib.sha256(sr).hexdigest(),qk_sha=hashlib.sha256(qr).hexdigest(),embeddings=embed.flatten().tolist(),gate_output=out.flatten().tolist())))
