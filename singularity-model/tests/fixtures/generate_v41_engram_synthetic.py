"""PyTorch CPU miniature Engram forward using checkpoint layouts."""
import json,torch
torch.set_num_threads(1)
rows=torch.tensor([(i*7+3)%112 for i in range(64)],dtype=torch.uint8).reshape(2,32)
scales=torch.tensor([125,126],dtype=torch.uint8)
embed=(rows.view(torch.float8_e4m3fn).float()*scales.view(torch.float8_e8m0fnu).float()[:,None]).to(torch.bfloat16)
x=embed[1].float(); scale=(x.abs().max().clamp_min(1e-4)/448).log2().ceil().exp2(); x=(x/scale).clamp(-448,448).to(torch.float8_e4m3fn).float()*scale
wb=torch.tensor([(i*13+5)%112 for i in range(96*32)],dtype=torch.uint8).reshape(96,32)
w=wb.view(torch.float8_e4m3fn).float()
p=(w@x).to(torch.bfloat16).float(); key=p[:64].reshape(2,32); value=p[64:]
h=torch.tensor([(i%13-6)/8 for i in range(64)],dtype=torch.bfloat16).float().reshape(2,32)
q=torch.tensor([(i%7-3)/4 for i in range(64)],dtype=torch.bfloat16).float().reshape(2,32)
k=torch.tensor([(i%11-5)/8 for i in range(64)],dtype=torch.bfloat16).float().reshape(2,32)
rstd=torch.rsqrt(h.square().mean(-1)+1e-20)*torch.rsqrt(key.square().mean(-1)+1e-20)
dot=(h*(q*k)*key).sum(-1)*rstd*32**-.5
gate=torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(),dot))
out=(h+gate[:,None]*value).to(torch.bfloat16).float()
print(json.dumps(dict(output=out.flatten().tolist())))
