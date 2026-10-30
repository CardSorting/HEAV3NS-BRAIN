"""CPU PyTorch formulation of checkpoint FP8 activation block quantization."""
import json,torch
x=torch.tensor([(i%17-8)*.137 for i in range(32)]+[0.]*32+[(i%11-5)*8.5 for i in range(32)]+[(i-16)*.0001 for i in range(32)],dtype=torch.bfloat16).float()
y=x.reshape(-1,32)
s=(y.abs().amax(dim=-1).clamp_min(1e-4)/448).log2().ceil().exp2()
q=(y/s[:,None]).clamp(-448,448).to(torch.float8_e4m3fn)
print(json.dumps(dict(input=x.tolist(),values=q.contiguous().view(torch.uint8).flatten().tolist(),scales=s.to(torch.float8_e8m0fnu).view(torch.uint8).tolist())))
