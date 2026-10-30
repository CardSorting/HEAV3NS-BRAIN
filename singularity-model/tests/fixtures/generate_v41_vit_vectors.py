"""Generate a tiny vision-tower oracle using pinned checkpoint vision.py."""
import importlib.util,json,sys,torch
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('vision_reference',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.set_num_threads(1);torch.set_default_dtype(torch.bfloat16)
args=SimpleNamespace(vision_patch_size=2,vision_dim=8,vision_n_heads=2,vision_inter_dim=6,vision_n_layers=2,vision_rope_theta=10000.)
model=m.ViT(args)
for name,p in model.named_parameters():
    i=torch.arange(p.numel(),dtype=torch.float32)
    v=(((i*13+len(name)*7)%29)-14)/32
    if name.endswith('norm.weight') or name.endswith('norm1.weight') or name.endswith('norm2.weight'):v=1+v/4
    p.data.copy_(v.to(torch.bfloat16).reshape(p.shape).to(p.dtype))
patches=torch.tensor([((i*7)%31-15)/16 for i in range(4*12)],dtype=torch.bfloat16).reshape(4,3,2,2)
out=model(patches,2,2).float().flatten().tolist()
tensors={name:{'shape':list(p.shape),'values':p.detach().to(torch.bfloat16).float().flatten().tolist()} for name,p in model.named_parameters()}
print(json.dumps({'patches':patches.float().flatten().tolist(),'output':out,'tensors':tensors}))
