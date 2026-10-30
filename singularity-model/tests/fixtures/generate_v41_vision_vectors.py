"""Generate image-grid and patch vectors from pinned checkpoint image_processor.py."""
import importlib.util, io, json, sys
from types import SimpleNamespace
from PIL import Image
import torch
import torch.nn.functional as F
spec=importlib.util.spec_from_file_location('v41_image_processor',sys.argv[1]); m=importlib.util.module_from_spec(spec); sys.modules[spec.name]=m; spec.loader.exec_module(m)
a=SimpleNamespace(vision_patch_size=14,vision_downsample_ratio=3,vision_max_n_token=1024,vision_min_pixels=295936,vision_max_wh_ratio=None)
sizes=[(512,512),(2000,1000),(1000,2000),(64,10000),(10000,64),(42,42),(1,1),(4096,4096)]
grids=[{'width':w,'height':h,'result':m.plan_image_grid(w,h,a)} for w,h in sizes]
a2=SimpleNamespace(vision_patch_size=2,vision_downsample_ratio=2,vision_max_n_token=128,vision_min_pixels=0,vision_max_wh_ratio=None)
rgb=bytes((i*37+11)%256 for i in range(4*4*3)); image=Image.frombytes('RGB',(4,4),rgb); buf=io.BytesIO();image.save(buf,format='PNG')
patches,vh,vw,lh,lw=m.load_image({'data':buf.getvalue()},a2)
features=torch.arange(3*4*2,dtype=torch.float32).reshape(3,4,2)
fold=F.unfold(F.pad(features.permute(2,0,1),(0,2,0,0)).unsqueeze(0),3,stride=3).squeeze(0).transpose(0,1).flatten().tolist()
unfold=torch.tensor(fold,dtype=torch.bfloat16).reshape(2,18)
w1=torch.tensor([((i*7)%17-8)/16 for i in range(4*18)],dtype=torch.bfloat16).reshape(4,18)
b1=torch.tensor([.125,-.25,.375,-.5],dtype=torch.bfloat16)
w2=torch.tensor([((i*5)%11-5)/8 for i in range(16)],dtype=torch.bfloat16).reshape(4,4)
b2=torch.tensor([.25,.125,-.125,-.25],dtype=torch.bfloat16)
aligned=F.linear(F.gelu(F.linear(unfold,w1,b1)),w2,b2).float().flatten().tolist()
print(json.dumps({"aligned":aligned,"unfold":fold,'grids':grids,'rgb':list(rgb),'patches':patches.float().flatten().tolist(),'types':m.image_token_types(lh,lw).tolist(),'vit':[vh,vw]}))
