"""Generate HC vectors with PyTorch CPU and pinned layer-6 attn parameter bytes.
Pass the raw byte range (base, fn, scale) extracted from checkpoint shard 9.
"""
import hashlib, json, struct, sys, torch
torch.set_num_threads(1)
def run(hidden, base, weight, scale):
    h=4; n=hidden*h
    residual=torch.tensor([(i%97-48)/64 for i in range(n)],dtype=torch.bfloat16).reshape(h,hidden)
    out=torch.tensor([(i%13-6)/8 for i in range(hidden)],dtype=torch.bfloat16)
    m=(torch.nn.functional.linear(residual.float().reshape(-1),weight) * torch.rsqrt(residual.float().square().mean()+1e-20))
    pre=torch.sigmoid(m[:h]*scale[0]+base[:h])+1e-6
    post=2*torch.sigmoid(m[h:2*h]*scale[1]+base[h:2*h])
    comb=(m[2*h:]*scale[2]+base[2*h:]).reshape(h,h).softmax(-1)+1e-6
    comb=comb/(comb.sum(-2,keepdim=True)+1e-6)
    for _ in range(19):
        comb=comb/(comb.sum(-1,keepdim=True)+1e-6)
        comb=comb/(comb.sum(-2,keepdim=True)+1e-6)
    current_pre=torch.tensor([1.,0.,0.,0.])
    collapsed=(current_pre[:,None]*residual.float()).sum(0).to(torch.bfloat16)
    expanded=(post[:,None]*out[None,:]+(comb[:,:,None]*residual[:,None,:]).sum(0)).to(torch.bfloat16)
    idx=[0,1,hidden//2,hidden-1]
    return dict(pre=pre.tolist(),post=post.tolist(),comb=comb.flatten().tolist(),collapsed=[float(collapsed[i]) for i in idx],expanded=[[float(expanded[j,i]) for i in idx] for j in range(h)])
raw=open(sys.argv[1],'rb').read()
assert len(raw)==1966188
values=torch.tensor(struct.unpack('<491547f',raw))
base=values[:24]; weight=values[24:24+24*20480].reshape(24,20480); scale=values[-3:]
small_weight=torch.tensor([(i%31-15)/64 for i in range(24*32)]).reshape(24,32)
small_base=torch.tensor([(i%11-5)/8 for i in range(24)])
small_scale=torch.tensor([.25,-.5,.75])
# Two shifted sublayers using the same synthetic parameters. Their simple
# sublayer is f(x)=x/2; HC uses the previous step's pre, not the newly computed one.
residual=torch.tensor([(i-4)/8 for i in range(8)],dtype=torch.bfloat16).repeat(4,1)
pre=torch.tensor([1.,0.,0.,0.])
for _ in range(2):
    m=torch.nn.functional.linear(residual.float().flatten(),small_weight)*torch.rsqrt(residual.float().square().mean()+1e-20)
    new_pre=torch.sigmoid(m[:4]*small_scale[0]+small_base[:4])+1e-6
    post=2*torch.sigmoid(m[4:8]*small_scale[1]+small_base[4:8])
    comb=(m[8:]*small_scale[2]+small_base[8:]).reshape(4,4).softmax(-1)+1e-6
    comb=comb/(comb.sum(-2,keepdim=True)+1e-6)
    for j in range(19):
        comb=comb/(comb.sum(-1,keepdim=True)+1e-6)
        comb=comb/(comb.sum(-2,keepdim=True)+1e-6)
    collapsed=(pre[:,None]*residual.float()).sum(0).to(torch.bfloat16)
    output=(collapsed.float()*.5).to(torch.bfloat16)
    residual=(post[:,None]*output[None,:]+(comb[:,:,None]*residual[:,None,:]).sum(0)).to(torch.bfloat16)
    pre=new_pre
print(json.dumps(dict(checkpoint_sha256=hashlib.sha256(raw).hexdigest(),checkpoint=run(5120,base,weight,scale),synthetic=run(8,small_base,small_weight,small_scale),two_step_residual=residual.float().flatten().tolist(),two_step_pre=pre.tolist())))
