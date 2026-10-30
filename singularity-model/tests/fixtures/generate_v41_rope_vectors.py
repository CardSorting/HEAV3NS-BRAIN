"""PyTorch CPU V4.1 rotary/YaRN reference values."""
import json,math,torch
dim=64
values=torch.tensor([(i%17-8)/16 for i in range(68)],dtype=torch.bfloat16)
def freqs(base,yarn):
    f=1.0/(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim))
    if yarn:
        def corrected(rot): return dim*math.log(65536/(rot*2*math.pi))/(2*math.log(base))
        low=max(math.floor(corrected(32)),0); high=min(math.ceil(corrected(1)),dim-1)
        ramp=((torch.arange(dim//2)-low)/max(high-low,1e-3)).clamp(0,1)
        f=f/16*ramp+f*(1-ramp)
    return f
cases=[]
for yarn,base in [(False,10000.),(True,160000.)]:
    f=freqs(base,yarn)
    for pos in [0,1024,65536,1048575]:
        angle=torch.tensor(pos,dtype=torch.float32)*f
        z=torch.view_as_complex(values[-dim:].float().reshape(-1,2))
        out=torch.view_as_real(z*torch.polar(torch.ones_like(angle),angle)).flatten().to(torch.bfloat16).float()
        cases.append(dict(yarn=yarn,base=base,position=pos,freqs=f.tolist(),output=out.tolist()))
print(json.dumps(dict(input=values.float().tolist(),cases=cases)))
