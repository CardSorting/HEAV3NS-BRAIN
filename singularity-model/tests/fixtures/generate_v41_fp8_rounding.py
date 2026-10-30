import torch,json
bits=torch.arange(127,dtype=torch.uint8); values=bits.view(torch.float8_e4m3fn).float(); mid=(values[:-1]+values[1:])/2
inputs=torch.cat((mid,-mid,values,-values,torch.tensor([-500.,500.,0.,-0.]))).float()
# Kernel clamps to representable finite range before cast.
y=inputs.clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8)
print(json.dumps(dict(input=inputs.tolist(),values=y.tolist())))
