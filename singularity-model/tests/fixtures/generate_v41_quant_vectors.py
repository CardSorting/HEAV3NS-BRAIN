"""Reference vectors from PyTorch CPU float8 casts and checkpoint FP4 packing.
Run with PyTorch >= 2.7: python generate_v41_quant_vectors.py > v41_quant_vectors.json
"""
import json, torch
bits = torch.arange(256, dtype=torch.uint8)
def f32bits(x):
    return x.float().contiguous().view(torch.int32).to(torch.int64).bitwise_and(0xffffffff).tolist()
# E4M3FN and E8M0 include NaN; tests compare their classification, not NaN payload bits.
e4 = f32bits(bits.view(torch.float8_e4m3fn))
e8 = f32bits(bits.view(torch.float8_e8m0fnu))
rows, cols = 35, 37
weights = torch.tensor([(i * 37 + 13) % 239 for i in range(rows * cols)], dtype=torch.uint8).view(rows, cols)
# Avoid E4M3FN NaN (0x7f or 0xff) for this matrix.
weights[weights == 127] = 126
scales = torch.tensor([125, 126, 127, 128], dtype=torch.uint8).view(2, 2)
x = torch.tensor([(i % 7 - 3) / 4 for i in range(cols)], dtype=torch.float32)
w = weights.view(torch.float8_e4m3fn).float()
s = scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 0).repeat_interleave(32, 1)[:rows, :cols]
y = (w * s) @ x
r, c = 3, 64
packed = torch.tensor([(i * 19 + 31) % 256 for i in range(r * c // 2)], dtype=torch.uint8).view(r, c // 2)
fp4 = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
expanded = torch.stack([fp4[(packed & 15).long()], fp4[(packed >> 4).long()]], dim=-1).reshape(r, c)
es = torch.tensor([[125, 127], [128, 126], [127, 129]], dtype=torch.uint8)
scale = es.view(torch.float8_e8m0fnu).float().repeat_interleave(32, 1)
ex = torch.tensor([(i % 9 - 4) / 8 for i in range(c)], dtype=torch.float32)
ey = (expanded * scale) @ ex
print(json.dumps(dict(e4_bits=e4,e8_bits=e8,dense=dict(rows=rows,cols=cols,weights=weights.flatten().tolist(),scales=scales.flatten().tolist(),x=x.tolist(),y=y.tolist()),expert=dict(rows=r,cols=c,weights=packed.flatten().tolist(),scales=es.flatten().tolist(),x=ex.tolist(),y=ey.tolist()))))
