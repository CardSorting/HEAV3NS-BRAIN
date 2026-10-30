import numpy as np
from sympy import nextprime
seen=set(); sizes=[]
for layer in (1,14):
  layer_sizes=[]
  for _ in range(3):
    row=[]; p=15999999
    for _ in range(8):
      p=int(nextprime(p))
      while p in seen: p=int(nextprime(p))
      seen.add(p); row.append(p)
    layer_sizes.append(row)
  sizes.append(layer_sizes)
print('totals', [sum(map(sum, layer)) for layer in sizes]); print('first and last', [(a[0][0],a[-1][-1]) for a in sizes])
b=(np.iinfo(np.int64).max//99092)//2
mult=[(np.random.default_rng(10007*l).integers(0,b,size=4,dtype=np.int64)*2+1).tolist() for l in (1,14)]
seq=[7,11,-1,19,23]; pad=2
for pos in range(len(seq)):
  toks=[]; blocked=False
  for shift in range(4):
    if pos<shift or seq[pos-shift]==-1: blocked=True
    toks.append(pad if blocked else seq[pos-shift])
  result=[]
  for layer in range(2):
    rolling=toks[0]*mult[layer][0]; out=[]; offset=0
    for n in range(1,4):
      rolling ^= toks[n]*mult[layer][n]
      for prime in sizes[layer][n-1]:
        out.append(rolling % prime + offset); offset+=prime
    result.append(out)
  print(pos,toks,result)
