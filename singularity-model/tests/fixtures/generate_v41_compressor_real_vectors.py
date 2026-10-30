"""Generate layer-2 compressor oracle from the pinned shard-5 byte range."""
import hashlib
import json
import sys
import torch

torch.set_num_threads(1)
raw = open(sys.argv[1], 'rb').read()
digest = hashlib.sha256(raw).hexdigest()
assert digest == '7d5b93e38f6a67fa63f13f06815b090247cfa13272ae27f5de97fc787e88667d'
norm = torch.frombuffer(bytearray(raw[:1024]), dtype=torch.bfloat16).float()
gate = torch.frombuffer(bytearray(raw[1024:1024+5242880]), dtype=torch.bfloat16).reshape(512, 5120).float()
kv = torch.frombuffer(bytearray(raw[1024+5242880:]), dtype=torch.bfloat16).reshape(512, 5120).float()
x = torch.tensor([[((i * 13 + t * 7) % 97 - 48) / 64 for i in range(5120)] for t in range(4)], dtype=torch.bfloat16).float()
projected = x @ kv.T
scores = x @ gate.T
pooled = (projected.reshape(2, 2, 512) * scores.reshape(2, 2, 512).softmax(dim=1)).sum(dim=1).to(torch.bfloat16).float()
out = (norm * pooled * torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + 1e-20)).to(torch.bfloat16).float()
print(json.dumps({'sha256': digest, 'outputs': out.tolist()}))
