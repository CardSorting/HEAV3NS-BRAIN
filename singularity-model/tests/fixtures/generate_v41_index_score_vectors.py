"""Generate BF16 index score oracle using PyTorch CPU."""
import json
import torch

torch.set_num_threads(1)
heads, dim, positions = 4, 32, 5
q = torch.tensor([((i * 31) % 101 - 50) / 37 for i in range(heads * dim)], dtype=torch.bfloat16).reshape(heads, dim)
k = torch.tensor([((i * 17) % 83 - 41) / 29 for i in range(positions * dim)], dtype=torch.bfloat16).reshape(positions, dim)
w = torch.tensor([0.093, -0.31, 0.77, 0.27], dtype=torch.bfloat16)
scores = (torch.einsum('hd,td->ht', q, k).relu() * w[:, None]).sum(dim=0)
print(json.dumps(dict(q=q.float().flatten().tolist(), keys=k.float().flatten().tolist(), weights=w.float().tolist(), scores=scores.float().tolist())))
