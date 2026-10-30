"""Independent PyTorch CPU transcription of the checkpoint's online sparse attention."""
import json
import torch

torch.set_num_threads(1)
h, d, n = 2, 32, 75
q = torch.tensor([((i * 13) % 53 - 26) / 13 for i in range(h * d)], dtype=torch.bfloat16).reshape(h, d)
kv = torch.tensor([((i * 17) % 97 - 48) / 31 for i in range(n * d)], dtype=torch.bfloat16).reshape(n, d)
sink = torch.tensor([0.25, -0.75], dtype=torch.float32)
indices = list(range(n))
indices[5] = None
indices[67] = None
maximum = torch.full((h,), -1e30)
denominator = torch.zeros(h)
out = torch.zeros((h, d))
for start in range(0, len(indices), 64):
    ids = indices[start:start+64]
    mask = torch.tensor([i is not None for i in ids])
    values = kv[torch.tensor([i if i is not None else 0 for i in ids])].float()
    scores = (q.float() @ values.T) * (d ** -0.5)
    scores[:, ~mask] = -torch.inf
    new_max = torch.maximum(maximum, scores.max(dim=1).values)
    rescale = (maximum - new_max).exp()
    probs = (scores - new_max[:, None]).exp()
    denominator = denominator * rescale + probs.sum(dim=1)
    out = out * rescale[:, None] + probs.to(torch.bfloat16).float() @ values
    maximum = new_max
denominator += (sink - maximum).exp()
out = (out / denominator[:, None]).to(torch.bfloat16).float()
print(json.dumps(dict(q=q.float().flatten().tolist(), kv=kv.float().flatten().tolist(), sink=sink.tolist(), indices=indices, output=out.flatten().tolist())))
