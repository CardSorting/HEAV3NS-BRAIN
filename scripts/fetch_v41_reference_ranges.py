#!/usr/bin/env python3
"""Fetch pinned, bounded checkpoint ranges used by the V4.1 CPU reference tests."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
BASE = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/{REVISION}"
# Exact shard offsets include the eight-byte header length and JSON header.
RANGES = {
    "gate-weight.bin": (4207672, 8139831, "6e46b2f91b46aad3420f3dab6dedca03aebf94540f3819dd0c96f29aa5d0dc34"),
    "gate-bias.bin": (258400, 261471, "0a81d7f8e454f084206ee5db885e3963843ac452873187e4ccf3879055497d42"),
    "expert309-weights.bin": (4735551032, 4753245751, "456d5f08e85b2630ccb2e0b7b80adcb6545bb7bdc5e8f5ae9a5e361c4fd90a58"),
    "expert309-scales.bin": (267059000, 268164919, "3a82b25659af5630781de73b3cf1e3148571ed209f23d6621502252d2bb285dc"),
    "hc-attn.bin": (261472, 2227659, "7c1dbec1f1e4f2e3e70dd6bcfc44304b8dd862a7753f5206fda2f7351cbc09bc"),
}
RANGES_48 = {
    "engram14-rows.bin": (672, 1695, "a5bf6632b73410b811706636e8e3d93615518bb240b1c5caf2a5c2da36483847"),
    "engram14-scales.bin": (98308271264, 98308271295, "a09263d594ff68e36d5e902f9b3c53d11cd2a05986a85598d7625fe17d3a57b8"),
    "engram14-qk.bin": (101380404720, 101380486639, "708fad769ea3fda043ca4d65e5481339b93c816036e2a8cc72ebfe03435d7d7f"),
}
ENGRAM_PROJECTION = {
    "engram14-wkv.bin": (101380486640, 101537773039, "48fcb9eea4edf7fa27d9612b4ff810bdba73b97945471f60c5b9144592e88137"),
    "engram14-wkv-scales.bin": (101537773040, 101537926639, "4977ea51a108dde2552feafae630000463f99be4035621e31eeba92fcd2523af"),
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def fetch(directory, name, start, end, expected, shard=9):
    target = directory / name
    if target.is_file() and target.stat().st_size == end - start + 1 and sha256(target) == expected:
        print(f"Verified {target}")
        return
    # Distinct query strings prevent stale range responses from intermediary caches.
    request = urllib.request.Request(f"{BASE}/model-{shard:05d}-of-00048.safetensors?singularity_reference={name}", headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"Server did not honor the bounded range for {name}: {response.status} {content_range!r}")
        remaining = end - start + 1
        digest = hashlib.sha256()
        fd, tmp = tempfile.mkstemp(prefix=".singularity-range-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as out:
                while remaining:
                    data = response.read(min(1024 * 1024, remaining))
                    if not data:
                        raise RuntimeError(f"Truncated response for {name}")
                    out.write(data)
                    digest.update(data)
                    remaining -= len(data)
                if response.read(1):
                    raise RuntimeError(f"Unexpected extra data for {name}")
            if digest.hexdigest() != expected:
                raise RuntimeError(f"SHA-256 mismatch for {name}")
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    print(f"Fetched and verified {target}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory for verified reference bytes")
    parser.add_argument("--engram-projection", action="store_true", help="also fetch 150 MiB of layer-14 projection weights for the full Engram component test")
    parser.add_argument("--compressor", action="store_true", help="also fetch the 10 MiB layer-2 compressor reference range")
    parser.add_argument("--indexer", action="store_true", help="also fetch the layer-2 indexer reference ranges")
    parser.add_argument("--aligner", action="store_true", help="also fetch 140 MiB of checkpoint vision aligner weights")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, (start, end, expected) in RANGES.items():
        fetch(args.directory, name, start, end, expected)
    for name, (start, end, expected) in RANGES_48.items():
        fetch(args.directory, name, start, end, expected, shard=48)
    if args.engram_projection:
        for name, (start, end, expected) in ENGRAM_PROJECTION.items():
            fetch(args.directory, name, start, end, expected, shard=48)
    if args.compressor:
        fetch(args.directory, "compressor-layer2.bin", 4194808, 14681591, "7d5b93e38f6a67fa63f13f06815b090247cfa13272ae27f5de97fc787e88667d", shard=5)
    if args.indexer:
        fetch(args.directory, "indexer-proj.bin", 14681592, 15140599, "28acb1dd040d4cf68be46b735e2e84ab14bccf6d1ad0bfeeb4011cb1e6fa20fc", shard=5)
        fetch(args.directory, "indexer-scale.bin", 19096824, 19101943, "63f7c8035fc839273c2db3daa8ba94fcad1f6e321ff8e5b4d82b6e3a8d682631", shard=5)
        fetch(args.directory, "indexer-query.bin", 443933432, 449176311, "316fe279230fdfde74c744cbaf7416f34ba913a3c7aeaa72a2f57ae8a4d189c5", shard=5)
    if args.aligner:
        fetch(args.directory, "aligner.bin", 27384, 146848503, "5929ee263ea80174c86266f366ef2d5b3a935b4ee313fe5daa19afa7f453b2e4", shard=1)
    print("Set SINGULARITY_V41_GATE_WEIGHT, SINGULARITY_V41_GATE_BIAS, SINGULARITY_V41_EXPERT_WEIGHT, and SINGULARITY_V41_EXPERT_SCALE to the corresponding files before running the ignored real-expert test.")

if __name__ == "__main__":
    main()
