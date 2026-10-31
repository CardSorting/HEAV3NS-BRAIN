"""Persistent Modal Volume checkpoint storage, CSF codec preparation, resumable downloader, and manifest validator."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

def _resolve_repo_root(file_path: Path) -> Path:
    resolved = file_path.resolve()
    for parent in resolved.parents:
        if (parent / "deploy" / "modal").is_dir() or (parent / "Cargo.toml").is_file():
            return parent
    candidates = [
        Path("/workspace/singularity"),
        Path("/opt/singularity"),
        Path.cwd(),
        *(Path(p) for p in sys.path if p),
    ]
    for candidate in candidates:
        if (candidate / "deploy" / "modal").is_dir() or (candidate / "Cargo.toml").is_file():
            return candidate
    return resolved.parent


if __package__ in (None, ""):
    _modal_dir = str(Path(__file__).resolve().parent)
    _repo_root = str(_resolve_repo_root(Path(__file__)))
    if _modal_dir in sys.path and (Path(_modal_dir) / "deploy.py").is_file():
        sys.path.remove(_modal_dir)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from deploy.modal.config import (
        CHECKPOINT_DIR,
        CSF_COMPRESSED_SCALE_COUNT,
        CSF_MODEL_ID,
        CSF_PINNED_REVISION,
        CSF_TOTAL_BYTES,
        MODEL_ID,
        PINNED_INDEX_SHA256,
        PINNED_REVISION,
        PINNED_SHARD_COUNT,
        PINNED_TENSOR_COUNT,
        PINNED_TOTAL_BYTES,
        expected_shard_filenames,
    )
else:
    from .config import (
        CHECKPOINT_DIR,
        CSF_COMPRESSED_SCALE_COUNT,
        CSF_MODEL_ID,
        CSF_PINNED_REVISION,
        CSF_TOTAL_BYTES,
        MODEL_ID,
        PINNED_INDEX_SHA256,
        PINNED_REVISION,
        PINNED_SHARD_COUNT,
        PINNED_TENSOR_COUNT,
        PINNED_TOTAL_BYTES,
        expected_shard_filenames,
    )

REQUIRED_METADATA_FILES: List[str] = [
    "config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
]

CSF_FIXED_SUFFIX: str = ".mxfp4_csf_fixed"
CSF_EXCEPTIONS_SUFFIX: str = ".mxfp4_csf_exceptions"

DTYPE_ELEMENT_BYTES: Dict[str, int] = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 of a file using bounded memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def decode_mxfp4_csf_scale(
    fixed_bytes: bytes,
    exception_bytes: bytes,
    rows: int,
    cols: int,
) -> bytes:
    """Decode a `row-base-offset1-u24-exceptions/1` compressed `F8_E8M0` scale tensor
    into its bit-exact `[rows, cols]` uncompressed byte payload.
    """
    if rows <= 0 or cols <= 0 or cols % 8 != 0:
        raise ValueError(
            f"Invalid CSF scale dimensions [{rows}, {cols}]: rows must be > 0 and cols a positive multiple of 8"
        )
    row_bytes = cols // 8
    expected_fixed_len = rows * (1 + row_bytes)
    if len(fixed_bytes) != expected_fixed_len:
        raise ValueError(
            f"Invalid CSF fixed byte length {len(fixed_bytes)}: expected {expected_fixed_len} for [{rows}, {cols}]"
        )
    if len(exception_bytes) % 4 != 0:
        raise ValueError(
            f"Invalid CSF exception byte length {len(exception_bytes)}: must be a multiple of 4"
        )

    total_elems = rows * cols
    out = bytearray(total_elems)
    pos = 0
    for b_start in range(0, rows, 16):
        blk_rows = min(16, rows - b_start)
        bases = fixed_bytes[pos : pos + blk_rows]
        pos += blk_rows
        bits = fixed_bytes[pos : pos + blk_rows * row_bytes]
        pos += blk_rows * row_bytes
        for r_in in range(blk_rows):
            base = bases[r_in]
            row_bits = bits[r_in * row_bytes : (r_in + 1) * row_bytes]
            row_offset = (b_start + r_in) * cols
            for byte_idx, mask in enumerate(row_bits):
                col_base = row_offset + byte_idx * 8
                for bit_idx in range(8):
                    bit = (mask >> bit_idx) & 1
                    out[col_base + bit_idx] = (base + bit) & 0xFF

    for (word,) in struct.iter_unpack("<I", exception_bytes):
        idx = word & 0xFFFFFF
        val = (word >> 24) & 0xFF
        if idx >= total_elems:
            raise ValueError(
                f"CSF exception flat index {idx} out of bounds for scale tensor of {total_elems} elements"
            )
        out[idx] = val

    return bytes(out)


def reconstruct_csf_shard(
    csf_shard_path: str | Path,
    output_shard_path: str | Path,
    *,
    receipt_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Reconstruct a bit-exact uncompressed Safetensors shard from a `lil-mxfp4-csf-checkpoint/1`
    compressed shard, optionally verifying every reconstructed scale tensor against its receipt SHA-256.
    """
    src_path = Path(csf_shard_path)
    dst_path = Path(output_shard_path)
    with src_path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len).decode("utf-8"))
        payload_start = 8 + header_len

    receipt_map: Dict[str, Dict[str, object]] = {}
    if receipt_path is not None:
        receipt_data = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        for item in receipt_data.get("compressed_tensors", []):
            receipt_map[item["name"]] = item

    # Determine logical tensor names in deterministic order
    raw_tensors: Dict[str, Dict[str, object]] = {
        k: v for k, v in header.items() if k != "__metadata__"
    }
    logical_entries: List[Tuple[str, str, List[int], Optional[Tuple[int, int]]]] = []
    csf_pairs: Dict[str, Tuple[Tuple[int, int], Tuple[int, int], int, int]] = {}
    visited_logical = set()

    for name, info in raw_tensors.items():
        if name.endswith(CSF_FIXED_SUFFIX):
            logical_name = name[: -len(CSF_FIXED_SUFFIX)]
            if logical_name in visited_logical:
                continue
            visited_logical.add(logical_name)
            exc_name = f"{logical_name}{CSF_EXCEPTIONS_SUFFIX}"
            if exc_name not in raw_tensors:
                raise ValueError(f"Missing companion '{exc_name}' in '{src_path}'")
            prefix = logical_name[: -len(".scale")]
            weight_info = raw_tensors.get(f"{prefix}.weight")
            if not weight_info:
                raise ValueError(f"Missing companion weight '{prefix}.weight' in '{src_path}'")
            w_shape = weight_info["shape"]
            rows, cols = int(w_shape[0]), int(w_shape[1]) // 16
            f_off = tuple(info["data_offsets"])
            e_off = tuple(raw_tensors[exc_name]["data_offsets"])
            csf_pairs[logical_name] = (f_off, e_off, rows, cols)
            logical_entries.append((logical_name, "F8_E8M0", [rows, cols], None))
        elif name.endswith(CSF_EXCEPTIONS_SUFFIX):
            continue
        else:
            visited_logical.add(name)
            logical_entries.append(
                (name, str(info["dtype"]), list(info["shape"]), tuple(info["data_offsets"]))
            )

    new_header: Dict[str, object] = {}
    cursor = 0
    for name, dtype, shape, _ in logical_entries:
        elem_count = 1
        for dim in shape:
            elem_count *= dim
        nbytes = elem_count * DTYPE_ELEMENT_BYTES[dtype]
        new_header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + nbytes],
        }
        cursor += nbytes

    new_header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    verified_receipts = 0

    with src_path.open("rb") as src_f, dst_path.open("wb") as dst_f:
        dst_f.write(struct.pack("<Q", len(new_header_bytes)))
        dst_f.write(new_header_bytes)
        for name, dtype, shape, raw_offsets in logical_entries:
            if raw_offsets is not None:
                start, end = raw_offsets
                src_f.seek(payload_start + start)
                remaining = end - start
                while remaining > 0:
                    chunk = src_f.read(min(4 * 1024 * 1024, remaining))
                    if not chunk:
                        raise IOError(f"Unexpected EOF reading '{name}' from '{src_path}'")
                    dst_f.write(chunk)
                    remaining -= len(chunk)
            else:
                f_off, e_off, rows, cols = csf_pairs[name]
                src_f.seek(payload_start + f_off[0])
                f_bytes = src_f.read(f_off[1] - f_off[0])
                src_f.seek(payload_start + e_off[0])
                e_bytes = src_f.read(e_off[1] - e_off[0])
                decoded = decode_mxfp4_csf_scale(f_bytes, e_bytes, rows, cols)
                if name in receipt_map:
                    expected_sha = receipt_map[name]["source_sha256"]
                    actual_sha = hashlib.sha256(decoded).hexdigest()
                    if actual_sha != expected_sha:
                        raise ValueError(
                            f"Receipt SHA-256 mismatch for '{name}': got {actual_sha}, expected {expected_sha}"
                        )
                    verified_receipts += 1
                dst_f.write(decoded)

    return {
        "source_shard": str(src_path),
        "output_shard": str(dst_path),
        "logical_tensors": len(logical_entries),
        "reconstructed_scales": len(csf_pairs),
        "verified_receipts": verified_receipts,
        "output_bytes": dst_path.stat().st_size,
    }


def prepare_csf_checkpoint_directory(
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
) -> Dict[str, object]:
    """Prepare a `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF` checkpoint directory
    so both `metadata/*` and `tensors/*.safetensors` are linked/exposed at the root paths expected
    by SINGULARITY (`config.json`, `tokenizer.json`, `tokenizer_config.json`,
    `model.safetensors.index.json`, and `model-*-of-00048.safetensors`).
    """
    root = Path(checkpoint_dir)
    meta_dir = root / "metadata"
    tensors_dir = root / "tensors"
    linked_metadata: List[str] = []
    linked_shards: int = 0

    if meta_dir.is_dir():
        for fname in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        ):
            src = meta_dir / fname
            dst = root / fname
            if src.is_file():
                # Ensure root/config.json and root/model.safetensors.index.json match upstream metadata/*
                if not dst.exists() or sha256_file(dst) != sha256_file(src):
                    shutil.copy2(src, dst)
                    linked_metadata.append(fname)

    if tensors_dir.is_dir():
        for shard_file in sorted(tensors_dir.glob("model-*-of-*.safetensors")):
            dst = root / shard_file.name
            if not dst.exists():
                try:
                    os.symlink(Path("tensors") / shard_file.name, dst)
                except OSError:
                    pass
                linked_shards += 1

    manifest_info: Dict[str, object] = {}
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest_info = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest_info = {}

    verification_info: Dict[str, object] = {}
    verification_path = root / "verification.json"
    if verification_path.is_file():
        try:
            verification_info = json.loads(verification_path.read_text(encoding="utf-8"))
        except Exception:
            verification_info = {}

    return {
        "checkpoint_dir": str(root),
        "is_csf_layout": meta_dir.is_dir() or tensors_dir.is_dir(),
        "linked_metadata": linked_metadata,
        "linked_shards": linked_shards,
        "schema": manifest_info.get("schema"),
        "codec": manifest_info.get("codec"),
        "verified_in_manifest": verification_info.get("verified"),
    }


def inspect_safetensors_header(shard_path: Path) -> Dict[str, object]:
    """Read and validate the 8-byte little-endian header length and JSON header of a Safetensors shard
    without reading the multi-gigabyte payload into memory. Supports both standard Safetensors shards
    and `lil-mxfp4-csf-checkpoint/1` lossless CSF shards.
    """
    file_size = shard_path.stat().st_size
    if file_size < 8:
        raise ValueError(f"Shard '{shard_path}' is smaller than 8 bytes ({file_size})")

    with shard_path.open("rb") as handle:
        header_len_bytes = handle.read(8)
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        if header_len == 0 or 8 + header_len > file_size:
            raise ValueError(
                f"Shard '{shard_path}' has invalid header length {header_len} for file size {file_size}"
            )
        raw_header = handle.read(header_len)

    header = json.loads(raw_header.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"Shard '{shard_path}' header is not a JSON object")

    metadata = header.get("__metadata__", {}) if isinstance(header.get("__metadata__"), dict) else {}
    payload_bytes = file_size - 8 - header_len
    raw_tensor_count = 0
    max_end_offset = 0
    fixed_keys = set()
    exc_keys = set()

    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Tensor '{name}' in '{shard_path}' has non-object metadata")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in DTYPE_ELEMENT_BYTES:
            raise ValueError(f"Tensor '{name}' in '{shard_path}' has unsupported dtype '{dtype}'")
        if not isinstance(shape, list) or any(not isinstance(d, int) or d < 0 for d in shape):
            raise ValueError(f"Tensor '{name}' in '{shard_path}' has invalid shape {shape}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) for o in offsets)
        ):
            raise ValueError(f"Tensor '{name}' in '{shard_path}' has invalid offsets {offsets}")
        start, end = offsets
        if start < 0 or end < start or end > payload_bytes:
            raise ValueError(
                f"Tensor '{name}' in '{shard_path}' offsets [{start}, {end}] exceed payload {payload_bytes}"
            )
        elem_count = 1
        for dim in shape:
            elem_count *= dim
        expected_bytes = elem_count * DTYPE_ELEMENT_BYTES[dtype]
        if (end - start) != expected_bytes:
            raise ValueError(
                f"Tensor '{name}' in '{shard_path}' byte span {end - start} != expected {expected_bytes}"
            )
        if name.endswith(CSF_FIXED_SUFFIX):
            fixed_keys.add(name[: -len(CSF_FIXED_SUFFIX)])
        elif name.endswith(CSF_EXCEPTIONS_SUFFIX):
            exc_keys.add(name[: -len(CSF_EXCEPTIONS_SUFFIX)])
        raw_tensor_count += 1
        if end > max_end_offset:
            max_end_offset = end

    if fixed_keys != exc_keys:
        raise ValueError(
            f"Shard '{shard_path}' has mismatched CSF fixed/exceptions tensor pairs"
        )

    csf_scale_count = len(fixed_keys)
    logical_tensor_count = raw_tensor_count - csf_scale_count

    return {
        "file": shard_path.name,
        "file_size": file_size,
        "header_len": header_len,
        "payload_bytes": payload_bytes,
        "tensor_count": logical_tensor_count,
        "raw_tensor_count": raw_tensor_count,
        "csf_scale_count": csf_scale_count,
        "schema": metadata.get("schema"),
        "codec": metadata.get("codec"),
        "max_end_offset": max_end_offset,
    }


def _resolve_metadata_file(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.is_file():
        return direct
    in_meta = root / "metadata" / filename
    if in_meta.is_file():
        return in_meta
    return direct


def _resolve_shard_file(root: Path, shard_name: str) -> Path:
    direct = root / shard_name
    if direct.is_file():
        return direct
    in_tensors = root / "tensors" / shard_name
    if in_tensors.is_file():
        return in_tensors
    return direct


def validate_checkpoint_directory(
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
    *,
    strict_pinned: bool = True,
    verify_shard_headers: bool = True,
) -> Dict[str, object]:
    """Validate a checkpoint directory on the persistent Modal Volume.

    Supports both standard 48-shard `deepseek-ai/DeepSeek-V4.1-Flash-1017` layouts and
    `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF` (`lil-mxfp4-csf-checkpoint/1`)
    layouts. When `strict_pinned=True`, enforces the exact pinned index SHA-256, 48 shards,
    and 96,085 logical tensors.
    """
    root = Path(checkpoint_dir)
    errors: List[str] = []
    if not root.is_dir():
        return {
            "valid": False,
            "checkpoint_dir": str(root),
            "index_sha256": None,
            "referenced_shards": 0,
            "present_shards": 0,
            "shard_count": 0,
            "indexed_tensors": 0,
            "verified_header_tensors": 0,
            "verified_csf_scales": 0,
            "schema": None,
            "codec": None,
            "total_size_metadata": None,
            "total_shard_file_bytes": 0,
            "total_bytes": 0,
            "errors": [f"Checkpoint directory does not exist: {root}"],
        }

    for req_file in REQUIRED_METADATA_FILES:
        if not _resolve_metadata_file(root, req_file).is_file():
            errors.append(f"Missing required file: {req_file}")

    if errors:
        return {
            "valid": False,
            "checkpoint_dir": str(root),
            "index_sha256": None,
            "referenced_shards": 0,
            "present_shards": 0,
            "shard_count": 0,
            "indexed_tensors": 0,
            "verified_header_tensors": 0,
            "verified_csf_scales": 0,
            "schema": None,
            "codec": None,
            "total_size_metadata": None,
            "total_shard_file_bytes": 0,
            "total_bytes": 0,
            "errors": errors,
        }

    # 1. Validate config.json
    config_path = _resolve_metadata_file(root, "config.json")
    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        if config_data.get("model_type") != "deepseek_v41":
            errors.append(
                f"Unexpected model_type '{config_data.get('model_type')}' in config.json"
            )
    except Exception as exc:
        errors.append(f"Failed to parse config.json: {exc}")

    # 2. Validate model.safetensors.index.json
    index_path = _resolve_metadata_file(root, "model.safetensors.index.json")
    index_sha256 = sha256_file(index_path)
    if strict_pinned and index_sha256 != PINNED_INDEX_SHA256:
        errors.append(
            f"model.safetensors.index.json SHA-256 mismatch: got {index_sha256}, expected {PINNED_INDEX_SHA256}"
        )

    weight_map: Dict[str, str] = {}
    total_size_meta: Optional[int] = None
    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index_data.get("weight_map", {})
        metadata = index_data.get("metadata", {})
        total_size_meta = metadata.get("total_size")
        if not isinstance(weight_map, dict) or not weight_map:
            errors.append("model.safetensors.index.json has empty or invalid weight_map")
    except Exception as exc:
        errors.append(f"Failed to parse model.safetensors.index.json: {exc}")

    referenced_shards = sorted(set(weight_map.values()))
    if strict_pinned:
        expected_shards = expected_shard_filenames(PINNED_SHARD_COUNT)
        if referenced_shards != expected_shards:
            errors.append(
                f"Shard list mismatch: got {len(referenced_shards)} shards, expected {PINNED_SHARD_COUNT}"
            )
        if len(weight_map) != PINNED_TENSOR_COUNT:
            errors.append(
                f"Tensor count mismatch in index: got {len(weight_map)}, expected {PINNED_TENSOR_COUNT}"
            )
        if total_size_meta is not None and int(total_size_meta) != PINNED_TOTAL_BYTES:
            errors.append(
                f"Metadata total_size mismatch: got {total_size_meta}, expected {PINNED_TOTAL_BYTES}"
            )

    # 3. Verify shard presence and headers
    present_shards = 0
    total_shard_file_bytes = 0
    verified_header_tensors = 0
    verified_csf_scales = 0
    detected_schema: Optional[str] = None
    detected_codec: Optional[str] = None

    for shard_name in referenced_shards:
        shard_path = _resolve_shard_file(root, shard_name)
        if not shard_path.is_file():
            errors.append(f"Missing shard file: {shard_name}")
            continue
        present_shards += 1
        total_shard_file_bytes += shard_path.stat().st_size
        if verify_shard_headers:
            try:
                info = inspect_safetensors_header(shard_path)
                verified_header_tensors += int(info["tensor_count"])
                verified_csf_scales += int(info.get("csf_scale_count", 0))
                if info.get("schema"):
                    detected_schema = str(info["schema"])
                if info.get("codec"):
                    detected_codec = str(info["codec"])
            except Exception as exc:
                errors.append(str(exc))

    if verify_shard_headers and not errors and verified_header_tensors != len(weight_map):
        errors.append(
            f"Header logical tensor sum ({verified_header_tensors}) != weight_map count ({len(weight_map)})"
        )

    if (
        strict_pinned
        and verify_shard_headers
        and not errors
        and verified_csf_scales not in (0, CSF_COMPRESSED_SCALE_COUNT)
    ):
        errors.append(
            f"Unexpected CSF compressed scale count {verified_csf_scales} (expected 0 or {CSF_COMPRESSED_SCALE_COUNT})"
        )

    return {
        "valid": len(errors) == 0,
        "checkpoint_dir": str(root),
        "index_sha256": index_sha256,
        "referenced_shards": len(referenced_shards),
        "present_shards": present_shards,
        "shard_count": present_shards,
        "indexed_tensors": len(weight_map),
        "verified_header_tensors": verified_header_tensors,
        "verified_csf_scales": verified_csf_scales,
        "schema": detected_schema,
        "codec": detected_codec,
        "total_size_metadata": total_size_meta,
        "total_shard_file_bytes": total_shard_file_bytes,
        "total_bytes": total_shard_file_bytes,
        "errors": errors,
    }


def download_pinned_checkpoint(
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
    *,
    repo_id: str = CSF_MODEL_ID,
    revision: str = CSF_PINNED_REVISION,
    hf_token: Optional[str] = None,
    max_workers: int = 8,
    dry_run: bool = False,
    commit_callback: Optional[Callable[[], None]] = None,
) -> Dict[str, object]:
    """Download the pinned `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF` (or
    `deepseek-ai/DeepSeek-V4.1-Flash-1017`) checkpoint into the persistent Modal Volume directory
    with resumable file verification and automatic CSF directory preparation.
    """
    target_dir = Path(checkpoint_dir)
    is_csf_repo = repo_id == CSF_MODEL_ID or "lossless-CSF" in repo_id
    if is_csf_repo:
        allow_patterns = [
            "config.json",
            "manifest.json",
            "lil-manifest.json",
            "build-contract.json",
            "verification.json",
            "SHA256SUMS",
            "LICENSE",
            "NOTICE",
            "metadata/*",
            "receipts/*.json",
            "tensors/model-*-of-00048.safetensors",
        ]
        expected_bytes = CSF_TOTAL_BYTES
    else:
        allow_patterns = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-*-of-00048.safetensors",
        ]
        expected_bytes = PINNED_TOTAL_BYTES

    if dry_run:
        return {
            "status": "dry_run",
            "repo_id": repo_id,
            "revision": revision,
            "checkpoint_dir": str(target_dir),
            "is_csf_repo": is_csf_repo,
            "allow_patterns": allow_patterns,
            "expected_shards": PINNED_SHARD_COUNT,
            "expected_bytes": expected_bytes,
        }

    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if is_csf_repo:
        # Step 1: Download metadata, manifests, and receipts first and prepare root links
        meta_patterns = [p for p in allow_patterns if not p.startswith("tensors/")]
        print(
            f"[CSF download] Fetching metadata & receipts from {repo_id}@{revision} into {target_dir}...",
            flush=True,
        )
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target_dir),
            allow_patterns=meta_patterns,
            max_workers=max_workers,
            token=token,
        )
        prepare_csf_checkpoint_directory(target_dir)
        if commit_callback is not None:
            commit_callback()
        print("[CSF download] Metadata & receipts committed to volume.", flush=True)

        # Step 2: Download the 48 CSF tensor shards in batches of 8, committing after each batch
        shard_names = expected_shard_filenames(PINNED_SHARD_COUNT)
        batch_size = 8
        total_batches = (len(shard_names) + batch_size - 1) // batch_size
        for batch_idx, batch_start in enumerate(range(0, len(shard_names), batch_size), start=1):
            batch = shard_names[batch_start : batch_start + batch_size]
            pending_batch = []
            for name in batch:
                shard_path = _resolve_shard_file(target_dir, name)
                if shard_path.is_file():
                    try:
                        inspect_safetensors_header(shard_path)
                        continue
                    except Exception:
                        pass
                pending_batch.append(name)
            if not pending_batch:
                print(
                    f"[CSF download] Batch {batch_idx}/{total_batches} already verified on volume; skipping.",
                    flush=True,
                )
                continue
            print(
                f"[CSF download] Downloading batch {batch_idx}/{total_batches}: {pending_batch[0]} .. {pending_batch[-1]} ({len(pending_batch)} shards)...",
                flush=True,
            )
            batch_patterns = [f"tensors/{name}" for name in pending_batch]
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(target_dir),
                allow_patterns=batch_patterns,
                max_workers=max_workers,
                token=token,
            )
            prepare_csf_checkpoint_directory(target_dir)
            if commit_callback is not None:
                commit_callback()
            print(
                f"[CSF download] Batch {batch_idx}/{total_batches} committed to volume.",
                flush=True,
            )
        downloaded_path = str(target_dir)
    else:
        print(
            f"[Checkpoint download] Fetching {repo_id}@{revision} into {target_dir}...",
            flush=True,
        )
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target_dir),
            allow_patterns=allow_patterns,
            max_workers=max_workers,
            token=token,
        )
        if commit_callback is not None:
            commit_callback()

    prep_summary = prepare_csf_checkpoint_directory(downloaded_path)
    strict = (repo_id == CSF_MODEL_ID and revision == CSF_PINNED_REVISION) or (
        repo_id == MODEL_ID and revision == PINNED_REVISION
    )
    validation = validate_checkpoint_directory(
        downloaded_path,
        strict_pinned=strict,
        verify_shard_headers=True,
    )
    if not validation["valid"]:
        raise RuntimeError(
            f"Downloaded checkpoint failed integrity validation: {validation['errors']}"
        )
    return {
        "status": "completed",
        "repo_id": repo_id,
        "revision": revision,
        "checkpoint_dir": str(downloaded_path),
        "preparation": prep_summary,
        "validation": validation,
    }
