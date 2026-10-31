use half::{bf16, f16};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use singularity_core::SingularityError;
use singularity_cuda::{GpuBackend, GpuBuffer};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/// Parse JSON while rejecting duplicate keys at every nesting level. A normal
/// serde_json::Value silently overwrites duplicates, hiding corrupt inventories.
struct StrictJson(serde_json::Value);
impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = StrictJson;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("JSON without duplicate object keys")
            }
            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(StrictJson(v.into()))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(StrictJson(v.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(StrictJson(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| StrictJson(n.into()))
                    .ok_or_else(|| E::custom("Non-finite JSON number"))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(StrictJson(v.into()))
            }
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(StrictJson(v.into()))
            }
            fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(StrictJson(serde_json::Value::Null))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(StrictJson(serde_json::Value::Null))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(StrictJson(v)) = seq.next_element()? {
                    values.push(v);
                }
                Ok(StrictJson(values.into()))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some((key, StrictJson(value))) = map.next_entry::<String, StrictJson>()? {
                    if values.insert(key.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "Duplicate JSON key '{key}'"
                        )));
                    }
                }
                Ok(StrictJson(values.into()))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

pub const CSF_SCHEMA: &str = "lil-mxfp4-csf-checkpoint/1";
pub const CSF_CODEC: &str = "row-base-offset1-u24-exceptions/1";
const CSF_FIXED_SUFFIX: &str = ".mxfp4_csf_fixed";
const CSF_EXCEPTIONS_SUFFIX: &str = ".mxfp4_csf_exceptions";

/// Decode a `row-base-offset1-u24-exceptions/1` compressed `F8_E8M0` scale tensor
/// into its bit-exact `[rows, cols]` uncompressed representation.
pub fn decode_mxfp4_csf_scale(
    fixed: &[u8],
    exceptions: &[u8],
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>, SingularityError> {
    if rows == 0 || cols == 0 || cols % 8 != 0 {
        return Err(SingularityError::ModelError {
            reason: format!(
                "Invalid CSF scale dimensions [{rows}, {cols}]: rows must be > 0 and cols a positive multiple of 8"
            ),
        });
    }
    let row_bytes = cols / 8;
    let expected_fixed_len =
        rows.checked_mul(1 + row_bytes)
            .ok_or_else(|| SingularityError::ModelError {
                reason: "CSF fixed byte length overflow".into(),
            })?;
    if fixed.len() != expected_fixed_len {
        return Err(SingularityError::ModelError {
            reason: format!(
                "Invalid CSF fixed payload size: expected {expected_fixed_len} bytes for [{rows}, {cols}], got {}",
                fixed.len()
            ),
        });
    }
    if exceptions.len() % 4 != 0 {
        return Err(SingularityError::ModelError {
            reason: format!(
                "Invalid CSF exceptions payload size {}: must be a multiple of 4",
                exceptions.len()
            ),
        });
    }
    let total_elems = rows
        .checked_mul(cols)
        .ok_or_else(|| SingularityError::ModelError {
            reason: "CSF scale element count overflow".into(),
        })?;
    let mut out = vec![0u8; total_elems];
    let mut pos = 0usize;
    for b_start in (0..rows).step_by(16) {
        let blk_rows = (rows - b_start).min(16);
        let bases = &fixed[pos..pos + blk_rows];
        pos += blk_rows;
        let bits_len = blk_rows * row_bytes;
        let bits = &fixed[pos..pos + bits_len];
        pos += bits_len;
        for r_in in 0..blk_rows {
            let base = bases[r_in];
            let row_bits = &bits[r_in * row_bytes..(r_in + 1) * row_bytes];
            let row_out = &mut out[(b_start + r_in) * cols..(b_start + r_in + 1) * cols];
            for (byte_idx, &b) in row_bits.iter().enumerate() {
                let col_base = byte_idx * 8;
                for bit_idx in 0..8 {
                    let bit = (b >> bit_idx) & 1;
                    row_out[col_base + bit_idx] = base.wrapping_add(bit);
                }
            }
        }
    }
    for chunk in exceptions.chunks_exact(4) {
        let word = u32::from_le_bytes(chunk.try_into().unwrap());
        let idx = (word & 0x00FF_FFFF) as usize;
        let val = (word >> 24) as u8;
        if idx >= total_elems {
            return Err(SingularityError::ModelError {
                reason: format!(
                    "CSF exception flat index {idx} out of bounds for scale tensor of {total_elems} elements"
                ),
            });
        }
        out[idx] = val;
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

#[derive(Debug, Deserialize)]
struct IndexManifest {
    #[serde(default)]
    metadata: HashMap<String, serde_json::Value>,
    weight_map: HashMap<String, String>,
}

struct CsfScaleEntry {
    fixed_offsets: (usize, usize),
    exceptions_offsets: (usize, usize),
    rows: usize,
    cols: usize,
    decoded: std::sync::OnceLock<Vec<u8>>,
}

struct ShardEntry {
    _path: PathBuf,
    mmap: Mmap,
    header_len: usize,
    tensors: HashMap<String, TensorInfo>,
    csf_scales: HashMap<String, CsfScaleEntry>,
}

pub struct SafetensorsMmapLoader {
    shards: Vec<ShardEntry>,
    weight_to_shard: HashMap<String, usize>,
    metadata: HashMap<String, String>,
}

impl SafetensorsMmapLoader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, SingularityError> {
        let p = path.as_ref();
        if p.is_dir() {
            let index_path = p.join("model.safetensors.index.json");
            if index_path.exists() {
                return Self::open_index(&index_path);
            }
            let meta_index_path = p.join("metadata").join("model.safetensors.index.json");
            if meta_index_path.exists() {
                return Self::open_index(&meta_index_path);
            }
            let single_path = p.join("model.safetensors");
            if single_path.exists() {
                return Self::open_single_file(&single_path);
            }
            return Err(SingularityError::ModelError {
                reason: format!(
                    "Directory '{}' contains neither model.safetensors.index.json, metadata/model.safetensors.index.json, nor model.safetensors",
                    p.display()
                ),
            });
        }

        let file_name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if file_name.ends_with(".index.json") || file_name.ends_with(".json") {
            Self::open_index(p)
        } else {
            Self::open_single_file(p)
        }
    }

    fn resolve_shard_path(parent_dir: &Path, shard_name: &str) -> PathBuf {
        let direct = parent_dir.join(shard_name);
        if direct.exists() {
            return direct;
        }
        let in_tensors = parent_dir.join("tensors").join(shard_name);
        if in_tensors.exists() {
            return in_tensors;
        }
        if parent_dir.file_name().and_then(|n| n.to_str()) == Some("metadata") {
            if let Some(root_dir) = parent_dir.parent() {
                let root_direct = root_dir.join(shard_name);
                if root_direct.exists() {
                    return root_direct;
                }
                let root_tensors = root_dir.join("tensors").join(shard_name);
                if root_tensors.exists() {
                    return root_tensors;
                }
            }
        }
        direct
    }

    fn open_single_file(path: &Path) -> Result<Self, SingularityError> {
        let shard = Self::parse_single_shard(path)?;
        let mut weight_to_shard = HashMap::new();
        for name in shard.tensors.keys() {
            weight_to_shard.insert(name.clone(), 0);
        }

        let mut metadata = HashMap::new();
        // If single shard contains metadata, load it
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() >= 8 {
            let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            if mmap.len() >= 8 + header_len {
                if let Ok(root) =
                    serde_json::from_slice::<serde_json::Value>(&mmap[8..8 + header_len])
                {
                    if let Some(obj) = root.as_object() {
                        if let Some(meta_obj) = obj.get("__metadata__").and_then(|v| v.as_object())
                        {
                            for (mk, mv) in meta_obj {
                                if let Some(s) = mv.as_str() {
                                    metadata.insert(mk.clone(), s.to_string());
                                } else {
                                    metadata.insert(mk.clone(), mv.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(Self {
            shards: vec![shard],
            weight_to_shard,
            metadata,
        })
    }

    fn open_index(index_path: &Path) -> Result<Self, SingularityError> {
        let file = File::open(index_path)?;
        let reader = BufReader::with_capacity(1024 * 1024, file);
        let StrictJson(root) = serde_json::from_reader(reader)?;
        let manifest: IndexManifest = serde_json::from_value(root)?;
        let parent_dir = index_path.parent().unwrap_or_else(|| Path::new("."));

        // Collect unique shard filenames in order
        let mut unique_shards: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for shard_name in manifest.weight_map.values() {
            if seen.insert(shard_name.clone()) {
                unique_shards.push(shard_name.clone());
            }
        }

        let shards: Vec<ShardEntry> = if unique_shards.len() > 1 {
            unique_shards
                .par_iter()
                .map(|shard_name| {
                    let shard_path = Self::resolve_shard_path(parent_dir, shard_name);
                    Self::parse_single_shard(&shard_path)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            unique_shards
                .iter()
                .map(|shard_name| {
                    let shard_path = Self::resolve_shard_path(parent_dir, shard_name);
                    Self::parse_single_shard(&shard_path)
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let shard_name_to_idx: HashMap<String, usize> = unique_shards
            .into_iter()
            .enumerate()
            .map(|(idx, name)| (name, idx))
            .collect();

        let mut weight_to_shard = HashMap::new();
        for (weight_name, shard_file) in manifest.weight_map {
            if let Some(&idx) = shard_name_to_idx.get(&shard_file) {
                if !shards[idx].tensors.contains_key(&weight_name) {
                    return Err(SingularityError::ModelError {
                        reason: format!(
                            "Tensor '{weight_name}' is missing from indexed shard '{shard_file}'"
                        ),
                    });
                }
                weight_to_shard.insert(weight_name, idx);
            }
        }

        for (idx, shard) in shards.iter().enumerate() {
            for name in shard.tensors.keys() {
                if weight_to_shard.get(name) != Some(&idx) {
                    return Err(SingularityError::ModelError { reason: format!("Tensor '{name}' in '{}' is unexpected, duplicated, or mapped to another shard", shard._path.display()) });
                }
            }
        }

        Ok(Self {
            shards,
            weight_to_shard,
            metadata: manifest
                .metadata
                .into_iter()
                .map(|(k, v)| {
                    let value = v
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| v.to_string());
                    (k, value)
                })
                .collect(),
        })
    }

    fn parse_single_shard(path: &Path) -> Result<ShardEntry, SingularityError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(SingularityError::ModelError {
                reason: format!(
                    "File '{}' too small to be a valid Safetensors file",
                    path.display()
                ),
            });
        }

        let header_len = usize::try_from(u64::from_le_bytes(mmap[0..8].try_into().unwrap()))
            .map_err(|_| SingularityError::ModelError {
                reason: "Safetensors header length exceeds address space".into(),
            })?;
        let header_end =
            8usize
                .checked_add(header_len)
                .ok_or_else(|| SingularityError::ModelError {
                    reason: "Safetensors header length overflow".into(),
                })?;
        if mmap.len() < header_end {
            return Err(SingularityError::ModelError {
                reason: format!(
                    "Corrupt header length in '{}': requires {} bytes, file only has {}",
                    path.display(),
                    header_end,
                    mmap.len()
                ),
            });
        }

        let header_bytes = &mmap[8..header_end];
        let StrictJson(root) = serde_json::from_slice(header_bytes)?;
        let obj = root
            .as_object()
            .ok_or_else(|| SingularityError::ModelError {
                reason: format!(
                    "Safetensors header in '{}' is not a JSON object",
                    path.display()
                ),
            })?;

        let mut tensors = HashMap::new();
        for (k, v) in obj {
            if k != "__metadata__" {
                let info: TensorInfo = serde_json::from_value(v.clone())?;
                let width = match info.dtype.as_str() {
                    "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => 1usize,
                    "I16" | "U16" | "F16" | "BF16" => 2,
                    "I32" | "U32" | "F32" => 4,
                    "I64" | "U64" | "F64" => 8,
                    _ => {
                        return Err(SingularityError::ModelError {
                            reason: format!("Unsupported dtype '{}' for tensor '{k}'", info.dtype),
                        })
                    }
                };
                let expected = info
                    .shape
                    .iter()
                    .try_fold(width, |n, dim| n.checked_mul(*dim));
                let actual = info.data_offsets.1.checked_sub(info.data_offsets.0);
                if expected.is_none() || expected != actual {
                    return Err(SingularityError::ModelError {
                        reason: format!("Invalid payload size for tensor '{k}'"),
                    });
                }
                tensors.insert(k.clone(), info);
            }
        }

        let data_len = mmap.len() - header_end;
        let mut ranges: Vec<_> = tensors
            .iter()
            .map(|(name, info): (&String, &TensorInfo)| {
                (info.data_offsets.0, info.data_offsets.1, name)
            })
            .collect();
        ranges.sort_unstable();
        let mut end = 0;
        for (start, next_end, name) in ranges {
            if start != end || next_end < start || next_end > data_len {
                return Err(SingularityError::ModelError {
                    reason: format!(
                        "Invalid data offsets for tensor '{name}' in '{}'",
                        path.display()
                    ),
                });
            }
            end = next_end;
        }
        if end != data_len {
            return Err(SingularityError::ModelError {
                reason: format!("Unindexed payload bytes in '{}'", path.display()),
            });
        }

        // Reconcile lil-mxfp4-csf-checkpoint/1 scale tensor pairs (.mxfp4_csf_fixed + .mxfp4_csf_exceptions)
        let fixed_keys: Vec<String> = tensors
            .keys()
            .filter(|k| k.ends_with(CSF_FIXED_SUFFIX))
            .cloned()
            .collect();
        let mut csf_scales = HashMap::new();
        for fixed_key in fixed_keys {
            let logical_name = fixed_key
                .strip_suffix(CSF_FIXED_SUFFIX)
                .unwrap()
                .to_string();
            let exc_key = format!("{logical_name}{CSF_EXCEPTIONS_SUFFIX}");
            let fixed_info = tensors.remove(&fixed_key).unwrap();
            let exc_info =
                tensors
                    .remove(&exc_key)
                    .ok_or_else(|| SingularityError::ModelError {
                        reason: format!(
                            "Missing companion CSF exceptions tensor '{exc_key}' in '{}'",
                            path.display()
                        ),
                    })?;
            if fixed_info.dtype != "U8"
                || fixed_info.shape.len() != 1
                || exc_info.dtype != "U32"
                || exc_info.shape.len() != 1
            {
                return Err(SingularityError::ModelError {
                    reason: format!(
                        "Invalid CSF tensor dtypes/shapes for '{logical_name}' in '{}'",
                        path.display()
                    ),
                });
            }
            let prefix = logical_name.strip_suffix(".scale").ok_or_else(|| {
                SingularityError::ModelError {
                    reason: format!(
                        "CSF compressed tensor '{logical_name}' does not end with '.scale'"
                    ),
                }
            })?;
            let weight_name = format!("{prefix}.weight");
            let weight_info = tensors.get(&weight_name).ok_or_else(|| {
                SingularityError::ModelError {
                    reason: format!(
                        "Missing companion weight tensor '{weight_name}' for CSF scale '{logical_name}' in '{}'",
                        path.display()
                    ),
                }
            })?;
            if weight_info.shape.len() != 2
                || weight_info.shape[0] == 0
                || weight_info.shape[1] == 0
                || weight_info.shape[1] % 16 != 0
            {
                return Err(SingularityError::ModelError {
                    reason: format!(
                        "Invalid companion weight shape {:?} for CSF scale '{logical_name}'",
                        weight_info.shape
                    ),
                });
            }
            let rows = weight_info.shape[0];
            let cols = weight_info.shape[1] / 16;
            if cols % 8 != 0 || rows.checked_mul(1 + cols / 8) != Some(fixed_info.shape[0]) {
                return Err(SingularityError::ModelError {
                    reason: format!(
                        "CSF fixed length {} does not match [{rows}, {cols}] for '{logical_name}'",
                        fixed_info.shape[0]
                    ),
                });
            }
            if tensors.contains_key(&logical_name) {
                return Err(SingularityError::ModelError {
                    reason: format!(
                        "Duplicate logical scale tensor '{logical_name}' in '{}'",
                        path.display()
                    ),
                });
            }
            tensors.insert(
                logical_name.clone(),
                TensorInfo {
                    dtype: "F8_E8M0".to_string(),
                    shape: vec![rows, cols],
                    data_offsets: (0, rows * cols),
                },
            );
            csf_scales.insert(
                logical_name,
                CsfScaleEntry {
                    fixed_offsets: fixed_info.data_offsets,
                    exceptions_offsets: exc_info.data_offsets,
                    rows,
                    cols,
                    decoded: std::sync::OnceLock::new(),
                },
            );
        }
        if let Some(orphan) = tensors.keys().find(|k| k.ends_with(CSF_EXCEPTIONS_SUFFIX)) {
            return Err(SingularityError::ModelError {
                reason: format!(
                    "Orphaned CSF exceptions tensor '{orphan}' without matching '{CSF_FIXED_SUFFIX}' in '{}'",
                    path.display()
                ),
            });
        }

        Ok(ShardEntry {
            _path: path.to_path_buf(),
            mmap,
            header_len,
            tensors,
            csf_scales,
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.weight_to_shard.keys()
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        let shard_idx = *self.weight_to_shard.get(name)?;
        self.shards.get(shard_idx)?.tensors.get(name)
    }

    pub fn is_csf_scale(&self, name: &str) -> bool {
        self.weight_to_shard
            .get(name)
            .and_then(|&idx| self.shards.get(idx))
            .is_some_and(|shard| shard.csf_scales.contains_key(name))
    }

    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    pub fn get_tensor_bytes(&self, name: &str) -> Result<&[u8], SingularityError> {
        let shard_idx = self.weight_to_shard.get(name).copied().ok_or_else(|| {
            SingularityError::ModelError {
                reason: format!("Tensor '{name}' not found in Safetensors manifest"),
            }
        })?;

        let shard = &self.shards[shard_idx];
        if let Some(csf_entry) = shard.csf_scales.get(name) {
            if let Some(decoded) = csf_entry.decoded.get() {
                return Ok(decoded.as_slice());
            }
            let fixed_start = 8 + shard.header_len + csf_entry.fixed_offsets.0;
            let fixed_end = 8 + shard.header_len + csf_entry.fixed_offsets.1;
            let exc_start = 8 + shard.header_len + csf_entry.exceptions_offsets.0;
            let exc_end = 8 + shard.header_len + csf_entry.exceptions_offsets.1;
            if fixed_end > shard.mmap.len() || exc_end > shard.mmap.len() {
                return Err(SingularityError::ModelError {
                    reason: format!("CSF scale '{name}' offsets exceed mmap boundary"),
                });
            }
            let decoded = decode_mxfp4_csf_scale(
                &shard.mmap[fixed_start..fixed_end],
                &shard.mmap[exc_start..exc_end],
                csf_entry.rows,
                csf_entry.cols,
            )?;
            let _ = csf_entry.decoded.set(decoded);
            return Ok(csf_entry.decoded.get().unwrap().as_slice());
        }

        let info = shard
            .tensors
            .get(name)
            .ok_or_else(|| SingularityError::ModelError {
                reason: format!("Tensor '{name}' not found in shard manifest"),
            })?;

        let data_start = 8 + shard.header_len + info.data_offsets.0;
        let data_end = 8 + shard.header_len + info.data_offsets.1;

        if data_end > shard.mmap.len() {
            return Err(SingularityError::ModelError {
                reason: format!(
                    "Tensor '{}' offset range [{}, {}] exceeds mmap boundary {}",
                    name,
                    data_start,
                    data_end,
                    shard.mmap.len()
                ),
            });
        }

        Ok(&shard.mmap[data_start..data_end])
    }

    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>, SingularityError> {
        let info = self
            .get_tensor_info(name)
            .ok_or_else(|| SingularityError::ModelError {
                reason: format!("Tensor '{name}' not found in Safetensors manifest"),
            })?
            .clone();

        let bytes = self.get_tensor_bytes(name)?;

        match info.dtype.as_str() {
            "F32" => {
                if bytes.len() % 4 != 0 {
                    return Err(SingularityError::ModelError {
                        reason: format!(
                            "Byte length {} is not a multiple of 4 for F32",
                            bytes.len()
                        ),
                    });
                }
                Ok(bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect())
            }
            "BF16" => {
                if bytes.len() % 2 != 0 {
                    return Err(SingularityError::ModelError {
                        reason: format!(
                            "Byte length {} is not a multiple of 2 for BF16",
                            bytes.len()
                        ),
                    });
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|c| bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
                    .collect())
            }
            "F16" => {
                if bytes.len() % 2 != 0 {
                    return Err(SingularityError::ModelError {
                        reason: format!(
                            "Byte length {} is not a multiple of 2 for F16",
                            bytes.len()
                        ),
                    });
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                    .collect())
            }
            other => Err(SingularityError::ModelError {
                reason: format!("Unsupported tensor dtype '{other}' for tensor '{name}'"),
            }),
        }
    }

    pub fn upload_tensor_to_gpu(
        &self,
        name: &str,
        backend: &dyn GpuBackend,
    ) -> Result<GpuBuffer, SingularityError> {
        let bytes = self.get_tensor_bytes(name)?;
        let buffer = backend.allocate(bytes.len())?;
        backend.copy_htod(&buffer, bytes)?;
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_safetensors_with_metadata_and_payload() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("model.safetensors");

        let header_json = r#"{"__metadata__": {"format": "pt", "creator": "deepseek"}, "layer1.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}"#;
        let header_bytes = header_json.as_bytes();
        let header_len = (header_bytes.len() as u64).to_le_bytes();

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_len);
        file_bytes.extend_from_slice(header_bytes);
        file_bytes.extend_from_slice(&[7u8; 16]); // 16 bytes tensor payload

        std::fs::write(&file_path, file_bytes).unwrap();

        let loader = SafetensorsMmapLoader::open(&file_path).unwrap();
        assert_eq!(loader.get_metadata("format"), Some(&"pt".to_string()));
        assert_eq!(
            loader.get_metadata("creator"),
            Some(&"deepseek".to_string())
        );

        let tensor_bytes = loader.get_tensor_bytes("layer1.weight").unwrap();
        assert_eq!(tensor_bytes.len(), 16);
        assert_eq!(tensor_bytes[0], 7);

        let f32_vals = loader.get_tensor_f32("layer1.weight").unwrap();
        assert_eq!(f32_vals.len(), 4);
    }

    #[test]
    fn test_safetensors_multi_shard_index() {
        let temp_dir = tempdir().unwrap();
        let shard1_path = temp_dir.path().join("model-00001-of-00002.safetensors");
        let shard2_path = temp_dir.path().join("model-00002-of-00002.safetensors");
        let index_path = temp_dir.path().join("model.safetensors.index.json");

        // Write shard 1
        let h1 = r#"{"layer1.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}"#;
        let h1_len = (h1.len() as u64).to_le_bytes();
        let mut s1_bytes = Vec::new();
        s1_bytes.extend_from_slice(&h1_len);
        s1_bytes.extend_from_slice(h1.as_bytes());
        s1_bytes.extend_from_slice(&1.5f32.to_le_bytes());
        s1_bytes.extend_from_slice(&2.5f32.to_le_bytes());
        std::fs::write(&shard1_path, s1_bytes).unwrap();

        // Write shard 2
        let h2 = r#"{"layer2.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}"#;
        let h2_len = (h2.len() as u64).to_le_bytes();
        let mut s2_bytes = Vec::new();
        s2_bytes.extend_from_slice(&h2_len);
        s2_bytes.extend_from_slice(h2.as_bytes());
        s2_bytes.extend_from_slice(&3.5f32.to_le_bytes());
        s2_bytes.extend_from_slice(&4.5f32.to_le_bytes());
        std::fs::write(&shard2_path, s2_bytes).unwrap();

        // Write index.json
        let index_json = serde_json::json!({
            "metadata": {"total_size": 16},
            "weight_map": {
                "layer1.weight": "model-00001-of-00002.safetensors",
                "layer2.weight": "model-00002-of-00002.safetensors"
            }
        });
        std::fs::write(&index_path, serde_json::to_string(&index_json).unwrap()).unwrap();

        // Open directory containing index
        let loader = SafetensorsMmapLoader::open(temp_dir.path()).unwrap();
        assert_eq!(loader.get_metadata("total_size"), Some(&"16".to_string()));

        let t1 = loader.get_tensor_f32("layer1.weight").unwrap();
        assert_eq!(t1, vec![1.5, 2.5]);

        let t2 = loader.get_tensor_f32("layer2.weight").unwrap();
        assert_eq!(t2, vec![3.5, 4.5]);
    }

    #[test]
    fn rejects_corrupt_offsets_and_inconsistent_index() {
        let dir = tempdir().unwrap();
        let shard = dir.path().join("model.safetensors");
        let write_shard = |header: &str, payload: &[u8]| {
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(payload);
            std::fs::write(&shard, bytes).unwrap();
        };
        write_shard(
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[3,2]}}"#,
            &[0; 4],
        );
        assert!(SafetensorsMmapLoader::open(&shard).is_err());
        write_shard(
            r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        assert!(SafetensorsMmapLoader::open(&shard).is_err());
        write_shard(
            r#"{"x":{"dtype":"UNKNOWN","shape":[4],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        assert!(SafetensorsMmapLoader::open(&shard).is_err());
        write_shard(
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        let index = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index, r#"{"weight_map":{"missing":"model.safetensors"}}"#).unwrap();
        assert!(SafetensorsMmapLoader::open(&index).is_err());
    }

    #[test]
    fn rejects_duplicate_tensor_and_manifest_keys() {
        let dir = tempdir().unwrap();
        let shard = dir.path().join("model.safetensors");
        let write_header = |header: &str| {
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend_from_slice(&[0; 4]);
            std::fs::write(&shard, bytes).unwrap();
        };
        write_header(
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        );
        assert!(SafetensorsMmapLoader::open(&shard).is_err());
        write_header(r#"{"x":{"dtype":"F32","dtype":"I32","shape":[1],"data_offsets":[0,4]}}"#);
        assert!(SafetensorsMmapLoader::open(&shard).is_err());
        write_header(r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#);
        let index = dir.path().join("model.safetensors.index.json");
        std::fs::write(
            &index,
            r#"{"weight_map":{"x":"model.safetensors","x":"model.safetensors"}}"#,
        )
        .unwrap();
        assert!(SafetensorsMmapLoader::open(index).is_err());
    }

    #[test]
    fn test_decode_mxfp4_csf_scale_and_loader_csf_layout() {
        // Test 18 rows x 8 cols (spans two 16-row blocks: 16 rows + 2 rows)
        let rows = 18usize;
        let cols = 8usize;
        let mut expected = vec![0u8; rows * cols];
        let mut fixed = Vec::new();
        for b_start in (0..rows).step_by(16) {
            let blk_rows = (rows - b_start).min(16);
            for r_in in 0..blk_rows {
                let r = b_start + r_in;
                fixed.push(120u8 + r as u8);
            }
            for r_in in 0..blk_rows {
                let r = b_start + r_in;
                let base = 120u8 + r as u8;
                let mask: u8 = if r % 2 == 0 { 0b1010_0101 } else { 0b0101_1010 };
                fixed.push(mask);
                for c in 0..cols {
                    let bit = (mask >> c) & 1;
                    expected[r * cols + c] = base + bit;
                }
            }
        }
        // Override two exception positions: (row=0, col=3) -> 135, (row=17, col=7) -> 111
        let exc0_idx: u32 = 3;
        let exc1_idx: u32 = (17 * cols + 7) as u32;
        expected[exc0_idx as usize] = 135;
        expected[exc1_idx as usize] = 111;
        let mut exceptions = Vec::new();
        exceptions.extend_from_slice(&(exc0_idx | (135u32 << 24)).to_le_bytes());
        exceptions.extend_from_slice(&(exc1_idx | (111u32 << 24)).to_le_bytes());

        let decoded = decode_mxfp4_csf_scale(&fixed, &exceptions, rows, cols).unwrap();
        assert_eq!(decoded, expected);

        // Out-of-bounds exception index should fail
        let bad_exc = ((rows * cols) as u32 | (127u32 << 24)).to_le_bytes();
        assert!(decode_mxfp4_csf_scale(&fixed, &bad_exc, rows, cols).is_err());

        // Now test full SafetensorsMmapLoader with CSF directory layout:
        // metadata/model.safetensors.index.json + tensors/model-00001-of-00001.safetensors
        let dir = tempdir().unwrap();
        let meta_dir = dir.path().join("metadata");
        let tensors_dir = dir.path().join("tensors");
        std::fs::create_dir_all(&meta_dir).unwrap();
        std::fs::create_dir_all(&tensors_dir).unwrap();

        let weight_shape = [rows, cols * 16]; // [18, 128] -> scale shape [18, 8]
        let weight_bytes = vec![0x21u8; weight_shape[0] * weight_shape[1]];
        let w_end = weight_bytes.len();
        let f_end = w_end + fixed.len();
        let e_end = f_end + exceptions.len();

        let shard_header = serde_json::json!({
            "__metadata__": {
                "schema": CSF_SCHEMA,
                "codec": CSF_CODEC
            },
            "layers.0.ffn.experts.0.w1.weight": {
                "dtype": "I8",
                "shape": [weight_shape[0], weight_shape[1]],
                "data_offsets": [0, w_end]
            },
            "layers.0.ffn.experts.0.w1.scale.mxfp4_csf_fixed": {
                "dtype": "U8",
                "shape": [fixed.len()],
                "data_offsets": [w_end, f_end]
            },
            "layers.0.ffn.experts.0.w1.scale.mxfp4_csf_exceptions": {
                "dtype": "U32",
                "shape": [exceptions.len() / 4],
                "data_offsets": [f_end, e_end]
            }
        });
        let shard_header_bytes = serde_json::to_vec(&shard_header).unwrap();
        let mut shard_file = (shard_header_bytes.len() as u64).to_le_bytes().to_vec();
        shard_file.extend_from_slice(&shard_header_bytes);
        shard_file.extend_from_slice(&weight_bytes);
        shard_file.extend_from_slice(&fixed);
        shard_file.extend_from_slice(&exceptions);
        std::fs::write(
            tensors_dir.join("model-00001-of-00001.safetensors"),
            &shard_file,
        )
        .unwrap();

        let index_json = serde_json::json!({
            "metadata": {"total_size": weight_bytes.len() + expected.len()},
            "weight_map": {
                "layers.0.ffn.experts.0.w1.weight": "model-00001-of-00001.safetensors",
                "layers.0.ffn.experts.0.w1.scale": "model-00001-of-00001.safetensors"
            }
        });
        std::fs::write(
            meta_dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&index_json).unwrap(),
        )
        .unwrap();

        let loader = SafetensorsMmapLoader::open(dir.path()).unwrap();
        let scale_info = loader
            .get_tensor_info("layers.0.ffn.experts.0.w1.scale")
            .unwrap();
        assert_eq!(scale_info.dtype, "F8_E8M0");
        assert_eq!(scale_info.shape, vec![rows, cols]);
        let loaded_scale = loader
            .get_tensor_bytes("layers.0.ffn.experts.0.w1.scale")
            .unwrap();
        assert_eq!(loaded_scale, expected.as_slice());
    }

    #[test]
    #[ignore = "requires SINGULARITY_V41_INDEX pointing to the pinned checkpoint index"]
    fn strict_index_parses_pinned_checkpoint_inventory() {
        use sha2::{Digest, Sha256};
        let bytes = std::fs::read(std::env::var("SINGULARITY_V41_INDEX").unwrap()).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
        );
        let StrictJson(root) = serde_json::from_slice(&bytes).unwrap();
        let manifest: IndexManifest = serde_json::from_value(root).unwrap();
        assert_eq!(manifest.weight_map.len(), 96085);
        assert_eq!(
            manifest.weight_map.values().collect::<HashSet<_>>().len(),
            48
        );
        assert_eq!(
            manifest.metadata["total_size"].as_u64(),
            Some(510_286_023_000)
        );
    }
}
