use half::{bf16, f16};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use singularity_core::SingularityError;
use singularity_cuda::{GpuBackend, GpuBuffer};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

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

struct ShardEntry {
    _path: PathBuf,
    mmap: Mmap,
    header_len: usize,
    tensors: HashMap<String, TensorInfo>,
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
            let single_path = p.join("model.safetensors");
            if single_path.exists() {
                return Self::open_single_file(&single_path);
            }
            return Err(SingularityError::ModelError {
                reason: format!(
                    "Directory '{}' contains neither model.safetensors.index.json nor model.safetensors",
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
        let manifest: IndexManifest = serde_json::from_reader(file)?;
        let parent_dir = index_path.parent().unwrap_or_else(|| Path::new("."));

        // Collect unique shard filenames in order
        let mut unique_shards: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for shard_name in manifest.weight_map.values() {
            if seen.insert(shard_name.clone()) {
                unique_shards.push(shard_name.clone());
            }
        }

        let mut shards = Vec::new();
        let mut shard_name_to_idx = HashMap::new();

        for (idx, shard_name) in unique_shards.iter().enumerate() {
            let shard_path = parent_dir.join(shard_name);
            let shard = Self::parse_single_shard(&shard_path)?;
            shards.push(shard);
            shard_name_to_idx.insert(shard_name.clone(), idx);
        }

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
        let root: serde_json::Value = serde_json::from_slice(header_bytes)?;
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

        Ok(ShardEntry {
            _path: path.to_path_buf(),
            mmap,
            header_len,
            tensors,
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.weight_to_shard.keys()
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        let shard_idx = *self.weight_to_shard.get(name)?;
        self.shards.get(shard_idx)?.tensors.get(name)
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
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        let index = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index, r#"{"weight_map":{"missing":"model.safetensors"}}"#).unwrap();
        assert!(SafetensorsMmapLoader::open(&index).is_err());
    }
}
