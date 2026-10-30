use serde::{de::DeserializeOwned, Serialize};
use singularity_core::SingularityError;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::info;
use uuid::Uuid;

pub struct TransactionalStore {
    base_dir: PathBuf,
}

impl TransactionalStore {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self, SingularityError> {
        let path = base_dir.as_ref().to_path_buf();
        fs::create_dir_all(&path)?;
        let store = Self { base_dir: path };
        store.sweep_stale_temporary_files()?;
        Ok(store)
    }

    /// Atomically persists an object to disk via temp-file, fsync, and POSIX rename.
    pub fn atomic_save<T: Serialize>(
        &self,
        filename: &str,
        data: &T,
    ) -> Result<(), SingularityError> {
        let target_path = self.base_dir.join(filename);
        let temp_filename = format!(".tmp.{}.{}", filename, Uuid::new_v4());
        let temp_path = self.base_dir.join(temp_filename);

        let json_bytes = serde_json::to_vec_pretty(data)?;

        // Write to temporary file
        let mut file = File::create(&temp_path)?;
        file.write_all(&json_bytes)?;
        file.sync_all()?; // Fsync guarantee
        drop(file);

        // Atomic POSIX rename replaces target file atomically
        fs::rename(&temp_path, &target_path)?;

        Ok(())
    }

    pub fn load<T: DeserializeOwned>(&self, filename: &str) -> Result<Option<T>, SingularityError> {
        let path = self.base_dir.join(filename);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        let data: T = serde_json::from_slice(&bytes)?;
        Ok(Some(data))
    }

    /// Sweeps stale temporary files left over from prior unexpected process crashes.
    pub fn sweep_stale_temporary_files(&self) -> Result<usize, SingularityError> {
        let mut swept = 0;
        if let Ok(entries) = fs::read_dir(&self.base_dir) {
            for entry in entries.flatten() {
                if let Ok(name) = entry.file_name().into_string() {
                    if name.starts_with(".tmp.") {
                        if let Err(e) = fs::remove_file(entry.path()) {
                            tracing::warn!("Failed to sweep stale temporary file {}: {}", name, e);
                        } else {
                            swept += 1;
                        }
                    }
                }
            }
        }
        if swept > 0 {
            info!(
                "Swept {} stale temporary files during startup recovery",
                swept
            );
        }
        Ok(swept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_transactional_store_save_and_load() {
        let temp_dir = tempdir().unwrap();
        let store = TransactionalStore::new(temp_dir.path()).unwrap();

        let data = vec!["a".to_string(), "b".to_string()];
        store.atomic_save("test.json", &data).unwrap();

        let loaded: Option<Vec<String>> = store.load("test.json").unwrap();
        assert_eq!(loaded, Some(data));
    }
}
