//! Face data storage module

use crate::error::{Error, Result};
use crate::models::{FaceData, FaceRecord, UserMetadata};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Face data storage manager
pub struct FaceStorage {
    base_dir: PathBuf,
}

impl FaceStorage {
    /// Create a new storage manager
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Ensure base directory exists
        fs::create_dir_all(&base_dir)?;

        Ok(Self { base_dir })
    }

    /// Get the user's face data directory
    pub fn user_dir(&self, username: &str) -> PathBuf {
        self.base_dir.join(username)
    }

    /// Get path to a face record file
    fn face_path(&self, username: &str, id: &Uuid) -> PathBuf {
        self.user_dir(username).join(format!("{}.face", id))
    }

    /// Get path to user metadata file
    fn metadata_path(&self, username: &str) -> PathBuf {
        self.user_dir(username).join("metadata.json")
    }

    /// Check if user has any registered faces
    pub fn has_faces(&self, username: &str) -> bool {
        let user_dir = self.user_dir(username);
        if !user_dir.exists() {
            return false;
        }

        // Check for any .face files
        if let Ok(entries) = fs::read_dir(&user_dir) {
            for entry in entries.flatten() {
                if entry.path().extension().map_or(false, |e| e == "face") {
                    return true;
                }
            }
        }

        false
    }

    /// Get user metadata, creating if not exists
    pub fn get_metadata(&self, username: &str) -> Result<UserMetadata> {
        let path = self.metadata_path(username);

        if path.exists() {
            let content = fs::read_to_string(&path)?;
            let metadata: UserMetadata = serde_json::from_str(&content)?;
            Ok(metadata)
        } else {
            Ok(UserMetadata::new(username))
        }
    }

    /// Save user metadata
    pub fn save_metadata(&self, metadata: &UserMetadata) -> Result<()> {
        let user_dir = self.user_dir(&metadata.username);
        fs::create_dir_all(&user_dir)?;

        let path = self.metadata_path(&metadata.username);
        let content = serde_json::to_string_pretty(metadata)?;
        fs::write(path, content)?;

        Ok(())
    }

    /// Save a face record
    pub fn save_face(&self, record: &FaceRecord) -> Result<()> {
        let user_dir = self.user_dir(&record.username);
        fs::create_dir_all(&user_dir)?;

        // Save the face data
        let path = self.face_path(&record.username, &record.id);
        let encoded = bincode::serialize(record)?;
        fs::write(path, encoded)?;

        // Update metadata
        let mut metadata = self.get_metadata(&record.username)?;
        metadata.add_face(record.id);
        self.save_metadata(&metadata)?;

        Ok(())
    }

    /// Load a face record by ID
    pub fn load_face(&self, username: &str, id: &Uuid) -> Result<FaceRecord> {
        let path = self.face_path(username, id);

        if !path.exists() {
            return Err(Error::Storage(format!("Face record not found: {}", id)));
        }

        let data = fs::read(&path)?;
        let record: FaceRecord = bincode::deserialize(&data)?;

        Ok(record)
    }

    /// Load all faces for a user
    pub fn load_all_faces(&self, username: &str) -> Result<Vec<FaceRecord>> {
        let user_dir = self.user_dir(username);

        if !user_dir.exists() {
            return Ok(Vec::new());
        }

        let mut records = Vec::new();

        for entry in fs::read_dir(&user_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map_or(false, |e| e == "face") {
                let data = fs::read(&path)?;
                if let Ok(record) = bincode::deserialize::<FaceRecord>(&data) {
                    records.push(record);
                }
            }
        }

        // Sort by creation time (oldest first)
        records.sort_by_key(|r| r.created_at);

        Ok(records)
    }

    /// Load all face data (features only) for a user
    pub fn load_face_data(&self, username: &str) -> Result<Vec<FaceData>> {
        let records = self.load_all_faces(username)?;
        Ok(records.into_iter().map(|r| r.data).collect())
    }

    /// Delete a face record
    pub fn delete_face(&self, username: &str, id: &Uuid) -> Result<()> {
        let path = self.face_path(username, id);

        if path.exists() {
            fs::remove_file(&path)?;
        }

        // Update metadata
        let mut metadata = self.get_metadata(username)?;
        metadata.remove_face(id);
        self.save_metadata(&metadata)?;

        Ok(())
    }

    /// Delete all faces for a user
    pub fn delete_all_faces(&self, username: &str) -> Result<()> {
        let user_dir = self.user_dir(username);

        if user_dir.exists() {
            fs::remove_dir_all(&user_dir)?;
        }

        Ok(())
    }

    /// Get face count for a user
    pub fn face_count(&self, username: &str) -> Result<usize> {
        let metadata = self.get_metadata(username)?;
        Ok(metadata.face_count)
    }

    /// List all users with registered faces
    pub fn list_users(&self) -> Result<Vec<String>> {
        let mut users = Vec::new();

        if !self.base_dir.exists() {
            return Ok(users);
        }

        for entry in fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name() {
                    if let Some(username) = name.to_str() {
                        // Check if user has any face files
                        if self.has_faces(username) {
                            users.push(username.to_string());
                        }
                    }
                }
            }
        }

        users.sort();
        Ok(users)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_storage_create() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();
        assert!(storage.base_dir.exists());
    }

    #[test]
    fn test_save_and_load_face() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let record = FaceRecord::new("testuser", "Test Face", vec![0.1; 128]);
        storage.save_face(&record).unwrap();

        // Load it back
        let loaded = storage.load_face("testuser", &record.id).unwrap();
        assert_eq!(loaded.username, "testuser");
        assert_eq!(loaded.label, "Test Face");
        assert_eq!(loaded.data.feature.len(), 128);
    }

    #[test]
    fn test_load_all_faces() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        // Save multiple faces
        for i in 0..3 {
            let record = FaceRecord::new("testuser", format!("Face {}", i), vec![i as f32; 128]);
            storage.save_face(&record).unwrap();
        }

        let faces = storage.load_all_faces("testuser").unwrap();
        assert_eq!(faces.len(), 3);
    }

    #[test]
    fn test_delete_face() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let record = FaceRecord::new("testuser", "Test Face", vec![0.1; 128]);
        let id = record.id;
        storage.save_face(&record).unwrap();

        assert!(storage.has_faces("testuser"));

        storage.delete_face("testuser", &id).unwrap();

        assert!(!storage.has_faces("testuser"));
    }

    #[test]
    fn test_list_users() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        // Add faces for multiple users
        for user in &["alice", "bob", "charlie"] {
            let record = FaceRecord::new(*user, "Face", vec![0.1; 128]);
            storage.save_face(&record).unwrap();
        }

        let users = storage.list_users().unwrap();
        assert_eq!(users.len(), 3);
        assert!(users.contains(&"alice".to_string()));
        assert!(users.contains(&"bob".to_string()));
        assert!(users.contains(&"charlie".to_string()));
    }
}
