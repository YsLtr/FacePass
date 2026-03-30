//! Face data storage module

use crate::error::{Error, Result};
use crate::models::{FaceData, FaceGroupMetadata, FaceGroupSummary, FaceRecord, UserMetadata};
use std::collections::HashSet;
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
        fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir })
    }

    /// Get the user's face data directory
    pub fn user_dir(&self, username: &str) -> PathBuf {
        self.base_dir.join(username)
    }

    fn groups_dir(&self, username: &str) -> PathBuf {
        self.user_dir(username).join("groups")
    }

    fn group_dir(&self, username: &str, group_id: &Uuid) -> PathBuf {
        self.groups_dir(username).join(group_id.to_string())
    }

    fn face_path(&self, username: &str, group_id: &Uuid, id: &Uuid) -> PathBuf {
        self.group_dir(username, group_id).join(format!("{}.face", id))
    }

    /// Get path to user metadata file
    fn metadata_path(&self, username: &str) -> PathBuf {
        self.user_dir(username).join("metadata.json")
    }

    fn group_metadata_path(&self, username: &str, group_id: &Uuid) -> PathBuf {
        self.group_dir(username, group_id).join("metadata.json")
    }

    /// Check if user has any registered faces
    pub fn has_faces(&self, username: &str) -> bool {
        self.total_face_count(username).unwrap_or(0) > 0
    }

    /// Get user metadata, creating in-memory defaults if not present
    pub fn get_metadata(&self, username: &str) -> Result<UserMetadata> {
        let path = self.metadata_path(username);
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(UserMetadata::new(username))
        }
    }

    /// Save user metadata
    pub fn save_metadata(&self, metadata: &UserMetadata) -> Result<()> {
        let user_dir = self.user_dir(&metadata.username);
        fs::create_dir_all(self.groups_dir(&metadata.username))?;
        fs::create_dir_all(&user_dir)?;

        let content = serde_json::to_string_pretty(metadata)?;
        fs::write(self.metadata_path(&metadata.username), content)?;
        Ok(())
    }

    /// Get full metadata for a single group
    pub fn get_group_metadata(&self, username: &str, group_id: &Uuid) -> Result<FaceGroupMetadata> {
        let path = self.group_metadata_path(username, group_id);
        if !path.exists() {
            return Err(Error::Storage(format!("Face group not found: {}", group_id)));
        }

        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    fn save_group_metadata(&self, username: &str, group: &FaceGroupMetadata) -> Result<()> {
        let group_dir = self.group_dir(username, &group.id);
        fs::create_dir_all(&group_dir)?;
        let content = serde_json::to_string_pretty(group)?;
        fs::write(self.group_metadata_path(username, &group.id), content)?;
        Ok(())
    }

    /// Create a new named group for a user
    pub fn create_group(&self, username: &str, name: &str) -> Result<FaceGroupSummary> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(Error::Storage("Group name cannot be empty".to_string()));
        }
        if trimmed.chars().all(|ch| ch.is_ascii_digit()) {
            return Err(Error::Storage(
                "Purely numeric group names are not allowed".to_string(),
            ));
        }

        let mut metadata = self.get_metadata(username)?;
        if metadata.groups.iter().any(|group| group.name == trimmed) {
            return Err(Error::Storage(format!("Group '{}' already exists", trimmed)));
        }

        let group = FaceGroupMetadata::new(trimmed);
        let summary = group.summary();
        metadata.upsert_group(summary.clone());
        if metadata.default_group_id.is_none() {
            metadata.default_group_id = Some(summary.id);
        }

        self.save_group_metadata(username, &group)?;
        self.save_metadata(&metadata)?;
        Ok(summary)
    }

    /// Ensure the user has a default group, creating one named after the user if needed
    pub fn ensure_default_group(&self, username: &str) -> Result<FaceGroupSummary> {
        let metadata = self.get_metadata(username)?;
        if metadata.groups.is_empty() {
            return self.create_group(username, username);
        }

        if let Some(group) = metadata.default_group() {
            if metadata.default_group_id == Some(group.id) {
                return Ok(group.clone());
            }
        }

        let mut metadata = metadata;
        let first = metadata
            .groups
            .first()
            .cloned()
            .ok_or_else(|| Error::Storage(format!("User '{}' has no face groups", username)))?;
        metadata.default_group_id = Some(first.id);
        self.save_metadata(&metadata)?;
        Ok(first)
    }

    /// Get the user's current default group without creating one
    pub fn get_default_group(&self, username: &str) -> Result<FaceGroupSummary> {
        let metadata = self.get_metadata(username)?;
        metadata.default_group().cloned().ok_or_else(|| {
            Error::Storage(format!("User '{}' has no default face group", username))
        })
    }

    /// List all known groups for a user
    pub fn list_groups(&self, username: &str) -> Result<Vec<FaceGroupSummary>> {
        Ok(self.get_metadata(username)?.groups)
    }

    /// Resolve a group selector (index or exact name) for a user
    pub fn resolve_group(&self, username: &str, selector: &str) -> Result<FaceGroupSummary> {
        let metadata = self.get_metadata(username)?;
        if metadata.groups.is_empty() {
            return Err(Error::Storage(format!("User '{}' has no face groups", username)));
        }

        if let Ok(index) = selector.parse::<usize>() {
            return metadata.groups.get(index).cloned().ok_or_else(|| {
                Error::Storage(format!(
                    "Invalid group index {}. Valid range: 0-{}",
                    index,
                    metadata.groups.len().saturating_sub(1)
                ))
            });
        }

        metadata
            .groups
            .iter()
            .find(|group| group.name == selector)
            .cloned()
            .ok_or_else(|| Error::Storage(format!("Face group '{}' not found", selector)))
    }

    /// Find a group by exact name
    pub fn find_group_by_name(&self, username: &str, name: &str) -> Result<Option<FaceGroupSummary>> {
        Ok(self
            .get_metadata(username)?
            .groups
            .into_iter()
            .find(|group| group.name == name))
    }

    /// Set the user's default group
    pub fn set_default_group(&self, username: &str, group_id: &Uuid) -> Result<()> {
        let mut metadata = self.get_metadata(username)?;
        if !metadata.groups.iter().any(|group| group.id == *group_id) {
            return Err(Error::Storage(format!("Face group not found: {}", group_id)));
        }
        metadata.default_group_id = Some(*group_id);
        self.save_metadata(&metadata)
    }

    /// Save a face record into an existing group
    pub fn save_face(&self, record: &FaceRecord) -> Result<()> {
        let mut group = self.get_group_metadata(&record.username, &record.group_id)?;
        let path = self.face_path(&record.username, &record.group_id, &record.id);
        let encoded = bincode::serialize(record)?;
        fs::write(path, encoded)?;

        group.add_face(record.id);
        self.save_group_metadata(&record.username, &group)?;

        let mut metadata = self.get_metadata(&record.username)?;
        metadata.upsert_group(group.summary());
        if metadata.default_group_id.is_none() {
            metadata.default_group_id = Some(group.id);
        }
        self.save_metadata(&metadata)?;
        Ok(())
    }

    /// Load a face record by ID from a group
    pub fn load_face(&self, username: &str, group_id: &Uuid, id: &Uuid) -> Result<FaceRecord> {
        let path = self.face_path(username, group_id, id);
        if !path.exists() {
            return Err(Error::Storage(format!("Face record not found: {}", id)));
        }
        let data = fs::read(path)?;
        Ok(bincode::deserialize(&data)?)
    }

    /// Load all faces for a specific group
    pub fn load_faces_in_group(&self, username: &str, group_id: &Uuid) -> Result<Vec<FaceRecord>> {
        let group = self.get_group_metadata(username, group_id)?;
        let mut records = Vec::new();
        for face_id in group.face_ids {
            if let Ok(record) = self.load_face(username, group_id, &face_id) {
                records.push(record);
            }
        }
        records.sort_by_key(|record| record.created_at);
        Ok(records)
    }

    /// Load all faces across every group for a user
    pub fn load_all_faces(&self, username: &str) -> Result<Vec<FaceRecord>> {
        let mut all = Vec::new();
        for group in self.list_groups(username)? {
            all.extend(self.load_faces_in_group(username, &group.id)?);
        }
        all.sort_by_key(|record| record.created_at);
        Ok(all)
    }

    /// Load face data (features only) from the default group
    pub fn load_face_data(&self, username: &str) -> Result<Vec<FaceData>> {
        self.load_default_face_data(username)
    }

    /// Load face data (features only) for a specific group
    pub fn load_face_data_in_group(&self, username: &str, group_id: &Uuid) -> Result<Vec<FaceData>> {
        let records = self.load_faces_in_group(username, group_id)?;
        Ok(records.into_iter().map(|record| record.data).collect())
    }

    /// Load face data (features only) for the default group
    pub fn load_default_face_data(&self, username: &str) -> Result<Vec<FaceData>> {
        let group = self.get_default_group(username)?;
        self.load_face_data_in_group(username, &group.id)
    }

    /// Resolve one or more face selectors within a group
    pub fn resolve_faces_in_group(
        &self,
        username: &str,
        group_id: &Uuid,
        selectors: &[String],
    ) -> Result<Vec<FaceRecord>> {
        let faces = self.load_faces_in_group(username, group_id)?;
        let mut matched = Vec::new();
        let mut seen = HashSet::new();

        for selector in selectors {
            if let Ok(index) = selector.parse::<usize>() {
                let face = faces.get(index).cloned().ok_or_else(|| {
                    Error::Storage(format!(
                        "Invalid face index {}. Valid range: 0-{}",
                        index,
                        faces.len().saturating_sub(1)
                    ))
                })?;
                if seen.insert(face.id) {
                    matched.push(face);
                }
                continue;
            }

            let label_matches: Vec<_> = faces
                .iter()
                .filter(|face| face.label == *selector)
                .cloned()
                .collect();

            if label_matches.is_empty() {
                return Err(Error::Storage(format!(
                    "No faces with label '{}' in selected group",
                    selector
                )));
            }

            for face in label_matches {
                if seen.insert(face.id) {
                    matched.push(face);
                }
            }
        }

        Ok(matched)
    }

    /// Delete specific faces from a group
    pub fn delete_faces(&self, username: &str, group_id: &Uuid, face_ids: &[Uuid]) -> Result<usize> {
        let mut group = self.get_group_metadata(username, group_id)?;
        let to_delete: HashSet<_> = face_ids.iter().copied().collect();
        let mut deleted = 0usize;

        for face_id in &to_delete {
            let path = self.face_path(username, group_id, face_id);
            if path.exists() {
                fs::remove_file(path)?;
            }
            if group.remove_face(face_id) {
                deleted += 1;
            }
        }

        self.save_group_metadata(username, &group)?;

        let mut metadata = self.get_metadata(username)?;
        metadata.upsert_group(group.summary());
        self.save_metadata(&metadata)?;
        Ok(deleted)
    }

    /// Delete a whole face group
    pub fn delete_group(&self, username: &str, group_id: &Uuid) -> Result<()> {
        let group_dir = self.group_dir(username, group_id);
        if group_dir.exists() {
            fs::remove_dir_all(group_dir)?;
        }

        let mut metadata = self.get_metadata(username)?;
        if !metadata.remove_group(group_id) {
            return Err(Error::Storage(format!("Face group not found: {}", group_id)));
        }

        if metadata.groups.is_empty() {
            self.delete_user(username)?;
            return Ok(());
        }

        if metadata.default_group_id.is_none() {
            metadata.default_group_id = metadata.groups.first().map(|group| group.id);
        }
        self.save_metadata(&metadata)
    }

    /// Delete all data for a user
    pub fn delete_user(&self, username: &str) -> Result<()> {
        let user_dir = self.user_dir(username);
        if user_dir.exists() {
            fs::remove_dir_all(user_dir)?;
        }
        Ok(())
    }

    /// Total face count across all groups for a user
    pub fn total_face_count(&self, username: &str) -> Result<usize> {
        Ok(self.get_metadata(username)?.total_face_count())
    }

    /// Compatibility helper for existing callers
    pub fn face_count(&self, username: &str) -> Result<usize> {
        self.total_face_count(username)
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
            if !path.is_dir() {
                continue;
            }

            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };

            if self.total_face_count(name).unwrap_or(0) > 0 {
                users.push(name.to_string());
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
    fn test_default_group_created_on_demand() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let group = storage.ensure_default_group("testuser").unwrap();
        assert_eq!(group.name, "testuser");
        assert_eq!(storage.get_default_group("testuser").unwrap().id, group.id);
    }

    #[test]
    fn test_save_and_load_face() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let group = storage.ensure_default_group("testuser").unwrap();
        let record = FaceRecord::new("testuser", group.id, "Test Face", vec![0.1; 128]);
        storage.save_face(&record).unwrap();

        let loaded = storage.load_face("testuser", &group.id, &record.id).unwrap();
        assert_eq!(loaded.username, "testuser");
        assert_eq!(loaded.group_id, group.id);
        assert_eq!(loaded.label, "Test Face");
        assert_eq!(loaded.data.feature.len(), 128);
    }

    #[test]
    fn test_create_and_resolve_multiple_groups() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let default_group = storage.ensure_default_group("testuser").unwrap();
        let work_group = storage.create_group("testuser", "work").unwrap();

        assert_eq!(storage.resolve_group("testuser", "0").unwrap().id, default_group.id);
        assert_eq!(storage.resolve_group("testuser", "work").unwrap().id, work_group.id);
    }

    #[test]
    fn test_delete_faces_without_deleting_group() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let group = storage.ensure_default_group("testuser").unwrap();
        let first = FaceRecord::new("testuser", group.id, "one", vec![0.1; 128]);
        let second = FaceRecord::new("testuser", group.id, "two", vec![0.2; 128]);
        storage.save_face(&first).unwrap();
        storage.save_face(&second).unwrap();

        let deleted = storage
            .delete_faces("testuser", &group.id, &[first.id])
            .unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(storage.load_faces_in_group("testuser", &group.id).unwrap().len(), 1);
        assert_eq!(storage.list_groups("testuser").unwrap()[0].face_count, 1);
    }

    #[test]
    fn test_delete_group_updates_default_group() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let default_group = storage.ensure_default_group("testuser").unwrap();
        let second_group = storage.create_group("testuser", "backup").unwrap();
        storage.set_default_group("testuser", &second_group.id).unwrap();

        storage.delete_group("testuser", &second_group.id).unwrap();

        let new_default = storage.get_default_group("testuser").unwrap();
        assert_eq!(new_default.id, default_group.id);
    }
}
