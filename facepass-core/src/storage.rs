//! Face data storage module

use crate::error::{Error, Result};
use crate::models::{FaceData, FaceGroupMetadata, FaceGroupSummary, FaceRecord, UserMetadata};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

const IDENTIFIER_RULE: &str = "[A-Za-z_][A-Za-z0-9_]*";

fn is_ascii_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };

    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

pub fn validate_selector_name(name: &str, kind: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::Storage(format!("{kind} cannot be empty")));
    }

    if !is_ascii_identifier(name) {
        return Err(Error::Storage(format!(
            "{kind} '{}' is invalid. Names must match {}",
            name, IDENTIFIER_RULE
        )));
    }

    Ok(())
}

fn parse_index_selector(selector: &str, scope: &str) -> Option<Result<usize>> {
    if !selector.is_empty() && selector.chars().all(|ch| ch.is_ascii_digit()) {
        return Some(
            selector
                .parse::<usize>()
                .map_err(|_| Error::Storage(format!("Invalid {} '{}'", scope, selector))),
        );
    }

    let index_str = selector
        .strip_prefix('@')
        .or_else(|| selector.strip_prefix('#'))?;
    if !index_str.is_empty() && index_str.chars().all(|ch| ch.is_ascii_digit()) {
        return Some(Err(Error::Storage(format!(
            "Invalid {} '{}'. Use bare numeric indexes like '{}'",
            scope, selector, index_str
        ))));
    }

    None
}

fn default_group_name_for_user(username: &str) -> &str {
    if is_ascii_identifier(username) {
        username
    } else {
        "default"
    }
}

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
        self.group_dir(username, group_id)
            .join(format!("{}.face", id))
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
            return Err(Error::Storage(format!(
                "Face group not found: {}",
                group_id
            )));
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
        validate_selector_name(trimmed, "Group name")?;

        let mut metadata = self.get_metadata(username)?;
        if metadata.groups.iter().any(|group| group.name == trimmed) {
            return Err(Error::Storage(format!(
                "Group '{}' already exists",
                trimmed
            )));
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
            return self.create_group(username, default_group_name_for_user(username));
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
        metadata
            .default_group()
            .cloned()
            .ok_or_else(|| Error::Storage(format!("User '{}' has no default face group", username)))
    }

    /// List all known groups for a user
    pub fn list_groups(&self, username: &str) -> Result<Vec<FaceGroupSummary>> {
        Ok(self.get_metadata(username)?.groups)
    }

    /// Resolve a registered user selector (index or exact username)
    pub fn resolve_user(&self, selector: &str) -> Result<String> {
        let users = self.list_users()?;

        if let Some(index) = parse_index_selector(selector, "user selector") {
            let index = index?;
            if users.is_empty() {
                return Err(Error::Storage(
                    "No registered users with saved face data".to_string(),
                ));
            }
            return users.get(index).cloned().ok_or_else(|| {
                Error::Storage(format!(
                    "Invalid user selector '{}'. Valid range: 0-{}",
                    selector,
                    users.len().saturating_sub(1)
                ))
            });
        }

        users
            .into_iter()
            .find(|user| user == selector)
            .ok_or_else(|| {
                Error::Storage(format!(
                    "User '{}' not found in registered face data",
                    selector
                ))
            })
    }

    /// Resolve a group selector (index or exact name) for a user
    pub fn resolve_group(&self, username: &str, selector: &str) -> Result<FaceGroupSummary> {
        let metadata = self.get_metadata(username)?;
        if metadata.groups.is_empty() {
            return Err(Error::Storage(format!(
                "User '{}' has no face groups",
                username
            )));
        }

        if let Some(index) = parse_index_selector(selector, "group selector") {
            let index = index?;
            return metadata.groups.get(index).cloned().ok_or_else(|| {
                Error::Storage(format!(
                    "Invalid group selector '{}' for user '{}'. Valid range: 0-{}",
                    selector,
                    username,
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
    pub fn find_group_by_name(
        &self,
        username: &str,
        name: &str,
    ) -> Result<Option<FaceGroupSummary>> {
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
            return Err(Error::Storage(format!(
                "Face group not found: {}",
                group_id
            )));
        }
        metadata.default_group_id = Some(*group_id);
        self.save_metadata(&metadata)
    }

    pub fn face_label_exists(&self, username: &str, group_id: &Uuid, label: &str) -> Result<bool> {
        Ok(self
            .load_faces_in_group(username, group_id)?
            .iter()
            .any(|face| face.label == label))
    }

    pub fn next_available_face_label(&self, username: &str, group_id: &Uuid) -> Result<String> {
        let faces = self.load_faces_in_group(username, group_id)?;
        let labels: HashSet<_> = faces.into_iter().map(|face| face.label).collect();

        let mut index = 1usize;
        loop {
            let candidate = format!("face_{}", index);
            if !labels.contains(&candidate) {
                return Ok(candidate);
            }
            index += 1;
        }
    }

    /// Save a face record into an existing group
    pub fn save_face(&self, record: &FaceRecord) -> Result<()> {
        validate_selector_name(&record.label, "Face label")?;
        if self.face_label_exists(&record.username, &record.group_id, &record.label)? {
            return Err(Error::Storage(format!(
                "Face label '{}' already exists in group '{}'",
                record.label,
                self.get_group_metadata(&record.username, &record.group_id)?
                    .name
            )));
        }

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
    pub fn load_face_data_in_group(
        &self,
        username: &str,
        group_id: &Uuid,
    ) -> Result<Vec<FaceData>> {
        let records = self.load_faces_in_group(username, group_id)?;
        Ok(records.into_iter().map(|record| record.data).collect())
    }

    /// Load face data (features only) for the default group
    pub fn load_default_face_data(&self, username: &str) -> Result<Vec<FaceData>> {
        let group = self.get_default_group(username)?;
        self.load_face_data_in_group(username, &group.id)
    }

    /// Resolve one or more face selectors (index or exact label) within a group
    pub fn resolve_faces_in_group(
        &self,
        username: &str,
        group_id: &Uuid,
        selectors: &[String],
    ) -> Result<Vec<FaceRecord>> {
        let group = self.get_group_metadata(username, group_id)?;
        let faces = self.load_faces_in_group(username, group_id)?;
        let mut matched = Vec::new();
        let mut seen = HashSet::new();

        for selector in selectors {
            if let Some(index) = parse_index_selector(selector, "face selector") {
                let index = index?;
                if faces.is_empty() {
                    return Err(Error::Storage(format!(
                        "Group '{}' has no faces",
                        group.name
                    )));
                }
                let face = faces.get(index).cloned().ok_or_else(|| {
                    Error::Storage(format!(
                        "Invalid face selector '{}' in group '{}'. Valid range: 0-{}",
                        selector,
                        group.name,
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
                    "No faces with label '{}' in group '{}'",
                    selector, group.name
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
    pub fn delete_faces(
        &self,
        username: &str,
        group_id: &Uuid,
        face_ids: &[Uuid],
    ) -> Result<usize> {
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
            return Err(Error::Storage(format!(
                "Face group not found: {}",
                group_id
            )));
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
        let record = FaceRecord::new("testuser", group.id, "test_face", vec![0.1; 128]);
        storage.save_face(&record).unwrap();

        let loaded = storage
            .load_face("testuser", &group.id, &record.id)
            .unwrap();
        assert_eq!(loaded.username, "testuser");
        assert_eq!(loaded.group_id, group.id);
        assert_eq!(loaded.label, "test_face");
        assert_eq!(loaded.data.feature.len(), 128);
    }

    #[test]
    fn test_create_and_resolve_multiple_groups() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let default_group = storage.ensure_default_group("testuser").unwrap();
        let work_group = storage.create_group("testuser", "work").unwrap();

        assert_eq!(
            storage.resolve_group("testuser", "0").unwrap().id,
            default_group.id
        );
        assert_eq!(
            storage.resolve_group("testuser", "work").unwrap().id,
            work_group.id
        );
    }

    #[test]
    fn test_numeric_group_name_is_rejected() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        storage.ensure_default_group("testuser").unwrap();
        let error = storage
            .create_group("testuser", "123")
            .unwrap_err()
            .to_string();
        assert!(error.contains("Group name '123' is invalid"));
    }

    #[test]
    fn test_resolve_registered_user_by_index() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let alice_group = storage.ensure_default_group("alice").unwrap();
        storage
            .save_face(&FaceRecord::new(
                "alice",
                alice_group.id,
                "normal",
                vec![0.1; 128],
            ))
            .unwrap();

        let bob_group = storage.ensure_default_group("bob").unwrap();
        storage
            .save_face(&FaceRecord::new(
                "bob",
                bob_group.id,
                "normal",
                vec![0.2; 128],
            ))
            .unwrap();

        assert_eq!(storage.resolve_user("0").unwrap(), "alice");
        assert_eq!(storage.resolve_user("1").unwrap(), "bob");
    }

    #[test]
    fn test_resolve_faces_by_index() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let group = storage.ensure_default_group("testuser").unwrap();
        let first = FaceRecord::new("testuser", group.id, "normal", vec![0.1; 128]);
        let second = FaceRecord::new("testuser", group.id, "alt", vec![0.2; 128]);
        let third = FaceRecord::new("testuser", group.id, "alt", vec![0.3; 128]);
        storage.save_face(&first).unwrap();
        storage.save_face(&second).unwrap();
        let error = storage.save_face(&third).unwrap_err().to_string();
        assert!(error.contains("Face label 'alt' already exists"));

        let resolved = storage
            .resolve_faces_in_group(
                "testuser",
                &group.id,
                &["1".to_string(), "normal".to_string()],
            )
            .unwrap();

        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].id, second.id);
        assert_eq!(resolved[1].id, first.id);
    }

    #[test]
    fn test_validate_selector_name_rules() {
        validate_selector_name("normal", "Face label").unwrap();
        validate_selector_name("with_glasses", "Face label").unwrap();
        validate_selector_name("_backup1", "Group name").unwrap();

        for invalid in ["123", "@a", "#a", "with-glasses", "hello world", "中文"] {
            let error = validate_selector_name(invalid, "Face label")
                .unwrap_err()
                .to_string();
            assert!(error.contains("Names must match"));
        }
    }

    #[test]
    fn test_default_group_falls_back_when_username_is_not_identifier() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let group = storage.ensure_default_group("user-name").unwrap();
        assert_eq!(group.name, "default");
    }

    #[test]
    fn test_next_available_face_label_skips_existing_numbers() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let group = storage.ensure_default_group("testuser").unwrap();
        storage
            .save_face(&FaceRecord::new(
                "testuser",
                group.id,
                "face_1",
                vec![0.1; 128],
            ))
            .unwrap();
        storage
            .save_face(&FaceRecord::new(
                "testuser",
                group.id,
                "face_3",
                vec![0.2; 128],
            ))
            .unwrap();

        assert_eq!(
            storage
                .next_available_face_label("testuser", &group.id)
                .unwrap(),
            "face_2"
        );
    }

    #[test]
    fn test_same_label_is_allowed_in_different_groups() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let first_group = storage.ensure_default_group("testuser").unwrap();
        let second_group = storage.create_group("testuser", "backup").unwrap();

        storage
            .save_face(&FaceRecord::new(
                "testuser",
                first_group.id,
                "normal",
                vec![0.1; 128],
            ))
            .unwrap();
        storage
            .save_face(&FaceRecord::new(
                "testuser",
                second_group.id,
                "normal",
                vec![0.2; 128],
            ))
            .unwrap();
    }

    #[test]
    fn test_legacy_prefixed_index_selectors_are_rejected() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let group = storage.ensure_default_group("testuser").unwrap();
        storage
            .save_face(&FaceRecord::new(
                "testuser",
                group.id,
                "normal",
                vec![0.1; 128],
            ))
            .unwrap();

        assert!(storage
            .resolve_group("testuser", "@0")
            .unwrap_err()
            .to_string()
            .contains("Use bare numeric indexes"));
        assert!(storage
            .resolve_faces_in_group("testuser", &group.id, &["#0".to_string()])
            .unwrap_err()
            .to_string()
            .contains("Use bare numeric indexes"));
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
        assert_eq!(
            storage
                .load_faces_in_group("testuser", &group.id)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(storage.list_groups("testuser").unwrap()[0].face_count, 1);
    }

    #[test]
    fn test_delete_group_updates_default_group() {
        let dir = tempdir().unwrap();
        let storage = FaceStorage::new(dir.path()).unwrap();

        let default_group = storage.ensure_default_group("testuser").unwrap();
        let second_group = storage.create_group("testuser", "backup").unwrap();
        storage
            .set_default_group("testuser", &second_group.id)
            .unwrap();

        storage.delete_group("testuser", &second_group.id).unwrap();

        let new_default = storage.get_default_group("testuser").unwrap();
        assert_eq!(new_default.id, default_group.id);
    }
}
