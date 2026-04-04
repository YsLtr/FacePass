//! Data models for FacePass

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Canonical 5-point face landmarks in ArcFace order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceLandmarks {
    pub left_eye: (f32, f32),
    pub right_eye: (f32, f32),
    pub nose: (f32, f32),
    pub mouth_left: (f32, f32),
    pub mouth_right: (f32, f32),
}

impl FaceLandmarks {
    pub fn from_arcface_order(points: [(f32, f32); 5]) -> Self {
        Self {
            left_eye: points[0],
            right_eye: points[1],
            nose: points[2],
            mouth_left: points[3],
            mouth_right: points[4],
        }
    }

    pub fn from_yunet_order(points: [(f32, f32); 5]) -> Self {
        Self {
            right_eye: points[0],
            left_eye: points[1],
            nose: points[2],
            mouth_right: points[3],
            mouth_left: points[4],
        }
    }

    pub fn arcface_points(&self) -> [(f32, f32); 5] {
        [
            self.left_eye,
            self.right_eye,
            self.nose,
            self.mouth_left,
            self.mouth_right,
        ]
    }
}

/// Unified face detection result.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionResult {
    /// Bounding box (x, y, width, height)
    pub bbox: (f32, f32, f32, f32),
    /// Detection confidence score
    pub confidence: f32,
    /// Canonical 5 facial landmarks in ArcFace order
    pub landmarks: FaceLandmarks,
}

impl DetectionResult {
    pub fn bbox(&self) -> (f32, f32, f32, f32) {
        self.bbox
    }

    pub fn landmarks(&self) -> &FaceLandmarks {
        &self.landmarks
    }

    pub fn bbox_xyxy(&self) -> (f32, f32, f32, f32) {
        let (x, y, w, h) = self.bbox;
        (x, y, x + w, y + h)
    }
}

/// Fresh embedding extracted by the active recognizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceEmbedding {
    pub model_id: String,
    pub embedding_dim: usize,
    pub feature: Vec<f32>,
}

impl FaceEmbedding {
    pub fn new(model_id: impl Into<String>, feature: Vec<f32>) -> Self {
        let embedding_dim = feature.len();
        Self {
            model_id: model_id.into(),
            embedding_dim,
            feature,
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.model_id.trim().is_empty()
            && self.embedding_dim > 0
            && self.embedding_dim == self.feature.len()
    }
}

/// Stored face feature descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceData {
    /// Face label/name
    pub label: String,
    /// Embedding model identity
    pub model_id: String,
    /// Stored embedding dimension
    pub embedding_dim: usize,
    /// Embedding vector
    pub feature: Vec<f32>,
}

impl FaceData {
    pub fn new(
        label: impl Into<String>,
        model_id: impl Into<String>,
        feature: Vec<f32>,
    ) -> Self {
        let embedding_dim = feature.len();
        Self {
            label: label.into(),
            model_id: model_id.into(),
            embedding_dim,
            feature,
        }
    }

    pub fn from_embedding(label: impl Into<String>, embedding: FaceEmbedding) -> Self {
        Self {
            label: label.into(),
            model_id: embedding.model_id,
            embedding_dim: embedding.embedding_dim,
            feature: embedding.feature,
        }
    }

    pub fn feature_slice(&self) -> &[f32] {
        &self.feature
    }

    pub fn is_valid(&self) -> bool {
        !self.label.trim().is_empty()
            && !self.model_id.trim().is_empty()
            && self.embedding_dim > 0
            && self.embedding_dim == self.feature.len()
    }
}

/// Face record stored in the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceRecord {
    /// Unique identifier
    pub id: Uuid,
    /// Username this face belongs to
    pub username: String,
    /// Group this face belongs to
    pub group_id: Uuid,
    /// User-provided label for this face
    pub label: String,
    /// Unix timestamp when created
    pub created_at: i64,
    /// The face feature data
    pub data: FaceData,
}

impl FaceRecord {
    pub fn new(
        username: impl Into<String>,
        group_id: Uuid,
        label: impl Into<String>,
        embedding: FaceEmbedding,
    ) -> Self {
        let label_str = label.into();
        Self {
            id: Uuid::new_v4(),
            username: username.into(),
            group_id,
            label: label_str.clone(),
            created_at: chrono_timestamp(),
            data: FaceData::from_embedding(label_str, embedding),
        }
    }

    pub fn filename(&self) -> String {
        format!("{}.face", self.id)
    }
}

/// Lightweight group summary stored in user metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceGroupSummary {
    /// Unique group identifier
    pub id: Uuid,
    /// Human-readable group name
    pub name: String,
    /// Unix timestamp when created
    pub created_at: i64,
    /// Number of faces in the group
    pub face_count: usize,
}

impl FaceGroupSummary {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            created_at: chrono_timestamp(),
            face_count: 0,
        }
    }
}

/// Full group metadata stored alongside the group directory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceGroupMetadata {
    /// Unique group identifier
    pub id: Uuid,
    /// Human-readable group name
    pub name: String,
    /// Unix timestamp when created
    pub created_at: i64,
    /// Face IDs stored in this group
    pub face_ids: Vec<Uuid>,
}

impl FaceGroupMetadata {
    pub fn new(name: impl Into<String>) -> Self {
        let summary = FaceGroupSummary::new(name);
        Self {
            id: summary.id,
            name: summary.name,
            created_at: summary.created_at,
            face_ids: Vec::new(),
        }
    }

    pub fn summary(&self) -> FaceGroupSummary {
        FaceGroupSummary {
            id: self.id,
            name: self.name.clone(),
            created_at: self.created_at,
            face_count: self.face_ids.len(),
        }
    }

    pub fn add_face(&mut self, id: Uuid) {
        self.face_ids.push(id);
    }

    pub fn remove_face(&mut self, id: &Uuid) -> bool {
        if let Some(pos) = self.face_ids.iter().position(|x| x == id) {
            self.face_ids.remove(pos);
            true
        } else {
            false
        }
    }
}

/// User metadata for face storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMetadata {
    /// Username
    pub username: String,
    /// Default face group for recognition
    pub default_group_id: Option<Uuid>,
    /// Known groups for this user
    pub groups: Vec<FaceGroupSummary>,
    /// Last update timestamp
    pub last_updated: i64,
}

impl UserMetadata {
    pub fn new(username: impl Into<String>) -> Self {
        Self {
            username: username.into(),
            default_group_id: None,
            groups: Vec::new(),
            last_updated: chrono_timestamp(),
        }
    }

    pub fn total_face_count(&self) -> usize {
        self.groups.iter().map(|group| group.face_count).sum()
    }

    pub fn upsert_group(&mut self, group: FaceGroupSummary) {
        if let Some(existing) = self
            .groups
            .iter_mut()
            .find(|existing| existing.id == group.id)
        {
            *existing = group;
        } else {
            self.groups.push(group);
        }
        self.last_updated = chrono_timestamp();
    }

    pub fn remove_group(&mut self, id: &Uuid) -> bool {
        if let Some(pos) = self.groups.iter().position(|group| group.id == *id) {
            self.groups.remove(pos);
            if self.default_group_id == Some(*id) {
                self.default_group_id = self.groups.first().map(|group| group.id);
            }
            self.last_updated = chrono_timestamp();
            true
        } else {
            false
        }
    }

    pub fn default_group(&self) -> Option<&FaceGroupSummary> {
        let default_group_id = self.default_group_id?;
        self.groups
            .iter()
            .find(|group| group.id == default_group_id)
            .or_else(|| self.groups.first())
    }
}

/// IPC Authentication request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthRequest {
    /// Message type
    pub msg_type: String,
    /// Username to authenticate
    pub username: String,
    /// Source of the request (sudo, polkit, login, etc.)
    pub source: String,
    /// Timeout in seconds
    pub timeout: u32,
}

impl AuthRequest {
    pub fn new(username: impl Into<String>, source: impl Into<String>, timeout: u32) -> Self {
        Self {
            msg_type: "auth".to_string(),
            username: username.into(),
            source: source.into(),
            timeout,
        }
    }
}

/// IPC cancellation request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRequest {
    /// Message type
    pub msg_type: String,
}

impl CancelRequest {
    pub fn new() -> Self {
        Self {
            msg_type: "cancel".to_string(),
        }
    }
}

/// IPC Authentication response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthResponse {
    /// Whether authentication succeeded
    pub success: bool,
    /// Result message
    pub message: String,
    /// Similarity confidence score (if matched)
    pub confidence: Option<f64>,
    /// Matched face label (if matched)
    pub matched_label: Option<String>,
}

impl AuthResponse {
    pub fn success(confidence: f64, label: impl Into<String>) -> Self {
        Self {
            success: true,
            message: "Face recognized".to_string(),
            confidence: Some(confidence),
            matched_label: Some(label.into()),
        }
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            confidence: None,
            matched_label: None,
        }
    }

    pub fn message(success: bool, message: impl Into<String>) -> Self {
        Self {
            success,
            message: message.into(),
            confidence: None,
            matched_label: None,
        }
    }

    pub fn timeout() -> Self {
        Self::failure("Authentication timeout")
    }

    pub fn no_face_data() -> Self {
        Self::failure("No face data registered for user")
    }
}

fn chrono_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_data_validation() {
        let valid = FaceData::new("test", "ghostfacenet-512", vec![0.0; 512]);
        assert!(valid.is_valid());

        let invalid = FaceData {
            embedding_dim: 128,
            ..FaceData::new("test", "ghostfacenet-512", vec![0.0; 64])
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_face_record_creation() {
        let group_id = Uuid::new_v4();
        let embedding = FaceEmbedding::new("ghostfacenet-512", vec![0.0; 512]);
        let record = FaceRecord::new("testuser", group_id, "Test Face", embedding);
        assert_eq!(record.username, "testuser");
        assert_eq!(record.group_id, group_id);
        assert_eq!(record.label, "Test Face");
        assert!(record.data.is_valid());
        assert_eq!(record.data.model_id, "ghostfacenet-512");
        assert_eq!(record.data.embedding_dim, 512);
    }

    #[test]
    fn test_face_landmarks_reorders_yunet_output() {
        let landmarks = FaceLandmarks::from_yunet_order([
            (10.0, 11.0),
            (20.0, 21.0),
            (30.0, 31.0),
            (40.0, 41.0),
            (50.0, 51.0),
        ]);

        assert_eq!(landmarks.left_eye, (20.0, 21.0));
        assert_eq!(landmarks.right_eye, (10.0, 11.0));
        assert_eq!(landmarks.mouth_left, (50.0, 51.0));
        assert_eq!(landmarks.mouth_right, (40.0, 41.0));
    }

    #[test]
    fn test_group_metadata_summary() {
        let mut group = FaceGroupMetadata::new("primary");
        let face_id = Uuid::new_v4();
        group.add_face(face_id);

        let summary = group.summary();
        assert_eq!(summary.name, "primary");
        assert_eq!(summary.face_count, 1);
    }

    #[test]
    fn test_auth_response() {
        let success = AuthResponse::success(0.85, "Main face");
        assert!(success.success);
        assert_eq!(success.confidence, Some(0.85));

        let failure = AuthResponse::failure("No match");
        assert!(!failure.success);
    }
}
