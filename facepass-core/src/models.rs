//! Data models for FacePass

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Face feature descriptor - stores the 128-dimensional feature vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceData {
    /// Face label/name
    pub label: String,
    /// 128-dimensional feature vector from SFace
    pub feature: Vec<f32>,
}

impl FaceData {
    /// Create new FaceData from label and feature vector
    pub fn new(label: impl Into<String>, feature: Vec<f32>) -> Self {
        Self {
            label: label.into(),
            feature,
        }
    }

    /// Get the feature vector as a slice
    pub fn feature_slice(&self) -> &[f32] {
        &self.feature
    }

    /// Validate the feature vector (should be 128 dimensions)
    pub fn is_valid(&self) -> bool {
        self.feature.len() == 128
    }
}

/// Face record stored in the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceRecord {
    /// Unique identifier
    pub id: Uuid,
    /// Username this face belongs to
    pub username: String,
    /// User-provided label for this face
    pub label: String,
    /// Unix timestamp when created
    pub created_at: i64,
    /// The face feature data
    pub data: FaceData,
}

impl FaceRecord {
    /// Create a new face record
    pub fn new(username: impl Into<String>, label: impl Into<String>, feature: Vec<f32>) -> Self {
        let label_str = label.into();
        Self {
            id: Uuid::new_v4(),
            username: username.into(),
            label: label_str.clone(),
            created_at: chrono_timestamp(),
            data: FaceData::new(label_str, feature),
        }
    }

    /// Get the filename for this record
    pub fn filename(&self) -> String {
        format!("{}.face", self.id)
    }
}

/// User metadata for face storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMetadata {
    /// Username
    pub username: String,
    /// Number of registered faces
    pub face_count: usize,
    /// Last update timestamp
    pub last_updated: i64,
    /// List of face IDs
    pub face_ids: Vec<Uuid>,
}

impl UserMetadata {
    /// Create new user metadata
    pub fn new(username: impl Into<String>) -> Self {
        Self {
            username: username.into(),
            face_count: 0,
            last_updated: chrono_timestamp(),
            face_ids: Vec::new(),
        }
    }

    /// Add a face ID
    pub fn add_face(&mut self, id: Uuid) {
        self.face_ids.push(id);
        self.face_count = self.face_ids.len();
        self.last_updated = chrono_timestamp();
    }

    /// Remove a face ID
    pub fn remove_face(&mut self, id: &Uuid) -> bool {
        if let Some(pos) = self.face_ids.iter().position(|x| x == id) {
            self.face_ids.remove(pos);
            self.face_count = self.face_ids.len();
            self.last_updated = chrono_timestamp();
            true
        } else {
            false
        }
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
    /// Create a new authentication request
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
    /// Create a new cancellation request
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
    /// Create a success response
    pub fn success(confidence: f64, label: impl Into<String>) -> Self {
        Self {
            success: true,
            message: "Face recognized".to_string(),
            confidence: Some(confidence),
            matched_label: Some(label.into()),
        }
    }

    /// Create a failure response
    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            confidence: None,
            matched_label: None,
        }
    }

    /// Create a generic response message
    pub fn message(success: bool, message: impl Into<String>) -> Self {
        Self {
            success,
            message: message.into(),
            confidence: None,
            matched_label: None,
        }
    }

    /// Create a timeout response
    pub fn timeout() -> Self {
        Self::failure("Authentication timeout")
    }

    /// Create a no face data response
    pub fn no_face_data() -> Self {
        Self::failure("No face data registered for user")
    }
}

/// Detection result from YuNet
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Bounding box (x, y, width, height)
    pub bbox: (f32, f32, f32, f32),
    /// Detection confidence score
    pub confidence: f32,
    /// 5 facial landmarks: right eye, left eye, nose tip, right mouth corner, left mouth corner
    pub landmarks: [(f32, f32); 5],
}

impl DetectionResult {
    /// Get bounding box as (x, y, width, height)
    pub fn bbox(&self) -> (f32, f32, f32, f32) {
        self.bbox
    }

    /// Get landmarks array
    pub fn landmarks(&self) -> &[(f32, f32); 5] {
        &self.landmarks
    }
}

/// Get current Unix timestamp
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
        let valid = FaceData::new("test", vec![0.0; 128]);
        assert!(valid.is_valid());

        let invalid = FaceData::new("test", vec![0.0; 64]);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_face_record_creation() {
        let record = FaceRecord::new("testuser", "Test Face", vec![0.0; 128]);
        assert_eq!(record.username, "testuser");
        assert_eq!(record.label, "Test Face");
        assert!(record.data.is_valid());
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
