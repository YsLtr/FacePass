//! Error types for FacePass

use thiserror::Error;

/// Result type alias for FacePass operations
pub type Result<T> = std::result::Result<T, Error>;

/// FacePass error types
#[derive(Error, Debug)]
pub enum Error {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Camera error: {0}")]
    Camera(String),

    #[error("Face detection error: {0}")]
    Detection(String),

    #[error("Face recognition error: {0}")]
    Recognition(String),

    #[error("Anti-spoofing error: {0}")]
    AntiSpoofing(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Model not initialized: {0}")]
    ModelNotInitialized(String),

    #[error("No face detected")]
    NoFaceDetected,

    #[error("Face not matched (similarity: {0:.4})")]
    FaceNotMatched(f64),

    #[error("Authentication timeout")]
    Timeout,

    #[error("User has no face data: {0}")]
    NoFaceData(String),

    #[error("Security check failed: {0}")]
    SecurityCheck(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("OpenCV error: {0}")]
    OpenCV(String),

    #[error("IPC error: {0}")]
    Ipc(String),
}

impl From<opencv::Error> for Error {
    fn from(e: opencv::Error) -> Self {
        Error::OpenCV(e.to_string())
    }
}

impl From<bincode::Error> for Error {
    fn from(e: bincode::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}

impl From<toml::de::Error> for Error {
    fn from(e: toml::de::Error) -> Self {
        Error::Config(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}
