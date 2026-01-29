//! FacePass Core Library
//!
//! Core functionality for face detection, recognition, and authentication.

pub mod alignment;
pub mod camera;
pub mod config;
pub mod detection;
pub mod error;
pub mod matching;
pub mod models;
pub mod recognition;
pub mod security;
pub mod storage;

pub use config::Config;
pub use error::{Error, Result};
pub use models::{FaceData, FaceRecord};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::config::Config;
    pub use crate::error::{Error, Result};
    pub use crate::models::{FaceData, FaceRecord};
}
