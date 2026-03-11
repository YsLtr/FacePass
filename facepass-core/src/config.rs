//! Configuration management for FacePass

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Default configuration file path (system)
pub const DEFAULT_CONFIG_PATH: &str = "/etc/facepass/config.toml";

/// User configuration file path
pub const USER_CONFIG_PATH: &str = ".config/facepass/config.toml";

/// Default data directory
pub const DEFAULT_DATA_DIR: &str = "/var/lib/facepass/faces";

/// Default models directory
pub const DEFAULT_MODELS_DIR: &str = "/usr/share/facepass/models";

/// Default socket path
pub const DEFAULT_SOCKET_PATH: &str = "/run/facepass/facepass.sock";

/// Video/camera configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    /// Camera device path
    #[serde(default = "default_device")]
    pub device: String,

    /// Authentication timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout: u32,

    /// Maximum frames to try
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,

    /// Frame width (0 = auto)
    #[serde(default = "default_frame_width")]
    pub frame_width: u32,

    /// Frame height (0 = auto)
    #[serde(default = "default_frame_height")]
    pub frame_height: u32,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            device: default_device(),
            timeout: default_timeout(),
            max_frames: default_max_frames(),
            frame_width: default_frame_width(),
            frame_height: default_frame_height(),
        }
    }
}

fn default_device() -> String {
    "/dev/video0".to_string()
}
fn default_timeout() -> u32 {
    5
}
fn default_max_frames() -> u32 {
    30
}
fn default_frame_width() -> u32 {
    640
}
fn default_frame_height() -> u32 {
    480
}

/// Face detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// YuNet detection confidence threshold
    #[serde(default = "default_score_threshold")]
    pub score_threshold: f32,

    /// Non-maximum suppression threshold
    #[serde(default = "default_nms_threshold")]
    pub nms_threshold: f32,

    /// Input width for YuNet model
    #[serde(default = "default_input_width")]
    pub input_width: i32,

    /// Input height for YuNet model
    #[serde(default = "default_input_height")]
    pub input_height: i32,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            score_threshold: default_score_threshold(),
            nms_threshold: default_nms_threshold(),
            input_width: default_input_width(),
            input_height: default_input_height(),
        }
    }
}

fn default_score_threshold() -> f32 {
    0.9
}
fn default_nms_threshold() -> f32 {
    0.3
}
fn default_input_width() -> i32 {
    320
}
fn default_input_height() -> i32 {
    320
}

/// Face recognition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionConfig {
    /// SFace cosine similarity threshold
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f64,

    /// Maximum faces per user
    #[serde(default = "default_max_faces_per_user")]
    pub max_faces_per_user: u32,

    /// Required consecutive matches
    #[serde(default = "default_required_matches")]
    pub required_matches: u32,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: default_similarity_threshold(),
            max_faces_per_user: default_max_faces_per_user(),
            required_matches: default_required_matches(),
        }
    }
}

fn default_similarity_threshold() -> f64 {
    0.4
}
fn default_max_faces_per_user() -> u32 {
    5
}
fn default_required_matches() -> u32 {
    1
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Skip face recognition for SSH sessions
    #[serde(default = "default_ignore_ssh")]
    pub ignore_ssh: bool,

    /// Skip when laptop lid is closed
    #[serde(default = "default_ignore_closed_lid")]
    pub ignore_closed_lid: bool,

    /// Show desktop notification
    #[serde(default)]
    pub show_notification: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            ignore_ssh: default_ignore_ssh(),
            ignore_closed_lid: default_ignore_closed_lid(),
            show_notification: false,
        }
    }
}

fn default_ignore_ssh() -> bool {
    true
}
fn default_ignore_closed_lid() -> bool {
    true
}

/// Daemon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// Unix socket path
    #[serde(default = "default_socket_path")]
    pub socket_path: String,

    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// PID file path
    #[serde(default = "default_pid_file")]
    pub pid_file: String,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            socket_path: default_socket_path(),
            log_level: default_log_level(),
            pid_file: default_pid_file(),
        }
    }
}

fn default_socket_path() -> String {
    DEFAULT_SOCKET_PATH.to_string()
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_pid_file() -> String {
    "/run/facepass/facepass.pid".to_string()
}

/// Model paths configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    /// Path to YuNet model
    #[serde(default = "default_yunet_path")]
    pub yunet_path: String,

    /// Path to SFace model
    #[serde(default = "default_sface_path")]
    pub sface_path: String,

    /// Path to anti-spoofing model
    #[serde(default = "default_anti_spoof_path")]
    pub anti_spoof_path: String,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            yunet_path: default_yunet_path(),
            sface_path: default_sface_path(),
            anti_spoof_path: default_anti_spoof_path(),
        }
    }
}

fn default_yunet_path() -> String {
    format!("{}/face_detection_yunet_2023mar.onnx", DEFAULT_MODELS_DIR)
}
fn default_sface_path() -> String {
    format!(
        "{}/face_recognition_sface_2021dec.onnx",
        DEFAULT_MODELS_DIR
    )
}
fn default_anti_spoof_path() -> String {
    format!(
        "{}/anti_spoof_minifasnetv2se.onnx",
        DEFAULT_MODELS_DIR
    )
}

/// Anti-spoofing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiSpoofConfig {
    /// Enable anti-spoofing detection
    #[serde(default = "default_anti_spoof_enabled")]
    pub enabled: bool,

    /// Liveness score threshold (0.0-1.0, higher = stricter)
    #[serde(default = "default_liveness_threshold")]
    pub threshold: f32,

    /// Input size for anti-spoofing model (width = height)
    #[serde(default = "default_anti_spoof_input_size")]
    pub input_size: i32,
}

impl Default for AntiSpoofConfig {
    fn default() -> Self {
        Self {
            enabled: default_anti_spoof_enabled(),
            threshold: default_liveness_threshold(),
            input_size: default_anti_spoof_input_size(),
        }
    }
}

fn default_anti_spoof_enabled() -> bool {
    true
}
fn default_liveness_threshold() -> f32 {
    0.5
}
fn default_anti_spoof_input_size() -> i32 {
    80
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Data directory for face data
    #[serde(default = "default_data_dir")]
    pub data_dir: String,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
        }
    }
}

fn default_data_dir() -> String {
    DEFAULT_DATA_DIR.to_string()
}

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub video: VideoConfig,

    #[serde(default)]
    pub detection: DetectionConfig,

    #[serde(default)]
    pub recognition: RecognitionConfig,

    #[serde(default)]
    pub security: SecurityConfig,

    #[serde(default)]
    pub daemon: DaemonConfig,

    #[serde(default)]
    pub models: ModelsConfig,

    #[serde(default)]
    pub anti_spoof: AntiSpoofConfig,

    #[serde(default)]
    pub storage: StorageConfig,
}

impl Config {
    /// Load configuration from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            Error::Config(format!(
                "Failed to read config file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;

        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from default path, or create default if not exists
    /// Priority: ~/.config/facepass/config.toml > /etc/facepass/config.toml > default
    pub fn load_or_default() -> Self {
        // Try user config first
        if let Some(home) = std::env::var_os("HOME") {
            let user_config = PathBuf::from(home).join(USER_CONFIG_PATH);
            if user_config.exists() {
                if let Ok(config) = Self::load(&user_config) {
                    return config;
                }
            }
        }

        // Try system config
        if Path::new(DEFAULT_CONFIG_PATH).exists() {
            if let Ok(config) = Self::load(DEFAULT_CONFIG_PATH) {
                return config;
            }
        }

        // Return default
        Self::default()
    }

    /// Get the user config path
    pub fn user_config_path() -> Option<PathBuf> {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(USER_CONFIG_PATH))
    }

    /// Save configuration to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("Failed to serialize config: {}", e)))?;

        // Ensure parent directory exists
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get the user's face data directory
    pub fn user_data_dir(&self, username: &str) -> PathBuf {
        PathBuf::from(&self.storage.data_dir).join(username)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Check model files exist
        if !Path::new(&self.models.yunet_path).exists() {
            return Err(Error::Config(format!(
                "YuNet model not found: {}",
                self.models.yunet_path
            )));
        }

        if !Path::new(&self.models.sface_path).exists() {
            return Err(Error::Config(format!(
                "SFace model not found: {}",
                self.models.sface_path
            )));
        }

        // Validate thresholds
        if self.detection.score_threshold < 0.0 || self.detection.score_threshold > 1.0 {
            return Err(Error::Config(
                "score_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.recognition.similarity_threshold < 0.0 || self.recognition.similarity_threshold > 1.0
        {
            return Err(Error::Config(
                "similarity_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.anti_spoof.threshold < 0.0 || self.anti_spoof.threshold > 1.0 {
            return Err(Error::Config(
                "anti_spoof.threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.anti_spoof.input_size <= 0 || self.anti_spoof.input_size > 512 {
            return Err(Error::Config(
                "anti_spoof.input_size must be between 1 and 512".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn missing_model_path(name: &str) -> String {
        PathBuf::from("/tmp")
            .join(format!("facepass-test-missing-{name}.onnx"))
            .display()
            .to_string()
    }

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.video.timeout, 5);
        assert_eq!(config.detection.score_threshold, 0.9);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.video.timeout, config.video.timeout);
    }

    #[test]
    fn test_validate_allows_missing_anti_spoof_model() {
        let mut config = Config::default();
        config.anti_spoof.enabled = true;
        config.models.anti_spoof_path = missing_model_path("anti-spoof");

        // Keep the always-required models present so this test only exercises
        // the anti-spoof fallback policy.
        config.models.yunet_path = "/bin/sh".to_string();
        config.models.sface_path = "/bin/sh".to_string();

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_invalid_anti_spoof_threshold() {
        let mut config = Config::default();
        config.models.yunet_path = "/bin/sh".to_string();
        config.models.sface_path = "/bin/sh".to_string();
        config.anti_spoof.threshold = 1.5;

        let err = config.validate().unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(err
            .to_string()
            .contains("anti_spoof.threshold must be between 0.0 and 1.0"));
    }
}
