//! Configuration management for FacePass

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use toml::{map::Map, Value};

/// Default configuration file path (system)
pub const DEFAULT_CONFIG_PATH: &str = "/etc/facepass/config.toml";

/// User configuration file path
pub const USER_CONFIG_PATH: &str = ".config/facepass/config.toml";

/// Default runtime state file path written by the daemon
pub const DEFAULT_RUNTIME_STATE_PATH: &str = "/run/facepass/runtime-state.json";

/// Built-in preset name used when no preset is explicitly selected
pub const DEFAULT_PRESET_NAME: &str = "default";

/// Default data directory
pub const DEFAULT_DATA_DIR: &str = "/var/lib/facepass/faces";

/// Default models directory
pub const DEFAULT_MODELS_DIR: &str = "/usr/share/facepass/models";

/// Default socket path
pub const DEFAULT_SOCKET_PATH: &str = "/run/facepass/facepass.sock";

/// Parsed on-disk configuration file containing a base config plus named preset overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigFile {
    #[serde(default = "default_active_preset")]
    pub active_preset: String,

    #[serde(default)]
    pub base: Config,

    #[serde(default)]
    pub presets: BTreeMap<String, Value>,
}

impl Default for ConfigFile {
    fn default() -> Self {
        let mut presets = BTreeMap::new();
        presets.insert("dev".to_string(), default_dev_preset());

        Self {
            active_preset: DEFAULT_PRESET_NAME.to_string(),
            base: Config::default(),
            presets,
        }
    }
}

impl ConfigFile {
    /// Load a configuration file without resolving a preset.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            Error::Config(format!(
                "Failed to read config file {}: {}",
                path.display(),
                e
            ))
        })?;

        Ok(toml::from_str(&content)?)
    }

    /// Load a config file and return the resolved source path.
    pub fn load_with_source<P: AsRef<Path>>(path: P) -> Result<(Self, Option<PathBuf>)> {
        let path = path.as_ref();
        Ok((Self::load(path)?, Some(path.to_path_buf())))
    }

    /// Load a config file from the preferred path, then fallback locations.
    pub fn load_with_fallback_and_source<P: AsRef<Path>>(
        preferred_path: P,
    ) -> Result<(Self, Option<PathBuf>)> {
        let preferred = preferred_path.as_ref();
        if preferred.exists() {
            return Self::load_with_source(preferred);
        }

        Self::load_or_default_with_source()
    }

    /// Load a config file from default search locations, or return the built-in default config file.
    pub fn load_or_default_with_source() -> Result<(Self, Option<PathBuf>)> {
        for candidate in Config::workspace_config_candidates() {
            if candidate.exists() {
                return Self::load_with_source(&candidate);
            }
        }

        if let Some(home) = std::env::var_os("HOME") {
            let user_config = PathBuf::from(home).join(USER_CONFIG_PATH);
            if user_config.exists() {
                return Self::load_with_source(&user_config);
            }
        }

        if Path::new(DEFAULT_CONFIG_PATH).exists() {
            return Self::load_with_source(DEFAULT_CONFIG_PATH);
        }

        Ok((Self::default(), None))
    }

    /// Save configuration file to disk.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("Failed to serialize config: {}", e)))?;

        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, content)?;
        Ok(())
    }

    /// List available preset names.
    pub fn list_presets(&self) -> Vec<String> {
        let mut names = vec![DEFAULT_PRESET_NAME.to_string()];
        names.extend(
            self.presets
                .keys()
                .filter(|name| name.as_str() != DEFAULT_PRESET_NAME)
                .cloned(),
        );
        names
    }

    /// Resolve the file's active preset into a final runtime config.
    pub fn resolve_active_preset(&self) -> Result<Config> {
        self.resolve_preset(self.active_preset_name()?.as_str())
    }

    /// Resolve an arbitrary preset name into a final runtime config.
    pub fn resolve_preset(&self, preset_name: &str) -> Result<Config> {
        let preset_name = normalize_preset_name(preset_name)?;
        let mut merged =
            Value::try_from(self.base.clone()).map_err(|e| Error::Config(e.to_string()))?;

        if let Some(overlay) = self.preset_override(preset_name)? {
            merge_toml_value(&mut merged, overlay);
        }

        let config: Config = merged
            .try_into()
            .map_err(|e| Error::Config(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Return the configured active preset, normalized and validated.
    pub fn active_preset_name(&self) -> Result<String> {
        Ok(normalize_preset_name(&self.active_preset)?.to_string())
    }

    /// Update the active preset.
    pub fn set_active_preset(&mut self, preset_name: &str) -> Result<()> {
        let preset_name = normalize_preset_name(preset_name)?;
        self.preset_override(preset_name)?;
        self.active_preset = preset_name.to_string();
        Ok(())
    }

    fn preset_override(&self, preset_name: &str) -> Result<Option<&Value>> {
        if preset_name == DEFAULT_PRESET_NAME {
            return self
                .presets
                .get(preset_name)
                .map(validate_preset_override)
                .transpose();
        }

        let Some(value) = self.presets.get(preset_name) else {
            return Err(Error::Config(format!(
                "Unknown preset '{}'. Available presets: {}",
                preset_name,
                self.list_presets().join(", ")
            )));
        };

        Ok(Some(validate_preset_override(value)?))
    }
}

/// Fully resolved config plus the source file and active preset used to build it.
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub config: Config,
    pub source: Option<PathBuf>,
    pub active_preset: String,
}

/// Runtime state published by the daemon for status reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRuntimeState {
    pub running_preset: String,
    pub socket_path: String,
    pub pid_file: String,
}

impl DaemonRuntimeState {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            Error::Config(format!(
                "Failed to read runtime state file {}: {}",
                path.display(),
                e
            ))
        })?;

        Ok(serde_json::from_str(&content)?)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("Failed to serialize runtime state: {}", e)))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, content)?;
        Ok(())
    }
}

fn default_active_preset() -> String {
    DEFAULT_PRESET_NAME.to_string()
}

fn normalize_preset_name(value: &str) -> Result<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Error::Config("Preset name cannot be empty".to_string()));
    }

    Ok(trimmed)
}

fn validate_preset_override(value: &Value) -> Result<&Value> {
    if !matches!(value, Value::Table(_)) {
        return Err(Error::Config(
            "Preset override must be a TOML table".to_string(),
        ));
    }

    Ok(value)
}

fn merge_toml_value(base: &mut Value, overlay: &Value) {
    match (base, overlay) {
        (Value::Table(base_table), Value::Table(overlay_table)) => {
            for (key, value) in overlay_table {
                match base_table.get_mut(key) {
                    Some(base_value) => merge_toml_value(base_value, value),
                    None => {
                        base_table.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (base_value, overlay_value) => *base_value = overlay_value.clone(),
    }
}

fn default_dev_preset() -> Value {
    let mut root = Map::new();

    root.insert(
        "video".to_string(),
        Value::Table(Map::from_iter([
            (
                "device".to_string(),
                Value::String("/dev/video2".to_string()),
            ),
            ("timeout".to_string(), Value::Integer(5)),
            ("max_frames".to_string(), Value::Integer(50)),
            ("frame_width".to_string(), Value::Integer(640)),
            ("frame_height".to_string(), Value::Integer(480)),
        ])),
    );

    root.insert(
        "recognition".to_string(),
        Value::Table(Map::from_iter([
            ("similarity_threshold".to_string(), Value::Float(0.7)),
            ("consecutive_match_frames".to_string(), Value::Integer(10)),
            ("valid_frames".to_string(), Value::Integer(30)),
            ("valid_crop_scale".to_string(), Value::Float(2.0)),
        ])),
    );

    root.insert(
        "daemon".to_string(),
        Value::Table(Map::from_iter([(
            "log_level".to_string(),
            Value::String("debug".to_string()),
        )])),
    );

    root.insert(
        "models".to_string(),
        Value::Table(Map::from_iter([
            (
                "yunet_path".to_string(),
                Value::String(
                    "/home/ysltr/builds/FacePass/facepass/models/face_detection_yunet_2023mar.onnx"
                        .to_string(),
                ),
            ),
            (
                "sface_path".to_string(),
                Value::String(
                    "/home/ysltr/builds/FacePass/facepass/models/face_recognition_sface_2021dec.onnx"
                        .to_string(),
                ),
            ),
            (
                "anti_spoof_v2_path".to_string(),
                Value::String(
                    "/home/ysltr/builds/FacePass/face-anti-spoofing/weights/MiniFASNetV2.onnx"
                        .to_string(),
                ),
            ),
            (
                "anti_spoof_v1se_path".to_string(),
                Value::String(
                    "/home/ysltr/builds/FacePass/face-anti-spoofing/weights/MiniFASNetV1SE.onnx"
                        .to_string(),
                ),
            ),
        ])),
    );

    root.insert(
        "anti_spoof".to_string(),
        Value::Table(Map::from_iter([(
            "threshold".to_string(),
            Value::Float(0.9),
        )])),
    );

    root.insert(
        "storage".to_string(),
        Value::Table(Map::from_iter([(
            "data_dir".to_string(),
            Value::String("/home/ysltr/builds/FacePass/facepass/data/faces".to_string()),
        )])),
    );

    Value::Table(root)
}

/// Video/camera configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    /// Camera device path
    #[serde(default = "default_device")]
    pub device: String,

    /// Authentication timeout in seconds (0 = unlimited)
    #[serde(default = "default_timeout")]
    pub timeout: u32,

    /// Maximum frames to try (0 = unlimited)
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

    /// Overlap threshold for removing duplicate face boxes
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

    /// Maximum faces per group
    #[serde(default = "default_max_faces_per_group")]
    pub max_faces_per_group: u32,

    /// Required consecutive matched frames
    #[serde(default = "default_consecutive_match_frames")]
    pub consecutive_match_frames: u32,

    /// Minimum number of valid recognition frames required per attempt (0 = disabled)
    #[serde(default = "default_valid_frames")]
    pub valid_frames: u32,

    /// Whether to stop immediately once valid_frames has been reached.
    /// Ignored when valid_frames = 0.
    #[serde(default = "default_stop_on_valid_frames")]
    pub stop_on_valid_frames: bool,

    /// Required scale for the valid-face square crop area (independent of models)
    #[serde(default = "default_valid_crop_scale")]
    pub valid_crop_scale: f32,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: default_similarity_threshold(),
            max_faces_per_group: default_max_faces_per_group(),
            consecutive_match_frames: default_consecutive_match_frames(),
            valid_frames: default_valid_frames(),
            stop_on_valid_frames: default_stop_on_valid_frames(),
            valid_crop_scale: default_valid_crop_scale(),
        }
    }
}

fn default_similarity_threshold() -> f64 {
    0.4
}
fn default_max_faces_per_group() -> u32 {
    5
}
fn default_consecutive_match_frames() -> u32 {
    1
}
fn default_valid_frames() -> u32 {
    5
}
fn default_stop_on_valid_frames() -> bool {
    true
}
fn default_valid_crop_scale() -> f32 {
    2.7
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

    /// Path to MiniFASNetV2 anti-spoofing model
    #[serde(default = "default_anti_spoof_v2_path")]
    pub anti_spoof_v2_path: String,

    /// Path to MiniFASNetV1SE anti-spoofing model
    #[serde(default = "default_anti_spoof_v1se_path")]
    pub anti_spoof_v1se_path: String,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            yunet_path: default_yunet_path(),
            sface_path: default_sface_path(),
            anti_spoof_v2_path: default_anti_spoof_v2_path(),
            anti_spoof_v1se_path: default_anti_spoof_v1se_path(),
        }
    }
}

fn default_yunet_path() -> String {
    format!("{}/face_detection_yunet_2023mar.onnx", DEFAULT_MODELS_DIR)
}
fn default_sface_path() -> String {
    format!("{}/face_recognition_sface_2021dec.onnx", DEFAULT_MODELS_DIR)
}
fn default_anti_spoof_v2_path() -> String {
    format!("{}/MiniFASNetV2.onnx", DEFAULT_MODELS_DIR)
}
fn default_anti_spoof_v1se_path() -> String {
    format!("{}/MiniFASNetV1SE.onnx", DEFAULT_MODELS_DIR)
}

/// Anti-spoofing configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AntiSpoofMode {
    #[serde(rename = "minifasnet_v2")]
    MiniFASNetV2,
    #[serde(rename = "minifasnet_v1se")]
    MiniFASNetV1SE,
    #[serde(rename = "fusion")]
    Fusion,
}

impl AntiSpoofMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MiniFASNetV2 => "minifasnet_v2",
            Self::MiniFASNetV1SE => "minifasnet_v1se",
            Self::Fusion => "fusion",
        }
    }
}

impl Default for AntiSpoofMode {
    fn default() -> Self {
        Self::MiniFASNetV2
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiSpoofConfig {
    /// Enable anti-spoofing detection
    #[serde(default = "default_anti_spoof_enabled")]
    pub enabled: bool,

    /// Liveness score threshold (0.0-1.0, higher = stricter)
    #[serde(default = "default_liveness_threshold")]
    pub threshold: f32,

    /// Anti-spoofing strategy: v2 / v1se / fusion
    #[serde(default = "default_anti_spoof_mode")]
    pub mode: AntiSpoofMode,

    /// Input size for MiniFASNetV2 (width = height)
    #[serde(default = "default_v2_input_size")]
    pub v2_input_size: i32,

    /// Crop scale for MiniFASNetV2
    #[serde(default = "default_v2_crop_scale")]
    pub v2_crop_scale: f32,

    /// Input size for MiniFASNetV1SE (width = height)
    #[serde(default = "default_v1se_input_size")]
    pub v1se_input_size: i32,

    /// Crop scale for MiniFASNetV1SE
    #[serde(default = "default_v1se_crop_scale")]
    pub v1se_crop_scale: f32,
}

impl Default for AntiSpoofConfig {
    fn default() -> Self {
        Self {
            enabled: default_anti_spoof_enabled(),
            threshold: default_liveness_threshold(),
            mode: default_anti_spoof_mode(),
            v2_input_size: default_v2_input_size(),
            v2_crop_scale: default_v2_crop_scale(),
            v1se_input_size: default_v1se_input_size(),
            v1se_crop_scale: default_v1se_crop_scale(),
            // keep per-model crop scales for inference only
        }
    }
}

fn default_anti_spoof_enabled() -> bool {
    true
}
fn default_liveness_threshold() -> f32 {
    0.5
}
fn default_anti_spoof_mode() -> AntiSpoofMode {
    AntiSpoofMode::MiniFASNetV2
}
fn default_v2_input_size() -> i32 {
    80
}
fn default_v2_crop_scale() -> f32 {
    2.7
}
fn default_v1se_input_size() -> i32 {
    80
}
fn default_v1se_crop_scale() -> f32 {
    4.0
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
        Ok(Self::load_with_source(path)?.config)
    }

    /// Load configuration from file and return the resolved source path and active preset.
    pub fn load_with_source<P: AsRef<Path>>(path: P) -> Result<ResolvedConfig> {
        let path = path.as_ref();
        let (config_file, source) = ConfigFile::load_with_source(path)?;
        let active_preset = config_file.active_preset_name()?;
        let config = config_file.resolve_active_preset()?;

        Ok(ResolvedConfig {
            config,
            source,
            active_preset,
        })
    }

    /// Load configuration from a preferred path, then current workspace, then
    /// user/system defaults.
    pub fn load_with_fallback<P: AsRef<Path>>(preferred_path: P) -> Result<Self> {
        Ok(Self::load_with_fallback_and_source(preferred_path)?.config)
    }

    /// Load configuration from a preferred path, then current workspace, then
    /// user/system defaults, and return the resolved source path and active preset.
    pub fn load_with_fallback_and_source<P: AsRef<Path>>(
        preferred_path: P,
    ) -> Result<ResolvedConfig> {
        let preferred = preferred_path.as_ref();
        if preferred.exists() {
            return Self::load_with_source(preferred);
        }

        Self::load_or_default_with_source()
    }

    /// Load configuration from default search locations.
    ///
    /// Priority:
    /// 1. current working directory or parent dirs: config/facepass.toml
    /// 2. ~/.config/facepass/config.toml
    /// 3. /etc/facepass/config.toml
    /// 4. default
    pub fn load_or_default() -> Result<Self> {
        Ok(Self::load_or_default_with_source()?.config)
    }

    /// Load configuration from default search locations and return the
    /// resolved source path and active preset if any.
    pub fn load_or_default_with_source() -> Result<ResolvedConfig> {
        for candidate in Self::workspace_config_candidates() {
            if candidate.exists() {
                return Self::load_with_source(&candidate);
            }
        }

        // Try user config first
        if let Some(home) = std::env::var_os("HOME") {
            let user_config = PathBuf::from(home).join(USER_CONFIG_PATH);
            if user_config.exists() {
                return Self::load_with_source(&user_config);
            }
        }

        // Try system config
        if Path::new(DEFAULT_CONFIG_PATH).exists() {
            return Self::load_with_source(DEFAULT_CONFIG_PATH);
        }

        // Return default
        let config = Self::default();
        config.validate()?;
        Ok(ResolvedConfig {
            config,
            source: None,
            active_preset: DEFAULT_PRESET_NAME.to_string(),
        })
    }

    /// Get the user config path
    pub fn user_config_path() -> Option<PathBuf> {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(USER_CONFIG_PATH))
    }

    /// Save configuration to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        ConfigFile {
            active_preset: DEFAULT_PRESET_NAME.to_string(),
            base: self.clone(),
            presets: BTreeMap::new(),
        }
        .save(path)
    }

    /// Get the user's face data directory
    pub fn user_data_dir(&self, username: &str) -> PathBuf {
        PathBuf::from(&self.storage.data_dir).join(username)
    }

    fn workspace_config_candidates() -> Vec<PathBuf> {
        let mut candidates = Vec::new();
        let mut cursor = match std::env::current_dir() {
            Ok(dir) => Some(dir),
            Err(_) => None,
        };

        while let Some(dir) = cursor {
            candidates.push(dir.join("config").join("facepass.toml"));
            cursor = dir.parent().map(Path::to_path_buf);
        }

        candidates
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

        if self.recognition.similarity_threshold < 0.0
            || self.recognition.similarity_threshold > 1.0
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

        if self.anti_spoof.v2_input_size <= 0 || self.anti_spoof.v2_input_size > 512 {
            return Err(Error::Config(
                "anti_spoof.v2_input_size must be between 1 and 512".to_string(),
            ));
        }

        if self.anti_spoof.v1se_input_size <= 0 || self.anti_spoof.v1se_input_size > 512 {
            return Err(Error::Config(
                "anti_spoof.v1se_input_size must be between 1 and 512".to_string(),
            ));
        }

        if self.anti_spoof.v2_crop_scale <= 0.0 || self.anti_spoof.v2_crop_scale > 10.0 {
            return Err(Error::Config(
                "anti_spoof.v2_crop_scale must be between 0.0 and 10.0".to_string(),
            ));
        }

        if self.anti_spoof.v1se_crop_scale <= 0.0 || self.anti_spoof.v1se_crop_scale > 10.0 {
            return Err(Error::Config(
                "anti_spoof.v1se_crop_scale must be between 0.0 and 10.0".to_string(),
            ));
        }

        if self.recognition.valid_crop_scale <= 0.0 || self.recognition.valid_crop_scale > 10.0 {
            return Err(Error::Config(
                "recognition.valid_crop_scale must be between 0.0 and 10.0".to_string(),
            ));
        }

        if self.video.max_frames != 0
            && self.recognition.valid_frames != 0
            && self.video.max_frames < self.recognition.valid_frames
        {
            return Err(Error::Config(format!(
                "video.max_frames ({}) must be greater than or equal to recognition.valid_frames ({})",
                self.video.max_frames, self.recognition.valid_frames
            )));
        }

        if self.recognition.valid_frames != 0
            && self.recognition.consecutive_match_frames > self.recognition.valid_frames
        {
            return Err(Error::Config(format!(
                "recognition.consecutive_match_frames ({}) must be less than or equal to recognition.valid_frames ({})",
                self.recognition.consecutive_match_frames, self.recognition.valid_frames
            )));
        }

        if self.video.max_frames == 0
            && self.recognition.valid_frames == 0
            && self.video.timeout == 0
        {
            return Err(Error::Config(
                "Invalid configuration: video.max_frames, recognition.valid_frames, and video.timeout cannot all be 0. 必须添加有效终止条件，若要测试请使用--debug".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn valid_base_config() -> Config {
        let mut config = Config::default();
        config.models.yunet_path = "/bin/sh".to_string();
        config.models.sface_path = "/bin/sh".to_string();
        config
    }

    fn config_file_with_base(base: Config) -> ConfigFile {
        ConfigFile {
            active_preset: DEFAULT_PRESET_NAME.to_string(),
            base,
            presets: BTreeMap::new(),
        }
    }

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
        let mut config = valid_base_config();
        config.anti_spoof.enabled = true;
        config.models.anti_spoof_v2_path = missing_model_path("anti-spoof-v2");
        config.models.anti_spoof_v1se_path = missing_model_path("anti-spoof-v1se");

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_invalid_anti_spoof_threshold() {
        let mut config = valid_base_config();
        config.anti_spoof.threshold = 1.5;

        let err = config.validate().unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(err
            .to_string()
            .contains("anti_spoof.threshold must be between 0.0 and 1.0"));
    }

    #[test]
    fn test_validate_rejects_invalid_anti_spoof_crop_scale() {
        let mut config = valid_base_config();
        config.anti_spoof.v2_crop_scale = 0.0;

        let err = config.validate().unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(err
            .to_string()
            .contains("anti_spoof.v2_crop_scale must be between 0.0 and 10.0"));
    }

    #[test]
    fn test_validate_allows_unlimited_frame_and_valid_frame_settings() {
        let mut config = valid_base_config();
        config.video.max_frames = 0;
        config.recognition.valid_frames = 0;
        config.recognition.consecutive_match_frames = 10;

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_max_frames_lower_than_valid_frames_when_limited() {
        let mut config = valid_base_config();
        config.video.max_frames = 10;
        config.recognition.valid_frames = 20;

        let err = config.validate().unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(err.to_string().contains(
            "video.max_frames (10) must be greater than or equal to recognition.valid_frames (20)"
        ));
    }

    #[test]
    fn test_validate_rejects_missing_stop_condition() {
        let mut config = valid_base_config();
        config.video.max_frames = 0;
        config.video.timeout = 0;
        config.recognition.valid_frames = 0;

        let err = config.validate().unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(err.to_string().contains(
            "video.max_frames, recognition.valid_frames, and video.timeout cannot all be 0"
        ));
    }

    #[test]
    fn test_load_runs_validation() {
        let mut config = valid_base_config();
        config.video.max_frames = 0;
        config.video.timeout = 0;
        config.recognition.valid_frames = 0;

        let config_file = config_file_with_base(config);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), toml::to_string(&config_file).unwrap()).unwrap();

        let err = Config::load(tmp.path()).unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(err.to_string().contains("必须添加有效终止条件"));
    }

    #[test]
    fn test_load_with_source_reports_loaded_file() {
        let config_file = config_file_with_base(valid_base_config());
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), toml::to_string(&config_file).unwrap()).unwrap();

        let resolved = Config::load_with_source(tmp.path()).unwrap();
        assert_eq!(resolved.source.as_deref(), Some(tmp.path()));
        assert_eq!(resolved.active_preset, DEFAULT_PRESET_NAME);
    }

    #[test]
    fn test_config_file_resolves_base_only_default_preset() {
        let config_file = config_file_with_base(valid_base_config());
        let resolved = config_file.resolve_active_preset().unwrap();

        assert_eq!(resolved.video.timeout, 5);
        assert_eq!(resolved.detection.score_threshold, 0.9);
    }

    #[test]
    fn test_config_file_merges_named_preset() {
        let mut config_file = config_file_with_base(valid_base_config());
        config_file.active_preset = "strict".to_string();
        config_file.presets.insert(
            "strict".to_string(),
            Value::Table(Map::from_iter([
                (
                    "recognition".to_string(),
                    Value::Table(Map::from_iter([
                        ("similarity_threshold".to_string(), Value::Float(0.48)),
                        ("valid_frames".to_string(), Value::Integer(8)),
                    ])),
                ),
                (
                    "video".to_string(),
                    Value::Table(Map::from_iter([("timeout".to_string(), Value::Integer(9))])),
                ),
            ])),
        );

        let resolved = config_file.resolve_active_preset().unwrap();
        assert_eq!(resolved.video.timeout, 9);
        assert_eq!(resolved.recognition.similarity_threshold, 0.48);
        assert_eq!(resolved.recognition.valid_frames, 8);
        assert_eq!(resolved.video.frame_width, 640);
    }

    #[test]
    fn test_missing_named_preset_is_rejected() {
        let mut config_file = config_file_with_base(valid_base_config());
        config_file.active_preset = "missing".to_string();

        let err = config_file.resolve_active_preset().unwrap_err();
        assert!(err.to_string().contains("Unknown preset 'missing'"));
    }

    #[test]
    fn test_workspace_config_candidates_do_not_include_facepass_dev() {
        let candidates = Config::workspace_config_candidates();
        assert!(!candidates
            .iter()
            .any(|path| path.ends_with(Path::new("config").join("facepass-dev.toml"))));
    }
}
