//! Configuration management for FacePass

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use toml::{map::Map, Value};

pub const DEFAULT_CONFIG_PATH: &str = "/etc/facepass/config.toml";
pub const USER_CONFIG_PATH: &str = ".config/facepass/config.toml";
pub const DEFAULT_RUNTIME_STATE_PATH: &str = "/run/facepass/runtime-state.json";
pub const DEFAULT_PRESET_NAME: &str = "default";
pub const DEFAULT_DATA_DIR: &str = "/var/lib/facepass/faces";
pub const DEFAULT_MODELS_DIR: &str = "/usr/share/facepass/models";
pub const DEFAULT_SOCKET_PATH: &str = "/run/facepass/facepass.sock";

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

    pub fn load_with_source<P: AsRef<Path>>(path: P) -> Result<(Self, Option<PathBuf>)> {
        let path = path.as_ref();
        Ok((Self::load(path)?, Some(path.to_path_buf())))
    }

    pub fn load_with_fallback_and_source<P: AsRef<Path>>(
        preferred_path: P,
    ) -> Result<(Self, Option<PathBuf>)> {
        let preferred = preferred_path.as_ref();
        if preferred.exists() {
            return Self::load_with_source(preferred);
        }

        Self::load_or_default_with_source()
    }

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

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("Failed to serialize config: {}", e)))?;

        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, content)?;
        Ok(())
    }

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

    pub fn resolve_active_preset(&self) -> Result<Config> {
        self.resolve_preset(self.active_preset_name()?.as_str())
    }

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

    pub fn active_preset_name(&self) -> Result<String> {
        Ok(normalize_preset_name(&self.active_preset)?.to_string())
    }

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

#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub config: Config,
    pub source: Option<PathBuf>,
    pub active_preset: String,
}

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
            ("similarity_threshold".to_string(), Value::Float(0.4)),
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
                "active_detector".to_string(),
                Value::String("scrfd".to_string()),
            ),
            (
                "active_recognizer".to_string(),
                Value::String("ghostfacenet".to_string()),
            ),
            (
                "yunet".to_string(),
                Value::Table(Map::from_iter([
                    (
                        "path".to_string(),
                        Value::String(
                            "/home/ysltr/builds/FacePass/facepass/models/face_detection_yunet_2023mar.onnx"
                                .to_string(),
                        ),
                    ),
                    ("input_width".to_string(), Value::Integer(320)),
                    ("input_height".to_string(), Value::Integer(320)),
                    ("score_threshold".to_string(), Value::Float(0.9)),
                    ("nms_threshold".to_string(), Value::Float(0.3)),
                ])),
            ),
            (
                "scrfd".to_string(),
                Value::Table(Map::from_iter([
                    (
                        "path".to_string(),
                        Value::String(
                            "/home/ysltr/builds/FacePass/facepass/models/SCRFD_2.5g_bnkps.onnx"
                                .to_string(),
                        ),
                    ),
                    ("input_width".to_string(), Value::Integer(640)),
                    ("input_height".to_string(), Value::Integer(640)),
                    ("score_threshold".to_string(), Value::Float(0.5)),
                    ("nms_threshold".to_string(), Value::Float(0.4)),
                ])),
            ),
            (
                "sface".to_string(),
                Value::Table(Map::from_iter([
                    (
                        "path".to_string(),
                        Value::String(
                            "/home/ysltr/builds/FacePass/facepass/models/face_recognition_sface_2021dec.onnx"
                                .to_string(),
                        ),
                    ),
                    (
                        "model_id".to_string(),
                        Value::String("sface-128".to_string()),
                    ),
                    ("embedding_dim".to_string(), Value::Integer(128)),
                    ("preprocess".to_string(), sface_preprocess_value()),
                ])),
            ),
            (
                "mobilefacenet".to_string(),
                Value::Table(Map::from_iter([
                    (
                        "path".to_string(),
                        Value::String(
                            "/home/ysltr/builds/FacePass/facepass/models/MobileFaceNet.onnx"
                                .to_string(),
                        ),
                    ),
                    (
                        "model_id".to_string(),
                        Value::String("mobilefacenet-128".to_string()),
                    ),
                    ("embedding_dim".to_string(), Value::Integer(128)),
                    ("preprocess".to_string(), mobilefacenet_preprocess_value()),
                ])),
            ),
            (
                "ghostfacenet".to_string(),
                Value::Table(Map::from_iter([
                    (
                        "path".to_string(),
                        Value::String(
                            "/home/ysltr/builds/FacePass/facepass/models/GhostFaceNet_W1.3_S1_ArcFace-ir11.onnx"
                                .to_string(),
                        ),
                    ),
                    (
                        "model_id".to_string(),
                        Value::String("ghostfacenet-w1.3-s1-arcface-ir11-512".to_string()),
                    ),
                    ("embedding_dim".to_string(), Value::Integer(512)),
                    ("preprocess".to_string(), ghostfacenet_preprocess_value()),
                ])),
            ),
            (
                "anti_spoof_v2_path".to_string(),
                Value::String(
                    "/home/ysltr/builds/FacePass/facepass/models/MiniFASNetV2.onnx".to_string(),
                ),
            ),
            (
                "anti_spoof_v1se_path".to_string(),
                Value::String(
                    "/home/ysltr/builds/FacePass/facepass/models/MiniFASNetV1SE.onnx".to_string(),
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

fn preprocess_value(
    input_width: i64,
    input_height: i64,
    input_layout: &str,
    color_order: &str,
    mean: [f64; 3],
    std: [f64; 3],
    l2_normalize: bool,
) -> Value {
    Value::Table(Map::from_iter([
        ("input_width".to_string(), Value::Integer(input_width)),
        ("input_height".to_string(), Value::Integer(input_height)),
        (
            "input_layout".to_string(),
            Value::String(input_layout.to_string()),
        ),
        (
            "color_order".to_string(),
            Value::String(color_order.to_string()),
        ),
        (
            "mean".to_string(),
            Value::Array(mean.into_iter().map(Value::Float).collect()),
        ),
        (
            "std".to_string(),
            Value::Array(std.into_iter().map(Value::Float).collect()),
        ),
        (
            "l2_normalize".to_string(),
            Value::Boolean(l2_normalize),
        ),
    ]))
}

fn sface_preprocess_value() -> Value {
    preprocess_value(112, 112, "nchw", "bgr", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], true)
}

fn mobilefacenet_preprocess_value() -> Value {
    preprocess_value(
        112,
        112,
        "nchw",
        "rgb",
        [127.5, 127.5, 127.5],
        [127.5, 127.5, 127.5],
        true,
    )
}

fn ghostfacenet_preprocess_value() -> Value {
    preprocess_value(
        112,
        112,
        "nhwc",
        "rgb",
        [127.5, 127.5, 127.5],
        [128.0, 128.0, 128.0],
        true,
    )
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DetectorKind {
    #[serde(rename = "yunet")]
    Yunet,
    #[serde(rename = "scrfd")]
    Scrfd,
}

impl DetectorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Yunet => "yunet",
            Self::Scrfd => "scrfd",
        }
    }
}

impl Default for DetectorKind {
    fn default() -> Self {
        Self::Scrfd
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecognizerKind {
    #[serde(rename = "sface")]
    Sface,
    #[serde(rename = "mobilefacenet")]
    Mobilefacenet,
    #[serde(rename = "ghostfacenet")]
    Ghostfacenet,
}

impl RecognizerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sface => "sface",
            Self::Mobilefacenet => "mobilefacenet",
            Self::Ghostfacenet => "ghostfacenet",
        }
    }
}

impl Default for RecognizerKind {
    fn default() -> Self {
        Self::Ghostfacenet
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InputLayout {
    #[serde(rename = "nchw")]
    Nchw,
    #[serde(rename = "nhwc")]
    Nhwc,
}

impl InputLayout {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nchw => "nchw",
            Self::Nhwc => "nhwc",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ColorOrder {
    #[serde(rename = "bgr")]
    Bgr,
    #[serde(rename = "rgb")]
    Rgb,
}

impl ColorOrder {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bgr => "bgr",
            Self::Rgb => "rgb",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoConfig {
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default = "default_timeout")]
    pub timeout: u32,
    #[serde(default = "default_max_frames")]
    pub max_frames: u32,
    #[serde(default = "default_frame_width")]
    pub frame_width: u32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionConfig {
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f64,
    #[serde(default = "default_max_faces_per_group")]
    pub max_faces_per_group: u32,
    #[serde(default = "default_consecutive_match_frames")]
    pub consecutive_match_frames: u32,
    #[serde(default = "default_valid_frames")]
    pub valid_frames: u32,
    #[serde(default = "default_stop_on_valid_frames")]
    pub stop_on_valid_frames: bool,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    #[serde(default = "default_ignore_ssh")]
    pub ignore_ssh: bool,
    #[serde(default = "default_ignore_closed_lid")]
    pub ignore_closed_lid: bool,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    #[serde(default = "default_socket_path")]
    pub socket_path: String,
    #[serde(default = "default_log_level")]
    pub log_level: String,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorModelConfig {
    #[serde(default)]
    pub path: String,
    #[serde(default = "default_detector_input_width")]
    pub input_width: i32,
    #[serde(default = "default_detector_input_height")]
    pub input_height: i32,
    #[serde(default = "default_detector_score_threshold")]
    pub score_threshold: f32,
    #[serde(default = "default_detector_nms_threshold")]
    pub nms_threshold: f32,
}

impl Default for DetectorModelConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            input_width: default_detector_input_width(),
            input_height: default_detector_input_height(),
            score_threshold: default_detector_score_threshold(),
            nms_threshold: default_detector_nms_threshold(),
        }
    }
}

fn default_detector_input_width() -> i32 {
    640
}
fn default_detector_input_height() -> i32 {
    640
}
fn default_detector_score_threshold() -> f32 {
    0.5
}
fn default_detector_nms_threshold() -> f32 {
    0.4
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizerPreprocessConfig {
    #[serde(default = "default_recognizer_input_width")]
    pub input_width: i32,
    #[serde(default = "default_recognizer_input_height")]
    pub input_height: i32,
    #[serde(default = "default_input_layout")]
    pub input_layout: InputLayout,
    #[serde(default = "default_color_order")]
    pub color_order: ColorOrder,
    #[serde(default = "default_channel_mean")]
    pub mean: [f32; 3],
    #[serde(default = "default_channel_std")]
    pub std: [f32; 3],
    #[serde(default = "default_l2_normalize")]
    pub l2_normalize: bool,
}

impl Default for RecognizerPreprocessConfig {
    fn default() -> Self {
        Self {
            input_width: default_recognizer_input_width(),
            input_height: default_recognizer_input_height(),
            input_layout: default_input_layout(),
            color_order: default_color_order(),
            mean: default_channel_mean(),
            std: default_channel_std(),
            l2_normalize: default_l2_normalize(),
        }
    }
}

fn default_recognizer_input_width() -> i32 {
    112
}
fn default_recognizer_input_height() -> i32 {
    112
}
fn default_input_layout() -> InputLayout {
    InputLayout::Nhwc
}
fn default_color_order() -> ColorOrder {
    ColorOrder::Rgb
}
fn default_channel_mean() -> [f32; 3] {
    [127.5, 127.5, 127.5]
}
fn default_channel_std() -> [f32; 3] {
    [128.0, 128.0, 128.0]
}
fn default_l2_normalize() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizerModelConfig {
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub model_id: String,
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,
    #[serde(default)]
    pub preprocess: RecognizerPreprocessConfig,
}

impl Default for RecognizerModelConfig {
    fn default() -> Self {
        Self {
            path: String::new(),
            model_id: String::new(),
            embedding_dim: default_embedding_dim(),
            preprocess: RecognizerPreprocessConfig::default(),
        }
    }
}

fn default_embedding_dim() -> usize {
    512
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    #[serde(default)]
    pub active_detector: DetectorKind,
    #[serde(default)]
    pub active_recognizer: RecognizerKind,
    #[serde(default = "default_yunet_detector")]
    pub yunet: DetectorModelConfig,
    #[serde(default = "default_scrfd_detector")]
    pub scrfd: DetectorModelConfig,
    #[serde(default = "default_sface_recognizer")]
    pub sface: RecognizerModelConfig,
    #[serde(default = "default_mobilefacenet_recognizer")]
    pub mobilefacenet: RecognizerModelConfig,
    #[serde(default = "default_ghostfacenet_recognizer")]
    pub ghostfacenet: RecognizerModelConfig,
    #[serde(default = "default_anti_spoof_v2_path")]
    pub anti_spoof_v2_path: String,
    #[serde(default = "default_anti_spoof_v1se_path")]
    pub anti_spoof_v1se_path: String,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            active_detector: DetectorKind::default(),
            active_recognizer: RecognizerKind::default(),
            yunet: default_yunet_detector(),
            scrfd: default_scrfd_detector(),
            sface: default_sface_recognizer(),
            mobilefacenet: default_mobilefacenet_recognizer(),
            ghostfacenet: default_ghostfacenet_recognizer(),
            anti_spoof_v2_path: default_anti_spoof_v2_path(),
            anti_spoof_v1se_path: default_anti_spoof_v1se_path(),
        }
    }
}

impl ModelsConfig {
    pub fn active_detector_config(&self) -> &DetectorModelConfig {
        match self.active_detector {
            DetectorKind::Yunet => &self.yunet,
            DetectorKind::Scrfd => &self.scrfd,
        }
    }

    pub fn active_recognizer_config(&self) -> &RecognizerModelConfig {
        match self.active_recognizer {
            RecognizerKind::Sface => &self.sface,
            RecognizerKind::Mobilefacenet => &self.mobilefacenet,
            RecognizerKind::Ghostfacenet => &self.ghostfacenet,
        }
    }
}

fn default_yunet_detector() -> DetectorModelConfig {
    DetectorModelConfig {
        path: format!("{}/face_detection_yunet_2023mar.onnx", DEFAULT_MODELS_DIR),
        input_width: 320,
        input_height: 320,
        score_threshold: 0.9,
        nms_threshold: 0.3,
    }
}

fn default_scrfd_detector() -> DetectorModelConfig {
    DetectorModelConfig {
        path: format!("{}/SCRFD_2.5g_bnkps.onnx", DEFAULT_MODELS_DIR),
        input_width: 640,
        input_height: 640,
        score_threshold: 0.5,
        nms_threshold: 0.4,
    }
}

fn default_sface_recognizer() -> RecognizerModelConfig {
    RecognizerModelConfig {
        path: format!("{}/face_recognition_sface_2021dec.onnx", DEFAULT_MODELS_DIR),
        model_id: "sface-128".to_string(),
        embedding_dim: 128,
        preprocess: RecognizerPreprocessConfig {
            input_width: 112,
            input_height: 112,
            input_layout: InputLayout::Nchw,
            color_order: ColorOrder::Bgr,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            l2_normalize: true,
        },
    }
}

fn default_mobilefacenet_recognizer() -> RecognizerModelConfig {
    RecognizerModelConfig {
        path: format!("{}/MobileFaceNet.onnx", DEFAULT_MODELS_DIR),
        model_id: "mobilefacenet-128".to_string(),
        embedding_dim: 128,
        preprocess: RecognizerPreprocessConfig {
            input_width: 112,
            input_height: 112,
            input_layout: InputLayout::Nchw,
            color_order: ColorOrder::Rgb,
            mean: [127.5, 127.5, 127.5],
            std: [127.5, 127.5, 127.5],
            l2_normalize: true,
        },
    }
}

fn default_ghostfacenet_recognizer() -> RecognizerModelConfig {
    RecognizerModelConfig {
        path: format!(
            "{}/GhostFaceNet_W1.3_S1_ArcFace-ir11.onnx",
            DEFAULT_MODELS_DIR
        ),
        model_id: "ghostfacenet-w1.3-s1-arcface-ir11-512".to_string(),
        embedding_dim: 512,
        preprocess: RecognizerPreprocessConfig {
            input_width: 112,
            input_height: 112,
            input_layout: InputLayout::Nhwc,
            color_order: ColorOrder::Rgb,
            mean: [127.5, 127.5, 127.5],
            std: [128.0, 128.0, 128.0],
            l2_normalize: true,
        },
    }
}

fn default_anti_spoof_v2_path() -> String {
    format!("{}/MiniFASNetV2.onnx", DEFAULT_MODELS_DIR)
}
fn default_anti_spoof_v1se_path() -> String {
    format!("{}/MiniFASNetV1SE.onnx", DEFAULT_MODELS_DIR)
}

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
    #[serde(default = "default_anti_spoof_enabled")]
    pub enabled: bool,
    #[serde(default = "default_liveness_threshold")]
    pub threshold: f32,
    #[serde(default = "default_anti_spoof_mode")]
    pub mode: AntiSpoofMode,
    #[serde(default = "default_v2_input_size")]
    pub v2_input_size: i32,
    #[serde(default = "default_v2_crop_scale")]
    pub v2_crop_scale: f32,
    #[serde(default = "default_v1se_input_size")]
    pub v1se_input_size: i32,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub video: VideoConfig,
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
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self::load_with_source(path)?.config)
    }

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

    pub fn load_with_fallback<P: AsRef<Path>>(preferred_path: P) -> Result<Self> {
        Ok(Self::load_with_fallback_and_source(preferred_path)?.config)
    }

    pub fn load_with_fallback_and_source<P: AsRef<Path>>(
        preferred_path: P,
    ) -> Result<ResolvedConfig> {
        let preferred = preferred_path.as_ref();
        if preferred.exists() {
            return Self::load_with_source(preferred);
        }

        Self::load_or_default_with_source()
    }

    pub fn load_or_default() -> Result<Self> {
        Ok(Self::load_or_default_with_source()?.config)
    }

    pub fn load_or_default_with_source() -> Result<ResolvedConfig> {
        for candidate in Self::workspace_config_candidates() {
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

        let config = Self::default();
        config.validate()?;
        Ok(ResolvedConfig {
            config,
            source: None,
            active_preset: DEFAULT_PRESET_NAME.to_string(),
        })
    }

    pub fn user_config_path() -> Option<PathBuf> {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(USER_CONFIG_PATH))
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        ConfigFile {
            active_preset: DEFAULT_PRESET_NAME.to_string(),
            base: self.clone(),
            presets: BTreeMap::new(),
        }
        .save(path)
    }

    pub fn user_data_dir(&self, username: &str) -> PathBuf {
        PathBuf::from(&self.storage.data_dir).join(username)
    }

    fn workspace_config_candidates() -> Vec<PathBuf> {
        let mut candidates = Vec::new();
        let mut cursor = std::env::current_dir().ok();

        while let Some(dir) = cursor {
            candidates.push(dir.join("config").join("facepass.toml"));
            cursor = dir.parent().map(Path::to_path_buf);
        }

        candidates
    }

    pub fn validate(&self) -> Result<()> {
        let detector = self.models.active_detector_config();
        if detector.path.trim().is_empty() || !Path::new(&detector.path).exists() {
            return Err(Error::Config(format!(
                "Active detector model not found: {}",
                detector.path
            )));
        }
        if detector.input_width <= 0 || detector.input_height <= 0 {
            return Err(Error::Config(
                "Detector input size must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&detector.score_threshold) {
            return Err(Error::Config(
                "Detector score_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&detector.nms_threshold) {
            return Err(Error::Config(
                "Detector nms_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        let recognizer = self.models.active_recognizer_config();
        if recognizer.path.trim().is_empty() || !Path::new(&recognizer.path).exists() {
            return Err(Error::Config(format!(
                "Active recognizer model not found: {}",
                recognizer.path
            )));
        }
        if recognizer.model_id.trim().is_empty() {
            return Err(Error::Config(
                "Active recognizer model_id cannot be empty".to_string(),
            ));
        }
        if recognizer.embedding_dim == 0 {
            return Err(Error::Config(
                "Active recognizer embedding_dim must be greater than 0".to_string(),
            ));
        }
        if recognizer.preprocess.input_width <= 0 || recognizer.preprocess.input_height <= 0 {
            return Err(Error::Config(
                "Recognizer input size must be positive".to_string(),
            ));
        }
        if recognizer.preprocess.std.iter().any(|value| *value <= 0.0) {
            return Err(Error::Config(
                "Recognizer preprocess std values must be greater than 0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.recognition.similarity_threshold) {
            return Err(Error::Config(
                "recognition.similarity_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.anti_spoof.threshold) {
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

    fn existing_path() -> String {
        "/bin/sh".to_string()
    }

    fn valid_base_config() -> Config {
        let mut config = Config::default();
        config.models.scrfd.path = existing_path();
        config.models.ghostfacenet.path = existing_path();
        config.models.yunet.path = existing_path();
        config.models.sface.path = existing_path();
        config.models.mobilefacenet.path = existing_path();
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
    fn test_default_config_uses_scrfd_and_ghostfacenet() {
        let config = Config::default();
        assert_eq!(config.models.active_detector, DetectorKind::Scrfd);
        assert_eq!(config.models.active_recognizer, RecognizerKind::Ghostfacenet);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.models.active_detector, config.models.active_detector);
        assert_eq!(
            parsed.models.ghostfacenet.preprocess.input_layout,
            InputLayout::Nhwc
        );
    }

    #[test]
    fn test_validate_allows_missing_anti_spoof_models() {
        let mut config = valid_base_config();
        config.anti_spoof.enabled = true;
        config.models.anti_spoof_v2_path = missing_model_path("anti-spoof-v2");
        config.models.anti_spoof_v1se_path = missing_model_path("anti-spoof-v1se");

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_missing_active_model() {
        let mut config = valid_base_config();
        config.models.active_recognizer = RecognizerKind::Ghostfacenet;
        config.models.ghostfacenet.path = missing_model_path("ghost");
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("Active recognizer model not found"));
    }

    #[test]
    fn test_validate_rejects_invalid_preprocess_std() {
        let mut config = valid_base_config();
        config.models.ghostfacenet.preprocess.std = [0.0, 1.0, 1.0];
        let err = config.validate().unwrap_err();
        assert!(err
            .to_string()
            .contains("Recognizer preprocess std values must be greater than 0"));
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
}
