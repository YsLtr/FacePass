//! Shared face runtime pipeline used by CLI and daemon flows.

use crate::{
    anti_spoofing::AntiSpoofDetector,
    config::Config,
    detection::FaceDetector,
    error::Result,
    face_validation::select_primary_face,
    models::{DetectionResult, FaceEmbedding},
    recognition::FaceRecognizer,
};
use opencv::core::Mat;

pub struct FaceRuntime {
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    anti_spoof: Option<AntiSpoofDetector>,
    valid_crop_scale: f32,
}

impl FaceRuntime {
    pub fn new(config: &Config) -> Result<Self> {
        let detector = FaceDetector::new(&config.models)?;
        let recognizer = FaceRecognizer::new(&config.models)?;
        let anti_spoof = if config.anti_spoof.enabled {
            Some(AntiSpoofDetector::new(&config.models, &config.anti_spoof)?)
        } else {
            None
        };

        Ok(Self {
            detector,
            recognizer,
            anti_spoof,
            valid_crop_scale: config.recognition.valid_crop_scale,
        })
    }

    pub fn detector(&self) -> &FaceDetector {
        &self.detector
    }

    pub fn recognizer(&self) -> &FaceRecognizer {
        &self.recognizer
    }

    pub fn anti_spoof(&self) -> Option<&AntiSpoofDetector> {
        self.anti_spoof.as_ref()
    }

    pub fn valid_crop_scale(&self) -> f32 {
        self.valid_crop_scale
    }

    pub fn detect_faces(&self, frame: &Mat) -> Result<Vec<DetectionResult>> {
        self.detector.detect(frame)
    }

    pub fn select_primary_face(
        &self,
        frame: &Mat,
        detections: &[DetectionResult],
    ) -> Result<DetectionResult> {
        select_primary_face(frame, detections, self.valid_crop_scale)
    }

    pub fn extract_embedding(
        &self,
        frame: &Mat,
        detection: &DetectionResult,
    ) -> Result<FaceEmbedding> {
        self.recognizer.extract_embedding_from_frame(frame, detection)
    }

    pub fn check_liveness(&self, frame: &Mat, detection: &DetectionResult) -> Result<Option<f32>> {
        match self.anti_spoof.as_ref() {
            Some(detector) => detector
                .check_liveness(frame, detection, self.valid_crop_scale)
                .map(Some),
            None => Ok(None),
        }
    }

    pub fn model_id(&self) -> &str {
        self.recognizer.model_id()
    }

    pub fn embedding_dim(&self) -> usize {
        self.recognizer.embedding_dim()
    }
}
