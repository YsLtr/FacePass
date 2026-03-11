//! Face detection module using YuNet

use crate::config::DetectionConfig;
use crate::error::{Error, Result};
use crate::models::DetectionResult;
use opencv::{
    core::{Mat, Ptr, Size},
    objdetect::FaceDetectorYN,
    prelude::*,
};
use std::sync::{Arc, Mutex};

/// Face detector wrapper using YuNet model
pub struct FaceDetector {
    detector: Arc<Mutex<Ptr<FaceDetectorYN>>>,
}

// Safety: We protect the detector with a Mutex
unsafe impl Send for FaceDetector {}
unsafe impl Sync for FaceDetector {}

impl FaceDetector {
    /// Create a new face detector
    pub fn new(model_path: &str, config: &DetectionConfig) -> Result<Self> {
        let detector = FaceDetectorYN::create(
            model_path,
            "",
            Size::new(config.input_width, config.input_height),
            config.score_threshold,
            config.nms_threshold,
            5000, // top_k
            0,    // backend_id (default)
            0,    // target_id (default)
        )
        .map_err(|e| Error::Detection(format!("Failed to create YuNet detector: {}", e)))?;

        Ok(Self {
            detector: Arc::new(Mutex::new(detector)),
        })
    }

    /// Detect faces in an image
    pub fn detect(&self, image: &Mat) -> Result<Vec<DetectionResult>> {
        let mut detector = self
            .detector
            .lock()
            .map_err(|e| Error::Detection(format!("Failed to lock detector: {}", e)))?;

        // Update input size to match image
        let size = image.size()?;
        detector.set_input_size(size)?;

        // Detect faces
        let mut faces = Mat::default();
        detector.detect(image, &mut faces)?;

        // Parse results
        let mut results = Vec::new();
        let rows = faces.rows();

        for i in 0..rows {
            // YuNet output format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            let x = *faces.at_2d::<f32>(i, 0)?;
            let y = *faces.at_2d::<f32>(i, 1)?;
            let w = *faces.at_2d::<f32>(i, 2)?;
            let h = *faces.at_2d::<f32>(i, 3)?;

            // Landmarks
            let right_eye = (*faces.at_2d::<f32>(i, 4)?, *faces.at_2d::<f32>(i, 5)?);
            let left_eye = (*faces.at_2d::<f32>(i, 6)?, *faces.at_2d::<f32>(i, 7)?);
            let nose_tip = (*faces.at_2d::<f32>(i, 8)?, *faces.at_2d::<f32>(i, 9)?);
            let right_mouth = (*faces.at_2d::<f32>(i, 10)?, *faces.at_2d::<f32>(i, 11)?);
            let left_mouth = (*faces.at_2d::<f32>(i, 12)?, *faces.at_2d::<f32>(i, 13)?);

            let confidence = *faces.at_2d::<f32>(i, 14)?;

            results.push(DetectionResult {
                bbox: (x, y, w, h),
                confidence,
                landmarks: [right_eye, left_eye, nose_tip, right_mouth, left_mouth],
            });
        }

        Ok(results)
    }

    /// Detect a single face (returns the most confident one)
    pub fn detect_single(&self, image: &Mat) -> Result<DetectionResult> {
        let faces = self.detect(image)?;

        faces
            .into_iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .ok_or(Error::NoFaceDetected)
    }

    /// Get the raw faces Mat for use with FaceRecognizer
    pub fn detect_raw(&self, image: &Mat) -> Result<Mat> {
        let mut detector = self
            .detector
            .lock()
            .map_err(|e| Error::Detection(format!("Failed to lock detector: {}", e)))?;

        // Update input size
        let size = image.size()?;
        detector.set_input_size(size)?;

        // Detect
        let mut faces = Mat::default();
        detector.detect(image, &mut faces)?;

        if faces.rows() == 0 {
            return Err(Error::NoFaceDetected);
        }

        Ok(faces)
    }

    /// Update score threshold
    pub fn set_score_threshold(&self, threshold: f32) -> Result<()> {
        let mut detector = self
            .detector
            .lock()
            .map_err(|e| Error::Detection(format!("Failed to lock detector: {}", e)))?;
        detector.set_score_threshold(threshold)?;
        Ok(())
    }

    /// Update NMS threshold
    pub fn set_nms_threshold(&self, threshold: f32) -> Result<()> {
        let mut detector = self
            .detector
            .lock()
            .map_err(|e| Error::Detection(format!("Failed to lock detector: {}", e)))?;
        detector.set_nms_threshold(threshold)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require the YuNet model file to be present
    // They are ignored by default

    #[test]
    #[ignore]
    fn test_detector_creation() {
        let config = DetectionConfig::default();
        let detector = FaceDetector::new(
            "/usr/share/facepass/models/face_detection_yunet_2023mar.onnx",
            &config,
        );
        assert!(detector.is_ok());
    }
}
