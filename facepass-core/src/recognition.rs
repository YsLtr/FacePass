//! Face recognition module using SFace

use crate::config::RecognitionConfig;
use crate::error::{Error, Result};
use crate::models::FaceData;
use opencv::{
    core::{Mat, Ptr},
    objdetect::{FaceRecognizerSF, FaceRecognizerSF_DisType},
    prelude::*,
};
use std::sync::{Arc, Mutex};

/// Face recognizer wrapper using SFace model
pub struct FaceRecognizer {
    recognizer: Arc<Mutex<Ptr<FaceRecognizerSF>>>,
    #[allow(dead_code)]
    config: RecognitionConfig,
}

// Safety: We protect the recognizer with a Mutex
unsafe impl Send for FaceRecognizer {}
unsafe impl Sync for FaceRecognizer {}

impl FaceRecognizer {
    /// Create a new face recognizer
    pub fn new(model_path: &str, config: &RecognitionConfig) -> Result<Self> {
        let recognizer = FaceRecognizerSF::create(
            model_path, "", 0, // backend_id
            0, // target_id
        )
        .map_err(|e| Error::Recognition(format!("Failed to create SFace recognizer: {}", e)))?;

        Ok(Self {
            recognizer: Arc::new(Mutex::new(recognizer)),
            config: config.clone(),
        })
    }

    /// Align and crop a face from the image using detection result
    pub fn align_crop(&self, image: &Mat, face_mat: &Mat) -> Result<Mat> {
        let recognizer = self
            .recognizer
            .lock()
            .map_err(|e| Error::Recognition(format!("Failed to lock recognizer: {}", e)))?;

        let mut aligned = Mat::default();
        recognizer.align_crop(image, face_mat, &mut aligned)?;

        Ok(aligned)
    }

    /// Extract face feature from aligned face image
    pub fn extract_feature(&self, aligned_face: &Mat) -> Result<Mat> {
        let mut recognizer = self
            .recognizer
            .lock()
            .map_err(|e| Error::Recognition(format!("Failed to lock recognizer: {}", e)))?;

        let mut feature = Mat::default();
        recognizer.feature(aligned_face, &mut feature)?;

        Ok(feature)
    }

    /// Extract feature and convert to FaceData
    pub fn extract_face_data(&self, aligned_face: &Mat, label: &str) -> Result<FaceData> {
        let feature_mat = self.extract_feature(aligned_face)?;

        // Convert Mat to Vec<f32>
        let feature = mat_to_vec(&feature_mat)?;

        Ok(FaceData::new(label, feature))
    }

    /// Compare two face features using cosine similarity
    pub fn match_features(&self, feature1: &Mat, feature2: &Mat) -> Result<f64> {
        let recognizer = self
            .recognizer
            .lock()
            .map_err(|e| Error::Recognition(format!("Failed to lock recognizer: {}", e)))?;

        let score =
            recognizer.match_(feature1, feature2, FaceRecognizerSF_DisType::FR_COSINE as i32)?;

        Ok(score)
    }

    /// Compare feature Mat with FaceData
    pub fn match_with_face_data(&self, feature: &Mat, face_data: &FaceData) -> Result<f64> {
        let stored_feature = vec_to_mat(&face_data.feature)?;
        self.match_features(feature, &stored_feature)
    }
}

/// Convert OpenCV Mat to Vec<f32>
pub fn mat_to_vec(mat: &Mat) -> Result<Vec<f32>> {
    let total = mat.total();
    let mut vec = vec![0.0f32; total];
    let data = mat.data_typed::<f32>()?;
    vec.copy_from_slice(data);
    Ok(vec)
}

/// Convert Vec<f32> to OpenCV Mat
pub fn vec_to_mat(vec: &[f32]) -> Result<Mat> {
    let mat = Mat::from_slice(vec)?;
    // Reshape to 1 row, N columns
    let reshaped = mat.reshape(1, 1)?;
    Ok(reshaped.try_clone()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_mat_conversion() {
        let original: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();

        let mat = vec_to_mat(&original).unwrap();
        let converted = mat_to_vec(&mat).unwrap();

        assert_eq!(original.len(), converted.len());
        for (a, b) in original.iter().zip(converted.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
