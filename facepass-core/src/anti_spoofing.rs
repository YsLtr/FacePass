//! Anti-spoofing module using MiniFASNetV2SE

use crate::config::AntiSpoofConfig;
use crate::error::{Error, Result};
use opencv::{
    core::{self, Mat, Scalar, Size, Vector, CV_32F, CV_8UC3},
    dnn,
    imgproc,
    prelude::*,
};
use std::sync::{Arc, Mutex};

/// Anti-spoofing detector using MiniFASNetV2SE ONNX model
pub struct AntiSpoofDetector {
    net: Arc<Mutex<dnn::Net>>,
    config: AntiSpoofConfig,
}

// Safety: We protect the net with a Mutex
unsafe impl Send for AntiSpoofDetector {}
unsafe impl Sync for AntiSpoofDetector {}

impl AntiSpoofDetector {
    /// Create a new anti-spoofing detector
    pub fn new(model_path: &str, config: &AntiSpoofConfig) -> Result<Self> {
        let mut net = dnn::read_net_from_onnx(model_path)
            .map_err(|e| Error::AntiSpoofing(format!("Failed to load anti-spoofing model: {}", e)))?;

        let mut cfg = config.clone();
        cfg.input_size = validate_or_probe_input_size(&mut net, cfg.input_size)?;

        Ok(Self {
            net: Arc::new(Mutex::new(net)),
            config: cfg,
        })
    }

    /// Check if a face is live (not spoofed)
    ///
    /// Takes an aligned/cropped face image and returns a liveness score (0.0-1.0).
    /// Higher score = more likely to be a real face.
    pub fn check_liveness(&self, face_crop: &Mat) -> Result<f32> {
        let mut net = self
            .net
            .lock()
            .map_err(|e| Error::AntiSpoofing(format!("Failed to lock anti-spoof net: {}", e)))?;

        forward_liveness(&mut net, face_crop, self.config.input_size)
    }
}

fn validate_or_probe_input_size(net: &mut dnn::Net, configured_size: i32) -> Result<i32> {
    let mut sizes = Vec::new();
    if configured_size > 0 {
        sizes.push(configured_size);
    }
    for size in [40, 80, 112, 128, 64] {
        if !sizes.contains(&size) {
            sizes.push(size);
        }
    }

    let mut last_err: Option<String> = None;
    let prev_level = core::get_log_level().ok();
    let _ = core::set_log_level(core::LogLevel::LOG_LEVEL_SILENT);

    for size in &sizes {
        let dummy = Mat::zeros(*size, *size, CV_8UC3)?.to_mat()?;
        match forward_liveness(net, &dummy, *size) {
            Ok(_) => {
                if let Some(level) = prev_level {
                    let _ = core::set_log_level(level);
                }
                return Ok(*size);
            }
            Err(Error::OpenCV(e)) => last_err = Some(e),
            Err(_) => {}
        }
    }

    if let Some(level) = prev_level {
        let _ = core::set_log_level(level);
    }

    Err(Error::AntiSpoofing(format!(
        "Failed to infer anti-spoof input size; tried {:?}. Last error: {}",
        sizes,
        last_err
            .map(|e| e)
            .unwrap_or_else(|| "unknown error".to_string())
    )))
}

fn forward_liveness(net: &mut dnn::Net, face_crop: &Mat, input_size: i32) -> Result<f32> {
    // Resize face crop to model input size
    let input_size = Size::new(input_size, input_size);
    let mut resized = Mat::default();
    imgproc::resize(face_crop, &mut resized, input_size, 0.0, 0.0, imgproc::INTER_LINEAR)?;

    // Create blob from image
    // MiniFASNetV2SE expects BGR input normalized to [0, 1]
    let blob = dnn::blob_from_image(
        &resized,
        1.0 / 255.0,          // scale factor
        input_size,
        Scalar::new(0.0, 0.0, 0.0, 0.0), // mean subtraction
        false,                 // swap RB (keep BGR)
        false,                 // crop
        CV_32F,
    )?;

    // Set input and run forward pass
    net.set_input(&blob, "", 1.0, Scalar::default())?;

    let out_layers = net.get_unconnected_out_layers_names()?;

    let mut outputs = Vector::<Mat>::new();
    net.forward(&mut outputs, &out_layers)?;

    if outputs.is_empty() {
        return Err(Error::AntiSpoofing("Empty model output".to_string()));
    }

    let output = outputs.get(0)?;
    let num_classes = output.total();

    if num_classes < 2 {
        return Err(Error::AntiSpoofing(format!(
            "Unexpected output size: {}",
            num_classes
        )));
    }

    // Output: [spoof_score, real_score] — apply softmax to get probabilities
    let data = output.data_typed::<f32>()?;
    let real_score = softmax_real(data[0], data[1]);

    Ok(real_score)
}

/// Compute softmax and return the probability of the "real" class (index 1)
fn softmax_real(spoof: f32, real: f32) -> f32 {
    let max_val = spoof.max(real);
    let exp_spoof = (spoof - max_val).exp();
    let exp_real = (real - max_val).exp();
    exp_real / (exp_spoof + exp_real)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        // When real >> spoof, should be close to 1.0
        let score = softmax_real(-5.0, 5.0);
        assert!(score > 0.99);

        // When spoof >> real, should be close to 0.0
        let score = softmax_real(5.0, -5.0);
        assert!(score < 0.01);

        // Equal scores should give 0.5
        let score = softmax_real(1.0, 1.0);
        assert!((score - 0.5).abs() < 1e-6);
    }
}
