//! Anti-spoofing module using the MiniFASNet ONNX models from
//! the companion `face-anti-spoofing` project.

use crate::config::AntiSpoofConfig;
use crate::error::{Error, Result};
use opencv::{
    core::{self, Mat, Rect, Scalar, Size, Vector, CV_32F, CV_8UC3},
    dnn, imgproc,
    prelude::*,
};
use std::sync::{Arc, Mutex};

/// Anti-spoofing detector using MiniFASNet ONNX models.
pub struct AntiSpoofDetector {
    net: Arc<Mutex<dnn::Net>>,
    config: AntiSpoofConfig,
}

// Safety: We protect the net with a Mutex
unsafe impl Send for AntiSpoofDetector {}
unsafe impl Sync for AntiSpoofDetector {}

impl AntiSpoofDetector {
    /// Create a new anti-spoofing detector.
    pub fn new(model_path: &str, config: &AntiSpoofConfig) -> Result<Self> {
        let mut net = dnn::read_net_from_onnx(model_path).map_err(|e| {
            Error::AntiSpoofing(format!("Failed to load anti-spoofing model: {}", e))
        })?;

        let mut cfg = config.clone();
        cfg.input_size = validate_or_probe_input_size(&mut net, cfg.input_size, cfg.crop_scale)?;

        Ok(Self {
            net: Arc::new(Mutex::new(net)),
            config: cfg,
        })
    }

    /// Check if a detected face is live (not spoofed).
    ///
    /// This mirrors the reference project: expand the detector bbox on the
    /// original frame, resize to the model input shape, and run MiniFASNet.
    /// Returns the probability of the "real" class.
    pub fn check_liveness(&self, frame: &Mat, face_row: &Mat) -> Result<f32> {
        let bbox = face_row_to_bbox(face_row)?;
        let mut net = self
            .net
            .lock()
            .map_err(|e| Error::AntiSpoofing(format!("Failed to lock anti-spoof net: {}", e)))?;

        forward_liveness(
            &mut net,
            frame,
            bbox,
            self.config.input_size,
            self.config.crop_scale,
        )
    }
}

fn validate_or_probe_input_size(
    net: &mut dnn::Net,
    configured_size: i32,
    crop_scale: f32,
) -> Result<i32> {
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
        let dummy = Mat::zeros(256, 256, CV_8UC3)?.to_mat()?;
        let bbox = [64.0, 64.0, 128.0, 128.0];
        match forward_liveness(net, &dummy, bbox, *size, crop_scale) {
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

fn face_row_to_bbox(face_row: &Mat) -> Result<[f32; 4]> {
    if face_row.total() < 4 {
        return Err(Error::AntiSpoofing(
            "Face detector output does not contain a valid bbox".to_string(),
        ));
    }

    Ok([
        *face_row.at_2d::<f32>(0, 0)?,
        *face_row.at_2d::<f32>(0, 1)?,
        *face_row.at_2d::<f32>(0, 2)?,
        *face_row.at_2d::<f32>(0, 3)?,
    ])
}

fn crop_face(image: &Mat, bbox: [f32; 4], scale_limit: f32, out_size: i32) -> Result<Mat> {
    let size = image.size()?;
    let src_w = size.width as f32;
    let src_h = size.height as f32;
    let [x, y, box_w, box_h] = bbox;

    if box_w <= 0.0 || box_h <= 0.0 {
        return Err(Error::AntiSpoofing(format!(
            "Invalid anti-spoof bbox: [{x:.1}, {y:.1}, {box_w:.1}, {box_h:.1}]"
        )));
    }

    let scale = ((src_h - 1.0) / box_h)
        .min((src_w - 1.0) / box_w)
        .min(scale_limit);
    let new_w = box_w * scale;
    let new_h = box_h * scale;

    let center_x = x + box_w / 2.0;
    let center_y = y + box_h / 2.0;

    let x1 = (center_x - new_w / 2.0).max(0.0) as i32;
    let y1 = (center_y - new_h / 2.0).max(0.0) as i32;
    let x2 = (center_x + new_w / 2.0).min(src_w - 1.0) as i32;
    let y2 = (center_y + new_h / 2.0).min(src_h - 1.0) as i32;

    if x2 < x1 || y2 < y1 {
        return Err(Error::AntiSpoofing(
            "Anti-spoof crop is out of bounds".to_string(),
        ));
    }

    let roi = Rect::new(x1, y1, (x2 - x1 + 1).max(1), (y2 - y1 + 1).max(1));
    let face_roi = image.roi(roi)?;
    let mut cropped = Mat::default();
    face_roi.copy_to(&mut cropped)?;

    let mut resized = Mat::default();
    imgproc::resize(
        &cropped,
        &mut resized,
        Size::new(out_size, out_size),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    Ok(resized)
}

fn forward_liveness(
    net: &mut dnn::Net,
    frame: &Mat,
    bbox: [f32; 4],
    input_size: i32,
    crop_scale: f32,
) -> Result<f32> {
    let resized = crop_face(frame, bbox, crop_scale, input_size)?;

    // Match the reference ONNX pipeline exactly: BGR input, float32, CHW,
    // no normalization or mean subtraction.
    let blob = dnn::blob_from_image(
        &resized,
        1.0,
        Size::new(input_size, input_size),
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        false,
        false,
        CV_32F,
    )?;

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

    #[test]
    fn test_crop_face_resizes_to_model_input() {
        let image = Mat::zeros(120, 160, CV_8UC3).unwrap().to_mat().unwrap();
        let cropped = crop_face(&image, [30.0, 20.0, 40.0, 50.0], 2.7, 80).unwrap();
        let size = cropped.size().unwrap();
        assert_eq!(size.width, 80);
        assert_eq!(size.height, 80);
    }
}
