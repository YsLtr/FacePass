//! Anti-spoofing module using MiniFASNet ONNX models.

use crate::config::{AntiSpoofConfig, AntiSpoofMode, ModelsConfig};
use crate::error::{Error, Result};
use crate::face_validation::{face_row_to_bbox, validate_face};
use opencv::{
    core::{self, Mat, Rect, Scalar, Size, Vector, CV_32F, CV_8UC3},
    dnn, imgproc,
    prelude::*,
};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy)]
struct AntiSpoofProfileSpec {
    label: &'static str,
    input_size: i32,
    crop_scale: f32,
}

struct LoadedAntiSpoofModel {
    label: &'static str,
    input_size: i32,
    crop_scale: f32,
    net: Arc<Mutex<dnn::Net>>,
}

/// Anti-spoofing detector using MiniFASNet ONNX models.
pub struct AntiSpoofDetector {
    models: Vec<LoadedAntiSpoofModel>,
}

// Safety: We protect every net with a Mutex.
unsafe impl Send for AntiSpoofDetector {}
unsafe impl Sync for AntiSpoofDetector {}

impl AntiSpoofDetector {
    /// Create a new anti-spoofing detector.
    pub fn new(models_config: &ModelsConfig, config: &AntiSpoofConfig) -> Result<Self> {
        let mut models = Vec::new();

        for (profile, model_path) in active_model_sources(models_config, config) {
            let mut net = dnn::read_net_from_onnx(&model_path).map_err(|e| {
                Error::AntiSpoofing(format!(
                    "Failed to load {} anti-spoofing model from {}: {}",
                    profile.label, model_path, e
                ))
            })?;

            let input_size =
                validate_or_probe_input_size(&mut net, profile.input_size, profile.crop_scale)?;

            models.push(LoadedAntiSpoofModel {
                label: profile.label,
                input_size,
                crop_scale: profile.crop_scale,
                net: Arc::new(Mutex::new(net)),
            });
        }

        if models.is_empty() {
            return Err(Error::AntiSpoofing(
                "No anti-spoofing model is configured".to_string(),
            ));
        }

        Ok(Self { models })
    }

    /// Check if a detected face is live (not spoofed).
    ///
    /// The input face must be a valid full face and the required crop range
    /// must be completely inside the frame for the configured valid crop scale.
    /// Returns the fused probability of the "real" class (index 1).
    pub fn check_liveness(
        &self,
        frame: &Mat,
        face_row: &Mat,
        valid_crop_scale: f32,
    ) -> Result<f32> {
        validate_face(frame, face_row, valid_crop_scale)?;
        let bbox = face_row_to_bbox(face_row)?;

        let mut fused_probs: Option<Vec<f32>> = None;
        let mut model_count = 0usize;

        for model in &self.models {
            let mut net = model.net.lock().map_err(|e| {
                Error::AntiSpoofing(format!("Failed to lock {} net: {}", model.label, e))
            })?;

            let probs = forward_probs(&mut net, frame, bbox, model.input_size, model.crop_scale)?;

            if probs.len() < 2 {
                return Err(Error::AntiSpoofing(format!(
                    "{} output must contain at least 2 classes, got {}",
                    model.label,
                    probs.len()
                )));
            }

            match fused_probs.as_mut() {
                Some(fused) => {
                    if fused.len() != probs.len() {
                        return Err(Error::AntiSpoofing(format!(
                            "Incompatible anti-spoof output sizes in fusion: {} vs {}",
                            fused.len(),
                            probs.len()
                        )));
                    }
                    for (dst, src) in fused.iter_mut().zip(probs.iter()) {
                        *dst += *src;
                    }
                }
                None => fused_probs = Some(probs),
            }

            model_count += 1;
        }

        let mut fused = fused_probs.ok_or_else(|| {
            Error::AntiSpoofing("Anti-spoofing produced no model outputs".to_string())
        })?;
        let divisor = model_count as f32;
        for value in &mut fused {
            *value /= divisor;
        }

        Ok(fused[1])
    }
}

fn active_profile_specs(config: &AntiSpoofConfig) -> Vec<AntiSpoofProfileSpec> {
    match config.mode {
        AntiSpoofMode::MiniFASNetV2 => vec![AntiSpoofProfileSpec {
            label: "MiniFASNetV2",
            input_size: config.v2_input_size,
            crop_scale: config.v2_crop_scale,
        }],
        AntiSpoofMode::MiniFASNetV1SE => vec![AntiSpoofProfileSpec {
            label: "MiniFASNetV1SE",
            input_size: config.v1se_input_size,
            crop_scale: config.v1se_crop_scale,
        }],
        AntiSpoofMode::Fusion => vec![
            AntiSpoofProfileSpec {
                label: "MiniFASNetV2",
                input_size: config.v2_input_size,
                crop_scale: config.v2_crop_scale,
            },
            AntiSpoofProfileSpec {
                label: "MiniFASNetV1SE",
                input_size: config.v1se_input_size,
                crop_scale: config.v1se_crop_scale,
            },
        ],
    }
}

fn active_model_sources(
    models_config: &ModelsConfig,
    config: &AntiSpoofConfig,
) -> Vec<(AntiSpoofProfileSpec, String)> {
    active_profile_specs(config)
        .into_iter()
        .map(|profile| {
            let path = match profile.label {
                "MiniFASNetV2" => models_config.anti_spoof_v2_path.clone(),
                "MiniFASNetV1SE" => models_config.anti_spoof_v1se_path.clone(),
                _ => String::new(),
            };
            (profile, path)
        })
        .collect()
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
    for size in [80, 40, 112, 128, 64] {
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
        match forward_probs_relaxed(net, &dummy, bbox, *size, crop_scale) {
            Ok(_) => {
                if let Some(level) = prev_level {
                    let _ = core::set_log_level(level);
                }
                return Ok(*size);
            }
            Err(Error::OpenCV(e)) => last_err = Some(e),
            Err(Error::AntiSpoofing(e)) => last_err = Some(e),
            Err(_) => {}
        }
    }

    if let Some(level) = prev_level {
        let _ = core::set_log_level(level);
    }

    Err(Error::AntiSpoofing(format!(
        "Failed to infer anti-spoof input size; tried {:?}. Last error: {}",
        sizes,
        last_err.unwrap_or_else(|| "unknown error".to_string())
    )))
}

fn crop_face(image: &Mat, bbox: [f32; 4], scale_limit: f32, out_size: i32) -> Result<Mat> {
    let rect = compute_model_crop_rect(
        image.size()?.width as f32,
        image.size()?.height as f32,
        bbox,
        scale_limit,
    )?;
    let roi = normalize_roi_rect(image.cols(), image.rows(), rect)?;
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

fn normalize_roi_rect(image_w: i32, image_h: i32, rect: (i32, i32, i32, i32)) -> Result<Rect> {
    if image_w <= 0 || image_h <= 0 {
        return Err(Error::AntiSpoofing(
            "Cannot crop anti-spoof ROI from an empty frame".to_string(),
        ));
    }

    let (x, y, width, height) = rect;
    if width <= 0 || height <= 0 {
        return Err(Error::AntiSpoofing(format!(
            "Invalid anti-spoof ROI dimensions: x={}, y={}, width={}, height={}",
            x, y, width, height
        )));
    }

    let max_x = image_w.saturating_sub(1);
    let max_y = image_h.saturating_sub(1);
    let clamped_x = x.clamp(0, max_x);
    let clamped_y = y.clamp(0, max_y);
    let clamped_width = width.min(image_w - clamped_x).max(1);
    let clamped_height = height.min(image_h - clamped_y).max(1);

    Ok(Rect::new(
        clamped_x,
        clamped_y,
        clamped_width,
        clamped_height,
    ))
}

fn compute_model_crop_rect(
    frame_w: f32,
    frame_h: f32,
    bbox: [f32; 4],
    scale_limit: f32,
) -> Result<(i32, i32, i32, i32)> {
    if let Ok(rect) = compute_flexible_crop_rect(frame_w, frame_h, bbox, scale_limit) {
        return Ok(rect);
    }

    let [_, _, box_w, box_h] = bbox;
    if box_w <= 0.0 || box_h <= 0.0 {
        return Err(Error::InvalidFace(
            "Detected face bbox is invalid".to_string(),
        ));
    }

    let target_area = box_w * scale_limit * box_h * scale_limit;
    let target_ratio = (box_w / box_h).max(f32::EPSILON);

    let ideal_w = target_area.sqrt() * target_ratio.sqrt();
    let ideal_h = target_area.sqrt() / target_ratio.sqrt();

    let preferred_w = ideal_w.clamp(box_w, frame_w);
    let preferred_h = (target_area / preferred_w).max(box_h).min(frame_h);
    if preferred_w >= box_w && preferred_h >= box_h {
        if let Ok(rect) = build_containing_rect(frame_w, frame_h, bbox, preferred_w, preferred_h) {
            return Ok(rect);
        }
    }

    let preferred_h = ideal_h.clamp(box_h, frame_h);
    let preferred_w = (target_area / preferred_h).max(box_w).min(frame_w);
    if preferred_w >= box_w && preferred_h >= box_h {
        if let Ok(rect) = build_containing_rect(frame_w, frame_h, bbox, preferred_w, preferred_h) {
            return Ok(rect);
        }
    }

    let full_w = frame_w.max(box_w);
    let full_h = frame_h.max(box_h);
    if let Ok(rect) = build_containing_rect(frame_w, frame_h, bbox, full_w, full_h) {
        return Ok(rect);
    }

    Err(Error::InvalidFace(
        "Face cannot fit within the available camera frame".to_string(),
    ))
}

fn compute_flexible_crop_rect(
    frame_w: f32,
    frame_h: f32,
    bbox: [f32; 4],
    scale_limit: f32,
) -> Result<(i32, i32, i32, i32)> {
    let [x, y, box_w, box_h] = bbox;

    if box_w <= 0.0 || box_h <= 0.0 {
        return Err(Error::InvalidFace(
            "Detected face bbox is invalid".to_string(),
        ));
    }

    let new_w = box_w * scale_limit;
    let new_h = box_h * scale_limit;

    if new_w > frame_w || new_h > frame_h {
        return Err(Error::InvalidFace(format!(
            "Face must leave full background context for valid crop scale {:.1}",
            scale_limit
        )));
    }

    let min_left = (x + box_w - new_w).max(0.0);
    let max_left = x.min(frame_w - new_w);
    if min_left > max_left {
        return Err(Error::InvalidFace(
            "Face cannot fit within the valid crop window".to_string(),
        ));
    }

    let min_top = (y + box_h - new_h).max(0.0);
    let max_top = y.min(frame_h - new_h);
    if min_top > max_top {
        return Err(Error::InvalidFace(
            "Face cannot fit within the valid crop window".to_string(),
        ));
    }

    let centered_left = x + box_w / 2.0 - new_w / 2.0;
    let centered_top = y + box_h / 2.0 - new_h / 2.0;

    let left = centered_left.clamp(min_left, max_left);
    let top = centered_top.clamp(min_top, max_top);

    Ok((
        left.round() as i32,
        top.round() as i32,
        new_w.round().max(1.0) as i32,
        new_h.round().max(1.0) as i32,
    ))
}

fn build_containing_rect(
    frame_w: f32,
    frame_h: f32,
    bbox: [f32; 4],
    rect_w: f32,
    rect_h: f32,
) -> Result<(i32, i32, i32, i32)> {
    let [x, y, box_w, box_h] = bbox;

    if box_w <= 0.0 || box_h <= 0.0 {
        return Err(Error::InvalidFace(
            "Detected face bbox is invalid".to_string(),
        ));
    }

    if rect_w < box_w || rect_h < box_h || rect_w > frame_w || rect_h > frame_h {
        return Err(Error::InvalidFace(
            "Face cannot fit within the valid crop window".to_string(),
        ));
    }

    let min_left = (x + box_w - rect_w).max(0.0);
    let max_left = x.min(frame_w - rect_w);
    if min_left > max_left {
        return Err(Error::InvalidFace(
            "Face cannot fit within the valid crop window".to_string(),
        ));
    }

    let min_top = (y + box_h - rect_h).max(0.0);
    let max_top = y.min(frame_h - rect_h);
    if min_top > max_top {
        return Err(Error::InvalidFace(
            "Face cannot fit within the valid crop window".to_string(),
        ));
    }

    let centered_left = x + box_w / 2.0 - rect_w / 2.0;
    let centered_top = y + box_h / 2.0 - rect_h / 2.0;
    let left = centered_left.clamp(min_left, max_left);
    let top = centered_top.clamp(min_top, max_top);

    Ok((
        left.round() as i32,
        top.round() as i32,
        rect_w.round().max(1.0) as i32,
        rect_h.round().max(1.0) as i32,
    ))
}

fn crop_face_relaxed(image: &Mat, bbox: [f32; 4], scale_limit: f32, out_size: i32) -> Result<Mat> {
    let size = image.size()?;
    let src_w = size.width as f32;
    let src_h = size.height as f32;
    let [x, y, box_w, box_h] = bbox;

    if box_w <= 0.0 || box_h <= 0.0 {
        return Err(Error::AntiSpoofing(
            "Invalid anti-spoof bbox for relaxed crop".to_string(),
        ));
    }

    let scale = ((src_h - 1.0) / box_h)
        .min((src_w - 1.0) / box_w)
        .min(scale_limit);
    let new_w = box_w * scale;
    let new_h = box_h * scale;

    let center_x = x + box_w / 2.0;
    let center_y = y + box_h / 2.0;

    let x1 = (center_x - new_w / 2.0).max(0.0).round() as i32;
    let y1 = (center_y - new_h / 2.0).max(0.0).round() as i32;
    let x2 = (center_x + new_w / 2.0).min(src_w - 1.0).round() as i32;
    let y2 = (center_y + new_h / 2.0).min(src_h - 1.0).round() as i32;

    if x2 < x1 || y2 < y1 {
        return Err(Error::AntiSpoofing(
            "Relaxed anti-spoof crop is out of bounds".to_string(),
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

fn forward_probs(
    net: &mut dnn::Net,
    frame: &Mat,
    bbox: [f32; 4],
    input_size: i32,
    crop_scale: f32,
) -> Result<Vec<f32>> {
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
    let data = output.data_typed::<f32>()?;
    if data.len() < 2 {
        return Err(Error::AntiSpoofing(format!(
            "Unexpected output size: {}",
            data.len()
        )));
    }

    Ok(softmax(data))
}

fn forward_probs_relaxed(
    net: &mut dnn::Net,
    frame: &Mat,
    bbox: [f32; 4],
    input_size: i32,
    crop_scale: f32,
) -> Result<Vec<f32>> {
    let resized = crop_face_relaxed(frame, bbox, crop_scale, input_size)?;

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
    let data = output.data_typed::<f32>()?;
    if data.len() < 2 {
        return Err(Error::AntiSpoofing(format!(
            "Unexpected output size: {}",
            data.len()
        )));
    }

    Ok(softmax(data))
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let exps: Vec<f32> = logits
        .iter()
        .map(|value| (*value - max_val).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|value| value / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_probability() {
        let probs = softmax(&[-5.0, 5.0, -2.0]);
        assert!(probs[1] > 0.99);

        let probs = softmax(&[5.0, -5.0, 1.0]);
        assert!(probs[1] < 0.01);

        let probs = softmax(&[1.0, 1.0, 1.0]);
        assert!((probs[0] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((probs[1] - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_crop_face_resizes_to_model_input() {
        let image = Mat::zeros(240, 320, CV_8UC3).unwrap().to_mat().unwrap();
        let cropped = crop_face(&image, [100.0, 80.0, 40.0, 50.0], 2.7, 80).unwrap();
        let size = cropped.size().unwrap();
        assert_eq!(size.width, 80);
        assert_eq!(size.height, 80);
    }

    #[test]
    fn test_normalize_roi_rect_clamps_rounded_overflow() {
        let roi = normalize_roi_rect(640, 480, (1, 0, 640, 480)).unwrap();
        assert_eq!(roi.x, 1);
        assert_eq!(roi.y, 0);
        assert_eq!(roi.width, 639);
        assert_eq!(roi.height, 480);
    }
}
