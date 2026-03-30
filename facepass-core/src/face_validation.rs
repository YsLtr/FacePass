//! Face validation and primary-face selection.

use crate::error::{Error, Result};
use opencv::{
    core::{Mat, Rect},
    prelude::*,
};

/// Select the most confident valid face from a YuNet detections matrix.
pub fn select_primary_face(frame: &Mat, faces: &Mat, valid_crop_scale: f32) -> Result<Mat> {
    if faces.rows() == 0 {
        return Err(Error::NoFaceDetected);
    }

    let mut best_valid: Option<(f32, Mat)> = None;
    let mut best_invalid: Option<(f32, String)> = None;

    for row_idx in 0..faces.rows() {
        let face_row = faces.row(row_idx)?.try_clone()?;
        let confidence = face_row_confidence(&face_row).unwrap_or(0.0);

        match validate_face(frame, &face_row, valid_crop_scale) {
            Ok(()) => match &best_valid {
                Some((best_conf, _)) if *best_conf >= confidence => {}
                _ => best_valid = Some((confidence, face_row)),
            },
            Err(Error::InvalidFace(reason)) => match &best_invalid {
                Some((best_conf, _)) if *best_conf >= confidence => {}
                _ => best_invalid = Some((confidence, reason)),
            },
            Err(e) => return Err(e),
        }
    }

    if let Some((_, face_row)) = best_valid {
        return Ok(face_row);
    }

    if let Some((_, reason)) = best_invalid {
        return Err(Error::InvalidFace(reason));
    }

    Err(Error::NoFaceDetected)
}

/// Validate that the face is complete and the configured valid crop window
/// fully fits inside the frame.
pub fn validate_face(frame: &Mat, face_row: &Mat, valid_crop_scale: f32) -> Result<()> {
    let frame_size = frame.size()?;
    let frame_w = frame_size.width as f32;
    let frame_h = frame_size.height as f32;

    if frame_w <= 1.0 || frame_h <= 1.0 {
        return Err(Error::InvalidFace("Frame size is invalid".to_string()));
    }

    let [x, y, box_w, box_h] = face_row_to_bbox(face_row)?;
    if box_w <= 0.0 || box_h <= 0.0 {
        return Err(Error::InvalidFace(
            "Detected face bbox is invalid".to_string(),
        ));
    }

    let x2 = x + box_w;
    let y2 = y + box_h;
    if x < 0.0 || y < 0.0 || x2 > frame_w || y2 > frame_h {
        return Err(Error::InvalidFace(
            "Detected face is not fully inside the frame".to_string(),
        ));
    }

    let landmarks = face_row_to_landmarks(face_row)?;
    for (idx, (lx, ly)) in landmarks.iter().enumerate() {
        if !lx.is_finite() || !ly.is_finite() {
            return Err(Error::InvalidFace(format!(
                "Face landmark {} is invalid",
                idx + 1
            )));
        }
        if *lx < 0.0 || *ly < 0.0 || *lx > frame_w || *ly > frame_h {
            return Err(Error::InvalidFace(format!(
                "Face landmark {} is outside the frame",
                idx + 1
            )));
        }
        if *lx < x || *lx > x2 || *ly < y || *ly > y2 {
            return Err(Error::InvalidFace(format!(
                "Face landmark {} is outside the detected face box",
                idx + 1
            )));
        }
    }

    let (_, valid) = compute_valid_crop_rect(frame, face_row, valid_crop_scale)?;
    if !valid {
        return Err(Error::InvalidFace(format!(
            "Face must leave full background context for valid crop scale {:.1}",
            valid_crop_scale
        )));
    }

    Ok(())
}

pub fn face_row_to_bbox(face_row: &Mat) -> Result<[f32; 4]> {
    if face_row.total() < 4 {
        return Err(Error::Detection(
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

pub fn face_row_to_landmarks(face_row: &Mat) -> Result<[(f32, f32); 5]> {
    if face_row.total() < 14 {
        return Err(Error::Detection(
            "Face detector output does not contain landmarks".to_string(),
        ));
    }

    Ok([
        (*face_row.at_2d::<f32>(0, 4)?, *face_row.at_2d::<f32>(0, 5)?),
        (*face_row.at_2d::<f32>(0, 6)?, *face_row.at_2d::<f32>(0, 7)?),
        (*face_row.at_2d::<f32>(0, 8)?, *face_row.at_2d::<f32>(0, 9)?),
        (
            *face_row.at_2d::<f32>(0, 10)?,
            *face_row.at_2d::<f32>(0, 11)?,
        ),
        (
            *face_row.at_2d::<f32>(0, 12)?,
            *face_row.at_2d::<f32>(0, 13)?,
        ),
    ])
}

pub fn face_row_confidence(face_row: &Mat) -> Result<f32> {
    if face_row.total() < 15 {
        return Err(Error::Detection(
            "Face detector output does not contain confidence".to_string(),
        ));
    }

    Ok(*face_row.at_2d::<f32>(0, 14)?)
}

fn compute_target_crop_area(bbox: [f32; 4], scale_limit: f32) -> Result<f32> {
    let [_, _, box_w, box_h] = bbox;

    if box_w <= 0.0 || box_h <= 0.0 {
        return Err(Error::InvalidFace(
            "Detected face bbox is invalid".to_string(),
        ));
    }

    Ok(box_w * box_h * scale_limit * scale_limit)
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

fn compute_adaptive_valid_crop_rect(
    frame_w: f32,
    frame_h: f32,
    bbox: [f32; 4],
    scale_limit: f32,
) -> Result<(i32, i32, i32, i32)> {
    let [_, _, box_w, box_h] = bbox;
    let target_area = compute_target_crop_area(bbox, scale_limit)?;
    let square_side = target_area.sqrt();

    if box_w > frame_w || box_h > frame_h || frame_w * frame_h + f32::EPSILON < target_area {
        return Err(Error::InvalidFace(format!(
            "Face must leave full background context for valid crop scale {:.1}",
            scale_limit
        )));
    }

    let preferred_w = square_side.clamp(box_w, frame_w);
    let preferred_h = (target_area / preferred_w).max(box_h);
    if preferred_h <= frame_h + f32::EPSILON {
        return build_containing_rect(frame_w, frame_h, bbox, preferred_w, preferred_h);
    }

    let preferred_h = square_side.clamp(box_h, frame_h);
    let preferred_w = (target_area / preferred_h).max(box_w);
    if preferred_w <= frame_w + f32::EPSILON {
        return build_containing_rect(frame_w, frame_h, bbox, preferred_w, preferred_h);
    }

    let stretched_w = frame_w;
    let stretched_h = (target_area / stretched_w).max(box_h);
    if stretched_h <= frame_h + f32::EPSILON {
        return build_containing_rect(frame_w, frame_h, bbox, stretched_w, stretched_h);
    }

    let stretched_h = frame_h;
    let stretched_w = (target_area / stretched_h).max(box_w);
    if stretched_w <= frame_w + f32::EPSILON {
        return build_containing_rect(frame_w, frame_h, bbox, stretched_w, stretched_h);
    }

    Err(Error::InvalidFace(format!(
        "Face must leave full background context for valid crop scale {:.1}",
        scale_limit
    )))
}

pub fn compute_valid_crop_rect(
    frame: &Mat,
    face_row: &Mat,
    scale_limit: f32,
) -> Result<(Rect, bool)> {
    let size = frame.size()?;
    let frame_w = size.width as f32;
    let frame_h = size.height as f32;
    let bbox = face_row_to_bbox(face_row)?;

    let [x, y, box_w, box_h] = bbox;
    if box_w <= 0.0 || box_h <= 0.0 {
        return Err(Error::InvalidFace(
            "Detected face bbox is invalid".to_string(),
        ));
    }

    let target_area = compute_target_crop_area(bbox, scale_limit)?;
    let square_side = target_area.sqrt();

    let fallback_rect = {
        let width = square_side.min(frame_w.max(box_w));
        let height = (target_area / width).max(box_h).min(frame_h.max(box_h));
        let width = (target_area / height).max(box_w).min(frame_w.max(box_w));
        let centered_left = x + box_w / 2.0 - width / 2.0;
        let centered_top = y + box_h / 2.0 - height / 2.0;
        Rect::new(
            centered_left.round() as i32,
            centered_top.round() as i32,
            width.round().max(1.0) as i32,
            height.round().max(1.0) as i32,
        )
    };

    match compute_adaptive_valid_crop_rect(frame_w, frame_h, bbox, scale_limit) {
        Ok((x, y, w, h)) => Ok((Rect::new(x, y, w, h), true)),
        Err(Error::InvalidFace(_)) => Ok((fallback_rect, false)),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{CV_32F, CV_8UC3};

    fn make_face_row(values: [f32; 15]) -> Mat {
        let mut mat = Mat::zeros(1, 15, CV_32F).unwrap().to_mat().unwrap();
        for (idx, value) in values.iter().enumerate() {
            *mat.at_2d_mut::<f32>(0, idx as i32).unwrap() = *value;
        }
        mat
    }

    #[test]
    fn test_validate_face_rejects_border_face_for_scale_four() {
        let frame = Mat::zeros(240, 320, CV_8UC3).unwrap().to_mat().unwrap();
        let face_row = make_face_row([
            20.0, 40.0, 100.0, 120.0, 45.0, 80.0, 95.0, 80.0, 70.0, 110.0, 50.0, 140.0, 90.0,
            140.0, 0.95,
        ]);
        let valid_crop_scale = 4.0;
        let err = validate_face(&frame, &face_row, valid_crop_scale).unwrap_err();
        assert!(matches!(err, Error::InvalidFace(_)));
    }

    #[test]
    fn test_validate_face_accepts_centered_face() {
        let frame = Mat::zeros(480, 640, CV_8UC3).unwrap().to_mat().unwrap();
        let face_row = make_face_row([
            240.0, 130.0, 100.0, 120.0, 265.0, 170.0, 315.0, 170.0, 290.0, 205.0, 270.0, 230.0,
            310.0, 230.0, 0.99,
        ]);
        let valid_crop_scale = 2.7;
        assert!(validate_face(&frame, &face_row, valid_crop_scale).is_ok());
    }

    #[test]
    fn test_valid_crop_rect_is_square() {
        let frame = Mat::zeros(480, 640, CV_8UC3).unwrap().to_mat().unwrap();
        let face_row = make_face_row([
            240.0, 130.0, 100.0, 120.0, 265.0, 170.0, 315.0, 170.0, 290.0, 205.0, 270.0, 230.0,
            310.0, 230.0, 0.99,
        ]);

        let (rect, valid) = compute_valid_crop_rect(&frame, &face_row, 2.7).unwrap();
        assert!(valid);
        assert_eq!(rect.width, rect.height);
    }

    #[test]
    fn test_valid_crop_rect_can_deform_to_preserve_area() {
        let frame = Mat::zeros(180, 260, CV_8UC3).unwrap().to_mat().unwrap();
        let face_row = make_face_row([
            80.0, 30.0, 150.0, 100.0, 110.0, 65.0, 190.0, 65.0, 150.0, 90.0, 120.0, 110.0, 180.0,
            110.0, 0.99,
        ]);

        let (rect, valid) = compute_valid_crop_rect(&frame, &face_row, 1.5).unwrap();
        assert!(valid);
        assert_ne!(rect.width, rect.height);
        assert!((rect.width * rect.height) as f32 >= 150.0 * 100.0 * 1.5 * 1.5);
    }
}
