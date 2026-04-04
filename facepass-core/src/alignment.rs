//! Face alignment utilities

use crate::error::{Error, Result};
use crate::models::{DetectionResult, FaceLandmarks};
use opencv::{
    calib3d,
    core::{Mat, Point, Point2f, Rect, Scalar, Size, Vector},
    imgproc,
    prelude::*,
};

const ARC_FACE_TEMPLATE_112: [[f32; 2]; 5] = [
    [46.2946, 51.6963],
    [81.5318, 51.5014],
    [64.0252, 71.7366],
    [49.5493, 92.3655],
    [78.7299, 92.2041],
];

pub fn resize_max_dimension(image: &Mat, max_dim: f32) -> Result<Mat> {
    let size = image.size()?;
    let scale = (max_dim / (size.width.max(size.height) as f32)).min(1.0);

    if scale >= 1.0 {
        return Ok(image.clone());
    }

    let new_size = Size::new(
        (size.width as f32 * scale) as i32,
        (size.height as f32 * scale) as i32,
    );

    let mut resized = Mat::default();
    imgproc::resize(image, &mut resized, new_size, 0.0, 0.0, imgproc::INTER_AREA)?;

    Ok(resized)
}

pub fn to_grayscale(image: &Mat) -> Result<Mat> {
    let mut gray = Mat::default();
    imgproc::cvt_color_def(image, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    Ok(gray)
}

pub fn apply_clahe(gray_image: &Mat, clip_limit: f64, tile_size: i32) -> Result<Mat> {
    let mut clahe = imgproc::create_clahe(clip_limit, Size::new(tile_size, tile_size))?;
    let mut equalized = Mat::default();
    clahe.apply(gray_image, &mut equalized)?;
    Ok(equalized)
}

pub fn check_darkness(gray_image: &Mat, threshold: f64) -> Result<f64> {
    let mut hist = Mat::default();
    let channels = Vector::<i32>::from(vec![0i32]);
    let hist_size = Vector::<i32>::from(vec![256i32]);
    let ranges = Vector::<f32>::from(vec![0.0f32, 256.0f32]);
    let images = Vector::<Mat>::from(vec![gray_image.clone()]);

    imgproc::calc_hist(
        &images,
        &channels,
        &Mat::default(),
        &mut hist,
        &hist_size,
        &ranges,
        false,
    )?;

    let total: f32 = hist.iter::<f32>()?.map(|(_, v)| v).sum();
    let dark_pixels = *hist.at::<f32>(0)?;
    let darkness = (dark_pixels / total * 100.0) as f64;

    if darkness > threshold {
        Err(Error::Detection(format!(
            "Image too dark: {:.1}% > {:.1}%",
            darkness, threshold
        )))
    } else {
        Ok(darkness)
    }
}

pub fn arcface_template(output_size: Size) -> [Point2f; 5] {
    let scale_x = output_size.width as f32 / 112.0;
    let scale_y = output_size.height as f32 / 112.0;

    ARC_FACE_TEMPLATE_112.map(|[x, y]| Point2f::new(x * scale_x, y * scale_y))
}

pub fn align_face(image: &Mat, landmarks: &FaceLandmarks, output_size: Size) -> Result<Mat> {
    let src = Vector::<Point2f>::from_iter(
        landmarks
            .arcface_points()
            .into_iter()
            .map(|(x, y)| Point2f::new(x, y)),
    );
    let dst = Vector::<Point2f>::from_iter(arcface_template(output_size));
    let transform = calib3d::estimate_affine_partial_2d_def(&src, &dst)?;

    if transform.empty() {
        return Err(Error::Recognition(
            "Failed to estimate face alignment transform".to_string(),
        ));
    }

    let mut aligned = Mat::default();
    imgproc::warp_affine(
        image,
        &mut aligned,
        &transform,
        output_size,
        imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
    )?;

    Ok(aligned)
}

pub fn align_detected_face(image: &Mat, detection: &DetectionResult, output_size: Size) -> Result<Mat> {
    align_face(image, &detection.landmarks, output_size)
}

pub fn draw_detection(image: &mut Mat, detection: &DetectionResult) -> Result<()> {
    let (x, y, w, h) = detection.bbox;

    imgproc::rectangle(
        image,
        Rect::new(x as i32, y as i32, w as i32, h as i32),
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;

    let landmark_color = Scalar::new(255.0, 0.0, 0.0, 0.0);
    for (px, py) in detection.landmarks.arcface_points() {
        imgproc::circle(
            image,
            Point::new(px as i32, py as i32),
            3,
            landmark_color,
            -1,
            imgproc::LINE_AA,
            0,
        )?;
    }

    Ok(())
}

pub fn crop_face(image: &Mat, bbox: (f32, f32, f32, f32), padding: f32) -> Result<Mat> {
    let (x, y, w, h) = bbox;
    let size = image.size()?;

    let pad_w = w * padding;
    let pad_h = h * padding;

    let x1 = (x - pad_w).max(0.0) as i32;
    let y1 = (y - pad_h).max(0.0) as i32;
    let x2 = (x + w + pad_w).min(size.width as f32) as i32;
    let y2 = (y + h + pad_h).min(size.height as f32) as i32;

    let roi = Rect::new(x1, y1, x2 - x1, y2 - y1);
    let cropped = Mat::roi(image, roi)?;

    Ok(cropped.try_clone()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::FaceLandmarks;

    #[test]
    fn test_resize_smaller() {
        let image = Mat::zeros(800, 1000, opencv::core::CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap();
        let resized = resize_max_dimension(&image, 500.0).unwrap();
        let size = resized.size().unwrap();

        assert_eq!(size.width, 500);
        assert_eq!(size.height, 400);
    }

    #[test]
    fn test_resize_no_change() {
        let image = Mat::zeros(200, 300, opencv::core::CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap();
        let resized = resize_max_dimension(&image, 500.0).unwrap();
        let size = resized.size().unwrap();

        assert_eq!(size.width, 300);
        assert_eq!(size.height, 200);
    }

    #[test]
    fn test_arcface_template_scales_to_output_size() {
        let template = arcface_template(Size::new(112, 112));
        assert!((template[0].x - 46.2946).abs() < 1e-4);
        assert!((template[4].y - 92.2041).abs() < 1e-4);

        let template_224 = arcface_template(Size::new(224, 224));
        assert!((template_224[0].x - 92.5892).abs() < 1e-4);
    }

    #[test]
    fn test_align_face_outputs_requested_size() {
        let image = Mat::zeros(160, 160, opencv::core::CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap();
        let landmarks = FaceLandmarks::from_arcface_order([
            (46.2946, 51.6963),
            (81.5318, 51.5014),
            (64.0252, 71.7366),
            (49.5493, 92.3655),
            (78.7299, 92.2041),
        ]);
        let aligned = align_face(&image, &landmarks, Size::new(112, 112)).unwrap();
        let size = aligned.size().unwrap();
        assert_eq!(size.width, 112);
        assert_eq!(size.height, 112);
    }
}
