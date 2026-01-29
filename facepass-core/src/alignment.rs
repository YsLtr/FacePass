//! Face alignment utilities
//!
//! Note: Most alignment is done by FaceRecognizerSF::align_crop(),
//! but this module provides additional utilities for image processing.

use crate::error::{Error, Result};
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size},
    imgproc,
    prelude::*,
};

/// Resize image while maintaining aspect ratio
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

/// Convert image to grayscale
pub fn to_grayscale(image: &Mat) -> Result<Mat> {
    let mut gray = Mat::default();
    imgproc::cvt_color_def(image, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    Ok(gray)
}

/// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
pub fn apply_clahe(gray_image: &Mat, clip_limit: f64, tile_size: i32) -> Result<Mat> {
    let mut clahe = imgproc::create_clahe(clip_limit, Size::new(tile_size, tile_size))?;
    let mut equalized = Mat::default();
    clahe.apply(gray_image, &mut equalized)?;
    Ok(equalized)
}

/// Check if image is too dark (returns darkness percentage)
pub fn check_darkness(gray_image: &Mat, threshold: f64) -> Result<f64> {
    let mut hist = Mat::default();
    let channels = opencv::core::Vector::<i32>::from(vec![0i32]);
    let hist_size = opencv::core::Vector::<i32>::from(vec![256i32]);
    let ranges = opencv::core::Vector::<f32>::from(vec![0.0f32, 256.0f32]);
    let images = opencv::core::Vector::<Mat>::from(vec![gray_image.clone()]);

    imgproc::calc_hist(
        &images,
        &channels,
        &Mat::default(),
        &mut hist,
        &hist_size,
        &ranges,
        false,
    )?;

    // Get total pixels and dark pixels (first bin, value 0)
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

/// Draw face detection results on image
pub fn draw_detection(
    image: &mut Mat,
    bbox: (f32, f32, f32, f32),
    landmarks: &[(f32, f32); 5],
) -> Result<()> {
    let (x, y, w, h) = bbox;

    // Draw bounding box
    let color = Scalar::new(0.0, 255.0, 0.0, 0.0); // Green
    imgproc::rectangle(
        image,
        Rect::new(x as i32, y as i32, w as i32, h as i32),
        color,
        2,
        imgproc::LINE_8,
        0,
    )?;

    // Draw landmarks
    let landmark_color = Scalar::new(255.0, 0.0, 0.0, 0.0); // Blue
    for (px, py) in landmarks {
        imgproc::circle(
            image,
            Point::new(*px as i32, *py as i32),
            3,
            landmark_color,
            -1,
            imgproc::LINE_AA,
            0,
        )?;
    }

    Ok(())
}

/// Crop face region from image with padding
pub fn crop_face(image: &Mat, bbox: (f32, f32, f32, f32), padding: f32) -> Result<Mat> {
    let (x, y, w, h) = bbox;
    let size = image.size()?;

    // Calculate padded region
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

    #[test]
    fn test_resize_smaller() {
        // Create a 1000x800 image
        let image = Mat::zeros(800, 1000, opencv::core::CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap();
        let resized = resize_max_dimension(&image, 500.0).unwrap();
        let size = resized.size().unwrap();

        // Should be scaled to 500x400
        assert_eq!(size.width, 500);
        assert_eq!(size.height, 400);
    }

    #[test]
    fn test_resize_no_change() {
        // Create a 300x200 image
        let image = Mat::zeros(200, 300, opencv::core::CV_8UC3)
            .unwrap()
            .to_mat()
            .unwrap();
        let resized = resize_max_dimension(&image, 500.0).unwrap();
        let size = resized.size().unwrap();

        // Should remain 300x200
        assert_eq!(size.width, 300);
        assert_eq!(size.height, 200);
    }
}
