//! Face detection backends.

use crate::config::{DetectorKind, DetectorModelConfig, ModelsConfig};
use crate::error::{Error, Result};
use crate::models::{DetectionResult, FaceLandmarks};
use opencv::{
    core::{Mat, Ptr, Rect, Scalar, Size, Vector, CV_32F, CV_8UC3},
    dnn,
    objdetect::FaceDetectorYN,
    prelude::*,
};
use std::cmp::Ordering;
use std::sync::{Arc, Mutex};

pub struct FaceDetector {
    backend: DetectorBackendImpl,
    kind: DetectorKind,
}

enum DetectorBackendImpl {
    YuNet {
        detector: Arc<Mutex<Ptr<FaceDetectorYN>>>,
    },
    Scrfd {
        net: Arc<Mutex<dnn::Net>>,
        config: DetectorModelConfig,
        output_names: Vector<String>,
    },
}

unsafe impl Send for FaceDetector {}
unsafe impl Sync for FaceDetector {}

impl FaceDetector {
    pub fn new(models: &ModelsConfig) -> Result<Self> {
        match models.active_detector {
            DetectorKind::Yunet => Self::build_yunet(&models.yunet),
            DetectorKind::Scrfd => Self::build_scrfd(&models.scrfd),
        }
    }

    pub fn kind(&self) -> DetectorKind {
        self.kind
    }

    fn build_yunet(config: &DetectorModelConfig) -> Result<Self> {
        let detector = FaceDetectorYN::create(
            &config.path,
            "",
            Size::new(config.input_width, config.input_height),
            config.score_threshold,
            config.nms_threshold,
            5000,
            0,
            0,
        )
        .map_err(|e| Error::Detection(format!("Failed to create YuNet detector: {}", e)))?;

        Ok(Self {
            backend: DetectorBackendImpl::YuNet {
                detector: Arc::new(Mutex::new(detector)),
            },
            kind: DetectorKind::Yunet,
        })
    }

    fn build_scrfd(config: &DetectorModelConfig) -> Result<Self> {
        let net = dnn::read_net_from_onnx(&config.path).map_err(|e| {
            Error::Detection(format!(
                "Failed to load SCRFD detector from {}: {}",
                config.path, e
            ))
        })?;

        let mut requested_outputs = Vector::<String>::new();
        for name in [
            "score_8",
            "bbox_8",
            "kps_8",
            "score_16",
            "bbox_16",
            "kps_16",
            "score_32",
            "bbox_32",
            "kps_32",
        ] {
            requested_outputs.push(name);
        }

        Ok(Self {
            backend: DetectorBackendImpl::Scrfd {
                net: Arc::new(Mutex::new(net)),
                config: config.clone(),
                output_names: requested_outputs,
            },
            kind: DetectorKind::Scrfd,
        })
    }

    pub fn detect(&self, image: &Mat) -> Result<Vec<DetectionResult>> {
        match &self.backend {
            DetectorBackendImpl::YuNet { detector } => detect_yunet(detector, image),
            DetectorBackendImpl::Scrfd {
                net,
                config,
                output_names,
            } => detect_scrfd(net, config, output_names, image),
        }
    }

    pub fn detect_single(&self, image: &Mat) -> Result<DetectionResult> {
        self.detect(image)?
            .into_iter()
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(Ordering::Equal)
            })
            .ok_or(Error::NoFaceDetected)
    }
}

fn detect_yunet(detector: &Arc<Mutex<Ptr<FaceDetectorYN>>>, image: &Mat) -> Result<Vec<DetectionResult>> {
    let mut detector = detector
        .lock()
        .map_err(|e| Error::Detection(format!("Failed to lock detector: {}", e)))?;

    detector.set_input_size(image.size()?)?;

    let mut faces = Mat::default();
    detector.detect(image, &mut faces)?;

    let mut results = Vec::new();
    for row in 0..faces.rows() {
        let bbox = (
            *faces.at_2d::<f32>(row, 0)?,
            *faces.at_2d::<f32>(row, 1)?,
            *faces.at_2d::<f32>(row, 2)?,
            *faces.at_2d::<f32>(row, 3)?,
        );
        let landmarks = FaceLandmarks::from_yunet_order([
            (*faces.at_2d::<f32>(row, 4)?, *faces.at_2d::<f32>(row, 5)?),
            (*faces.at_2d::<f32>(row, 6)?, *faces.at_2d::<f32>(row, 7)?),
            (*faces.at_2d::<f32>(row, 8)?, *faces.at_2d::<f32>(row, 9)?),
            (*faces.at_2d::<f32>(row, 10)?, *faces.at_2d::<f32>(row, 11)?),
            (*faces.at_2d::<f32>(row, 12)?, *faces.at_2d::<f32>(row, 13)?),
        ]);
        let confidence = *faces.at_2d::<f32>(row, 14)?;

        results.push(DetectionResult {
            bbox,
            confidence,
            landmarks,
        });
    }

    Ok(results)
}

fn detect_scrfd(
    net: &Arc<Mutex<dnn::Net>>,
    config: &DetectorModelConfig,
    output_names: &Vector<String>,
    image: &Mat,
) -> Result<Vec<DetectionResult>> {
    let (det_image, det_scale) =
        prepare_scrfd_input(image, config.input_width, config.input_height)?;
    let blob = dnn::blob_from_image(
        &det_image,
        1.0 / 128.0,
        Size::new(config.input_width, config.input_height),
        Scalar::new(127.5, 127.5, 127.5, 0.0),
        true,
        false,
        CV_32F,
    )?;

    let mut net = net
        .lock()
        .map_err(|e| Error::Detection(format!("Failed to lock SCRFD detector: {}", e)))?;
    net.set_input(&blob, "", 1.0, Scalar::default())?;

    let mut outputs = Vector::<Mat>::new();
    net.forward(&mut outputs, output_names)?;
    if outputs.len() != output_names.len() {
        return Err(Error::Detection(format!(
            "SCRFD output count mismatch: requested {}, got {}",
            output_names.len(),
            outputs.len()
        )));
    }

    let mut proposals = Vec::new();
    let feature_strides = [8usize, 16, 32];
    let input_w = config.input_width as usize;
    let input_h = config.input_height as usize;

    for (output_offset, stride) in feature_strides.into_iter().enumerate() {
        let feature_w = input_w / stride;
        let feature_h = input_h / stride;
        let score_mat = outputs.get(output_offset * 3)?;
        let bbox_mat = outputs.get(output_offset * 3 + 1)?;
        let kps_mat = outputs.get(output_offset * 3 + 2)?;

        let scores = decode_scrfd_scores(&score_mat, feature_h, feature_w)?;
        let num_anchors = scores.len() / (feature_h * feature_w);
        if num_anchors == 0 {
            return Err(Error::Detection(format!(
                "SCRFD produced zero anchors for stride {}",
                stride
            )));
        }

        let bbox_rows = decode_scrfd_rows(&bbox_mat, scores.len(), 4, feature_h, feature_w)?;
        let kps_rows = decode_scrfd_rows(&kps_mat, scores.len(), 10, feature_h, feature_w)?;
        let anchor_centers = anchor_centers(feature_h, feature_w, stride as f32, num_anchors);

        for idx in 0..scores.len() {
            let score = scores[idx];
            if score < config.score_threshold {
                continue;
            }

            let bbox = distance2bbox(anchor_centers[idx], &bbox_rows[idx], stride as f32);
            let kps = distance2kps(anchor_centers[idx], &kps_rows[idx], stride as f32);
            proposals.push(ScrfdProposal {
                score,
                bbox,
                landmarks: kps,
            });
        }
    }

    let keep = nms_indices(&proposals, config.nms_threshold);
    let image_size = image.size()?;

    Ok(keep
        .into_iter()
        .map(|idx| proposals[idx].to_detection_result(det_scale, image_size.width, image_size.height))
        .collect())
}

fn prepare_scrfd_input(image: &Mat, input_width: i32, input_height: i32) -> Result<(Mat, f32)> {
    let size = image.size()?;
    let image_w = size.width as f32;
    let image_h = size.height as f32;
    let scale = ((input_width as f32) / image_w).min((input_height as f32) / image_h);

    let resized_w = (image_w * scale).round() as i32;
    let resized_h = (image_h * scale).round() as i32;

    let mut resized = Mat::default();
    opencv::imgproc::resize(
        image,
        &mut resized,
        Size::new(resized_w, resized_h),
        0.0,
        0.0,
        opencv::imgproc::INTER_LINEAR,
    )?;

    let mut det_image = Mat::zeros(input_height, input_width, CV_8UC3)?.to_mat()?;
    let roi = Rect::new(0, 0, resized_w, resized_h);
    let mut det_roi = Mat::roi_mut(&mut det_image, roi)?;
    resized.copy_to(&mut det_roi)?;

    Ok((det_image, scale))
}

fn decode_scrfd_scores(mat: &Mat, feature_h: usize, feature_w: usize) -> Result<Vec<f32>> {
    let dims = mat_shape(mat)?;
    let data = mat.data_typed::<f32>()?;
    let spatial = feature_h * feature_w;

    if dims.len() == 4 && dims[2] == feature_h && dims[3] == feature_w {
        let num_anchors = dims[1];
        let mut scores = Vec::with_capacity(num_anchors * spatial);
        for cell_idx in 0..spatial {
            for anchor in 0..num_anchors {
                scores.push(data[anchor * spatial + cell_idx]);
            }
        }
        return Ok(scores);
    }

    Ok(data.to_vec())
}

fn decode_scrfd_rows(
    mat: &Mat,
    rows: usize,
    cols: usize,
    feature_h: usize,
    feature_w: usize,
) -> Result<Vec<Vec<f32>>> {
    let dims = mat_shape(mat)?;
    let data = mat.data_typed::<f32>()?;
    if data.len() != rows * cols {
        return Err(Error::Detection(format!(
            "Unexpected SCRFD tensor size: expected {}, got {}",
            rows * cols,
            data.len()
        )));
    }

    if dims.len() == 4 && dims[2] == feature_h && dims[3] == feature_w && dims[1] % cols == 0 {
        let num_anchors = dims[1] / cols;
        let spatial = feature_h * feature_w;
        let mut decoded = Vec::with_capacity(rows);
        for cell_idx in 0..spatial {
            for anchor in 0..num_anchors {
                let mut row = Vec::with_capacity(cols);
                for col in 0..cols {
                    row.push(data[((anchor * cols + col) * spatial) + cell_idx]);
                }
                decoded.push(row);
            }
        }
        return Ok(decoded);
    }

    let mut decoded = Vec::with_capacity(rows);
    for row_idx in 0..rows {
        let start = row_idx * cols;
        decoded.push(data[start..start + cols].to_vec());
    }
    Ok(decoded)
}

fn mat_shape(mat: &Mat) -> Result<Vec<usize>> {
    let dims = mat.dims();
    let mat_size = mat.mat_size();
    let mut shape = Vec::with_capacity(dims as usize);
    for i in 0..dims {
        shape.push(mat_size.get(i)? as usize);
    }
    Ok(shape)
}

fn anchor_centers(feature_h: usize, feature_w: usize, stride: f32, num_anchors: usize) -> Vec<(f32, f32)> {
    let mut centers = Vec::with_capacity(feature_h * feature_w * num_anchors);
    for y in 0..feature_h {
        for x in 0..feature_w {
            let center = (x as f32 * stride, y as f32 * stride);
            for _ in 0..num_anchors {
                centers.push(center);
            }
        }
    }
    centers
}

fn distance2bbox(anchor: (f32, f32), distance: &[f32], stride: f32) -> (f32, f32, f32, f32) {
    let left = distance[0] * stride;
    let top = distance[1] * stride;
    let right = distance[2] * stride;
    let bottom = distance[3] * stride;

    (
        anchor.0 - left,
        anchor.1 - top,
        anchor.0 + right,
        anchor.1 + bottom,
    )
}

fn distance2kps(anchor: (f32, f32), distance: &[f32], stride: f32) -> [(f32, f32); 5] {
    let mut points = [(0.0f32, 0.0f32); 5];
    for idx in 0..5 {
        let x = anchor.0 + distance[idx * 2] * stride;
        let y = anchor.1 + distance[idx * 2 + 1] * stride;
        points[idx] = (x, y);
    }
    points
}

fn nms_indices(proposals: &[ScrfdProposal], iou_threshold: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..proposals.len()).collect();
    order.sort_by(|&a, &b| {
        proposals[b]
            .score
            .partial_cmp(&proposals[a].score)
            .unwrap_or(Ordering::Equal)
    });

    let mut keep = Vec::new();
    while let Some(current) = order.first().copied() {
        keep.push(current);
        order.remove(0);
        order.retain(|&candidate| iou(&proposals[current].bbox, &proposals[candidate].bbox) <= iou_threshold);
    }

    keep
}

fn iou(a: &(f32, f32, f32, f32), b: &(f32, f32, f32, f32)) -> f32 {
    let inter_x1 = a.0.max(b.0);
    let inter_y1 = a.1.max(b.1);
    let inter_x2 = a.2.min(b.2);
    let inter_y2 = a.3.min(b.3);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;
    let area_a = (a.2 - a.0).max(0.0) * (a.3 - a.1).max(0.0);
    let area_b = (b.2 - b.0).max(0.0) * (b.3 - b.1).max(0.0);
    let union = area_a + area_b - inter_area;

    if union <= 0.0 {
        0.0
    } else {
        inter_area / union
    }
}

struct ScrfdProposal {
    score: f32,
    bbox: (f32, f32, f32, f32),
    landmarks: [(f32, f32); 5],
}

impl ScrfdProposal {
    fn to_detection_result(&self, det_scale: f32, image_width: i32, image_height: i32) -> DetectionResult {
        let clamp = |value: f32, max_value: i32| value.clamp(0.0, (max_value - 1).max(0) as f32);
        let x1 = clamp(self.bbox.0 / det_scale, image_width);
        let y1 = clamp(self.bbox.1 / det_scale, image_height);
        let x2 = clamp(self.bbox.2 / det_scale, image_width);
        let y2 = clamp(self.bbox.3 / det_scale, image_height);
        let landmarks = self.landmarks.map(|(x, y)| {
            (
                clamp(x / det_scale, image_width),
                clamp(y / det_scale, image_height),
            )
        });

        DetectionResult {
            bbox: (x1, y1, (x2 - x1).max(1.0), (y2 - y1).max(1.0)),
            confidence: self.score,
            landmarks: FaceLandmarks::from_arcface_order(landmarks),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_to_bbox() {
        let bbox = distance2bbox((16.0, 24.0), &[1.0, 2.0, 3.0, 4.0], 8.0);
        assert_eq!(bbox, (8.0, 8.0, 40.0, 56.0));
    }

    #[test]
    fn test_anchor_centers_repeat_per_anchor() {
        let centers = anchor_centers(1, 2, 8.0, 2);
        assert_eq!(centers, vec![(0.0, 0.0), (0.0, 0.0), (8.0, 0.0), (8.0, 0.0)]);
    }

    #[test]
    fn test_nms_prefers_higher_score() {
        let proposals = vec![
            ScrfdProposal {
                score: 0.9,
                bbox: (0.0, 0.0, 10.0, 10.0),
                landmarks: [(0.0, 0.0); 5],
            },
            ScrfdProposal {
                score: 0.8,
                bbox: (1.0, 1.0, 11.0, 11.0),
                landmarks: [(0.0, 0.0); 5],
            },
        ];

        let keep = nms_indices(&proposals, 0.5);
        assert_eq!(keep, vec![0]);
    }
}
