//! Face recognition backends.

use crate::alignment::align_detected_face;
use crate::config::{
    ColorOrder, InputLayout, ModelsConfig, RecognizerKind, RecognizerModelConfig,
    RecognizerPreprocessConfig,
};
use crate::error::{Error, Result};
use crate::models::{DetectionResult, FaceEmbedding};
use opencv::{
    core::{Mat, Ptr, Scalar, Size, Vec3b, Vector, CV_32F, CV_8UC3},
    dnn,
    objdetect::FaceRecognizerSF,
    prelude::*,
};
use std::sync::{Arc, Mutex};

pub struct FaceRecognizer {
    backend: RecognizerBackendImpl,
    kind: RecognizerKind,
    config: RecognizerModelConfig,
}

enum RecognizerBackendImpl {
    SFace {
        recognizer: Arc<Mutex<Ptr<FaceRecognizerSF>>>,
    },
    Onnx {
        net: Arc<Mutex<dnn::Net>>,
    },
}

unsafe impl Send for FaceRecognizer {}
unsafe impl Sync for FaceRecognizer {}

impl FaceRecognizer {
    pub fn new(models: &ModelsConfig) -> Result<Self> {
        match models.active_recognizer {
            RecognizerKind::Sface => Self::build_sface(&models.sface),
            RecognizerKind::Mobilefacenet => {
                Self::build_onnx(RecognizerKind::Mobilefacenet, &models.mobilefacenet)
            }
            RecognizerKind::Ghostfacenet => {
                Self::build_onnx(RecognizerKind::Ghostfacenet, &models.ghostfacenet)
            }
        }
    }

    fn build_sface(config: &RecognizerModelConfig) -> Result<Self> {
        let recognizer = FaceRecognizerSF::create(&config.path, "", 0, 0).map_err(|e| {
            Error::Recognition(format!(
                "Failed to create SFace recognizer from {}: {}",
                config.path, e
            ))
        })?;

        Ok(Self {
            backend: RecognizerBackendImpl::SFace {
                recognizer: Arc::new(Mutex::new(recognizer)),
            },
            kind: RecognizerKind::Sface,
            config: config.clone(),
        })
    }

    fn build_onnx(kind: RecognizerKind, config: &RecognizerModelConfig) -> Result<Self> {
        let net = dnn::read_net_from_onnx(&config.path).map_err(|e| {
            Error::Recognition(format!(
                "Failed to load {:?} recognizer from {}: {}",
                kind, config.path, e
            ))
        })?;

        Ok(Self {
            backend: RecognizerBackendImpl::Onnx {
                net: Arc::new(Mutex::new(net)),
            },
            kind,
            config: config.clone(),
        })
    }

    pub fn kind(&self) -> RecognizerKind {
        self.kind
    }

    pub fn model_id(&self) -> &str {
        &self.config.model_id
    }

    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    pub fn input_size(&self) -> Size {
        Size::new(
            self.config.preprocess.input_width,
            self.config.preprocess.input_height,
        )
    }

    pub fn extract_embedding_from_frame(
        &self,
        frame: &Mat,
        detection: &DetectionResult,
    ) -> Result<FaceEmbedding> {
        let aligned = align_detected_face(frame, detection, self.input_size())?;
        self.extract_embedding(&aligned)
    }

    pub fn extract_embedding(&self, aligned_face: &Mat) -> Result<FaceEmbedding> {
        let feature = match &self.backend {
            RecognizerBackendImpl::SFace { recognizer } => {
                let mut recognizer = recognizer
                    .lock()
                    .map_err(|e| Error::Recognition(format!("Failed to lock recognizer: {}", e)))?;
                let mut feature = Mat::default();
                recognizer.feature(aligned_face, &mut feature)?;
                mat_to_vec(&feature)?
            }
            RecognizerBackendImpl::Onnx { net } => {
                let blob = image_to_tensor(aligned_face, &self.config.preprocess)?;
                let mut net = net
                    .lock()
                    .map_err(|e| Error::Recognition(format!("Failed to lock recognizer: {}", e)))?;
                net.set_input(&blob, "", 1.0, Scalar::default())?;
                let out_names = net.get_unconnected_out_layers_names()?;
                let mut outputs = Vector::<Mat>::new();
                net.forward(&mut outputs, &out_names)?;
                if outputs.is_empty() {
                    return Err(Error::Recognition("Empty recognizer output".to_string()));
                }
                mat_to_vec(&outputs.get(0)?)?
            }
        };

        if feature.len() != self.config.embedding_dim {
            return Err(Error::Recognition(format!(
                "Recognizer '{}' returned {} dims, expected {}",
                self.config.model_id,
                feature.len(),
                self.config.embedding_dim
            )));
        }

        let feature = if self.config.preprocess.l2_normalize {
            l2_normalize(feature)
        } else {
            feature
        };

        let embedding = FaceEmbedding::new(self.config.model_id.clone(), feature);
        if !embedding.is_valid() {
            return Err(Error::Recognition(format!(
                "Recognizer '{}' produced invalid embedding metadata",
                self.config.model_id
            )));
        }

        Ok(embedding)
    }
}

pub fn mat_to_vec(mat: &Mat) -> Result<Vec<f32>> {
    Ok(mat.data_typed::<f32>()?.to_vec())
}

fn image_to_tensor(image: &Mat, config: &RecognizerPreprocessConfig) -> Result<Mat> {
    let mut resized = Mat::default();
    opencv::imgproc::resize(
        image,
        &mut resized,
        Size::new(config.input_width, config.input_height),
        0.0,
        0.0,
        opencv::imgproc::INTER_LINEAR,
    )?;

    let reordered = match config.color_order {
        ColorOrder::Bgr => resized,
        ColorOrder::Rgb => {
            let mut rgb = Mat::default();
            opencv::imgproc::cvt_color_def(&resized, &mut rgb, opencv::imgproc::COLOR_BGR2RGB)?;
            rgb
        }
    };

    if reordered.typ() != CV_8UC3 {
        return Err(Error::Recognition(format!(
            "Recognizer preprocess expected CV_8UC3 image, got OpenCV type {}",
            reordered.typ()
        )));
    }

    let width = config.input_width as usize;
    let height = config.input_height as usize;
    let channels = 3usize;
    let mut tensor = vec![0.0f32; width * height * channels];

    for y in 0..height {
        for x in 0..width {
            let pixel = reordered.at_2d::<Vec3b>(y as i32, x as i32)?;
            for c in 0..channels {
                let normalized =
                    (pixel[c] as f32 - config.mean[c]) / config.std[c].max(f32::EPSILON);
                let dst_idx = match config.input_layout {
                    InputLayout::Nchw => (c * height * width) + (y * width) + x,
                    InputLayout::Nhwc => ((y * width + x) * channels) + c,
                };
                tensor[dst_idx] = normalized;
            }
        }
    }

    let dims = match config.input_layout {
        InputLayout::Nchw => [1, 3, config.input_height, config.input_width],
        InputLayout::Nhwc => [1, config.input_height, config.input_width, 3],
    };
    let mut blob = Mat::new_nd_with_default(&dims, CV_32F, Scalar::default())?;
    let blob_data = blob.data_typed_mut::<f32>()?;
    blob_data.copy_from_slice(&tensor);
    Ok(blob)
}

fn l2_normalize(mut feature: Vec<f32>) -> Vec<f32> {
    let norm = feature
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        .sqrt();

    if norm <= 1e-12 {
        return feature;
    }

    for value in &mut feature {
        *value = (*value as f64 / norm) as f32;
    }

    feature
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Mat, Scalar};

    #[test]
    fn test_l2_normalize_produces_unit_vector() {
        let normalized = l2_normalize(vec![3.0, 4.0]);
        let norm = normalized
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_image_to_tensor_accepts_cv_8uc3_input() {
        let image = Mat::new_rows_cols_with_default(
            112,
            112,
            CV_8UC3,
            Scalar::new(10.0, 20.0, 30.0, 0.0),
        )
        .unwrap();
        let config = RecognizerPreprocessConfig {
            input_width: 112,
            input_height: 112,
            input_layout: InputLayout::Nhwc,
            color_order: ColorOrder::Rgb,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            l2_normalize: false,
        };

        let blob = image_to_tensor(&image, &config).unwrap();
        let values = blob.data_typed::<f32>().unwrap();

        assert_eq!(values.len(), 112 * 112 * 3);
        assert_eq!(values[0], 30.0);
        assert_eq!(values[1], 20.0);
        assert_eq!(values[2], 10.0);
    }
}
