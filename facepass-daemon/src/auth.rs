//! Authentication logic

use opencv::prelude::*;
use facepass_core::{
    camera::Camera,
    config::Config,
    detection::FaceDetector,
    matching::find_best_match,
    models::{AuthRequest, AuthResponse},
    recognition::{mat_to_vec, FaceRecognizer},
    security::check_security,
    storage::FaceStorage,
};
use log::{debug, info, warn};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Perform face authentication
pub async fn authenticate(config: &Arc<Config>, request: &AuthRequest) -> AuthResponse {
    // Run in blocking task since OpenCV operations are synchronous
    let config = config.clone();
    let username = request.username.clone();
    let timeout = request.timeout;

    tokio::task::spawn_blocking(move || authenticate_sync(&config, &username, timeout))
        .await
        .unwrap_or_else(|e| AuthResponse::failure(format!("Task error: {}", e)))
}

fn authenticate_sync(config: &Config, username: &str, timeout: u32) -> AuthResponse {
    // Security checks
    if let Err(e) = check_security(&config.security) {
        return AuthResponse::failure(e.to_string());
    }

    // Load user's face data
    let storage = match FaceStorage::new(&config.storage.data_dir) {
        Ok(s) => s,
        Err(e) => return AuthResponse::failure(format!("Storage error: {}", e)),
    };

    if !storage.has_faces(username) {
        return AuthResponse::no_face_data();
    }

    let face_data = match storage.load_face_data(username) {
        Ok(data) => data,
        Err(e) => return AuthResponse::failure(format!("Failed to load faces: {}", e)),
    };

    if face_data.is_empty() {
        return AuthResponse::no_face_data();
    }

    debug!("Loaded {} face(s) for user {}", face_data.len(), username);

    // Initialize components
    let camera = match Camera::open(&config.video) {
        Ok(c) => c,
        Err(e) => return AuthResponse::failure(format!("Camera error: {}", e)),
    };

    let detector = match FaceDetector::new(&config.models.yunet_path, &config.detection) {
        Ok(d) => d,
        Err(e) => return AuthResponse::failure(format!("Detector error: {}", e)),
    };

    let recognizer = match FaceRecognizer::new(&config.models.sface_path, &config.recognition) {
        Ok(r) => r,
        Err(e) => return AuthResponse::failure(format!("Recognizer error: {}", e)),
    };

    let threshold = config.recognition.similarity_threshold;
    let required_matches = config.recognition.required_matches;
    let max_frames = config.video.max_frames;

    let start_time = Instant::now();
    let timeout_duration = Duration::from_secs(timeout as u64);

    let mut consecutive_matches = 0u32;
    let mut frame_count = 0u32;
    let mut best_match_info: Option<(f64, String)> = None;

    info!("Starting face recognition (timeout: {}s)", timeout);

    while start_time.elapsed() < timeout_duration && frame_count < max_frames {
        frame_count += 1;

        // Read frame
        let frame = match camera.read_frame() {
            Ok(f) => f,
            Err(_) => continue,
        };

        // Detect face
        let faces = match detector.detect_raw(&frame) {
            Ok(f) => f,
            Err(_) => {
                consecutive_matches = 0;
                continue;
            }
        };

        // Get first face
        let face_row = match faces.row(0) {
            Ok(r) => match r.try_clone() {
                Ok(m) => m,
                Err(_) => continue,
            },
            Err(_) => continue,
        };

        // Align and extract feature
        let aligned = match recognizer.align_crop(&frame, &face_row) {
            Ok(a) => a,
            Err(_) => continue,
        };

        let feature_mat = match recognizer.extract_feature(&aligned) {
            Ok(f) => f,
            Err(_) => continue,
        };

        let feature = match mat_to_vec(&feature_mat) {
            Ok(f) => f,
            Err(_) => continue,
        };

        // Match against stored faces
        match find_best_match(&feature, &face_data, threshold) {
            Ok(Some(m)) => {
                consecutive_matches += 1;
                debug!(
                    "Match: {} (similarity: {:.2}%, consecutive: {})",
                    m.face_data.label,
                    m.similarity * 100.0,
                    consecutive_matches
                );

                // Track best match
                if best_match_info.is_none()
                    || m.similarity > best_match_info.as_ref().unwrap().0
                {
                    best_match_info = Some((m.similarity, m.face_data.label.clone()));
                }

                // Check if we have enough consecutive matches
                if consecutive_matches >= required_matches {
                    let (confidence, label) = best_match_info.unwrap();
                    return AuthResponse::success(confidence, label);
                }
            }
            Ok(None) => {
                consecutive_matches = 0;
            }
            Err(e) => {
                warn!("Match error: {}", e);
                consecutive_matches = 0;
            }
        }
    }

    // Timeout
    AuthResponse::timeout()
}
