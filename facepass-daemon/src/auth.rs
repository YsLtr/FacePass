//! Authentication logic

use facepass_core::{
    anti_spoofing::AntiSpoofDetector,
    camera::Camera,
    config::Config,
    detection::FaceDetector,
    face_validation::select_primary_face,
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

    let anti_spoof = if config.anti_spoof.enabled {
        match AntiSpoofDetector::new(&config.models, &config.anti_spoof) {
            Ok(d) => Some(d),
            Err(e) => {
                warn!(
                    "Anti-spoofing unavailable, falling back to face recognition only: {}",
                    e
                );
                None
            }
        }
    } else {
        None
    };

    let threshold = config.recognition.similarity_threshold;
    let consecutive_match_frames = config.recognition.consecutive_match_frames;
    let required_valid_frames = config.recognition.valid_frames;
    let stop_on_valid_frames = config.recognition.stop_on_valid_frames;
    let max_frames = config.video.max_frames;

    let start_time = Instant::now();
    let timeout_duration = Duration::from_secs(timeout as u64);

    let mut consecutive_matches = 0u32;
    let mut max_consecutive_matches = 0u32;
    let mut valid_frame_count = 0u32;
    let mut frame_count = 0u32;
    let mut best_match_info: Option<(f64, String)> = None;

    info!(
        "Starting face recognition (timeout: {}s, max_frames: {}, valid_frames: {}, consecutive_match_frames: {}, stop_on_valid_frames: {})",
        timeout,
        max_frames,
        required_valid_frames,
        consecutive_match_frames,
        stop_on_valid_frames
    );

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

        let face_row = match select_primary_face(
            &frame,
            &faces,
            config.recognition.valid_crop_scale,
        ) {
            Ok(face_row) => face_row,
            Err(facepass_core::Error::InvalidFace(reason)) => {
                debug!("Skipping invalid face: {}", reason);
                consecutive_matches = 0;
                continue;
            }
            Err(_) => {
                consecutive_matches = 0;
                continue;
            }
        };

        valid_frame_count += 1;

        // Anti-spoofing check
        if let Some(ref detector) = anti_spoof {
            match detector.check_liveness(
                &frame,
                &face_row,
                config.recognition.valid_crop_scale,
            ) {
                Ok(score) if score >= config.anti_spoof.threshold => {
                    debug!("Liveness check passed (score: {:.3})", score);
                }
                Ok(score) => {
                    debug!("Liveness check failed (score: {:.3})", score);
                    consecutive_matches = 0;
                    continue;
                }
                Err(facepass_core::Error::InvalidFace(reason)) => {
                    debug!("Skipping invalid face before liveness: {}", reason);
                    consecutive_matches = 0;
                    continue;
                }
                Err(e) => {
                    warn!("Liveness check error: {}", e);
                    consecutive_matches = 0;
                    continue;
                }
            }
        }

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
                max_consecutive_matches = max_consecutive_matches.max(consecutive_matches);
                debug!(
                    "Match: {} (similarity: {:.2}%, consecutive: {}, valid_frames: {})",
                    m.face_data.label,
                    m.similarity * 100.0,
                    consecutive_matches,
                    valid_frame_count
                );

                // Track best match
                if best_match_info.is_none() || m.similarity > best_match_info.as_ref().unwrap().0 {
                    best_match_info = Some((m.similarity, m.face_data.label.clone()));
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

        if stop_on_valid_frames && valid_frame_count >= required_valid_frames {
            if max_consecutive_matches >= consecutive_match_frames {
                if let Some((confidence, label)) = best_match_info {
                    return AuthResponse::success(confidence, label);
                }
            }

            return AuthResponse::failure(format!(
                "Recognition failed after {} valid frame(s); required {} consecutive matched frame(s)",
                valid_frame_count, consecutive_match_frames
            ));
        }
    }

    if valid_frame_count < required_valid_frames {
        return AuthResponse::failure(format!(
            "Recognition ended with only {} valid frame(s); required at least {}",
            valid_frame_count, required_valid_frames
        ));
    }

    if max_consecutive_matches >= consecutive_match_frames {
        if let Some((confidence, label)) = best_match_info {
            return AuthResponse::success(confidence, label);
        }
    }

    AuthResponse::failure(format!(
        "Recognition ended after {} valid frame(s); required {} consecutive matched frame(s)",
        valid_frame_count, consecutive_match_frames
    ))
}
