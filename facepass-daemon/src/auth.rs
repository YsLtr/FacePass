//! Authentication logic

use crate::control::{AuthControl, AuthSession};
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

const MAX_ANTI_SPOOF_ERRORS: u32 = 3;
const MANUAL_INTERRUPT_MESSAGE: &str = "用户手动打断";

/// Perform face authentication
pub async fn authenticate(
    config: &Arc<Config>,
    control: &Arc<AuthControl>,
    request: &AuthRequest,
) -> AuthResponse {
    // Run in blocking task since OpenCV operations are synchronous
    let config = config.clone();
    let control = control.clone();
    let username = request.username.clone();
    let timeout = request.timeout;
    let session = control.register();

    let session_id = session.id();
    let result = tokio::task::spawn_blocking(move || {
        authenticate_sync(&config, &session, &username, timeout)
    })
    .await
    .unwrap_or_else(|e| AuthResponse::failure(format!("Task error: {}", e)));

    control.finish(session_id);
    result
}

fn authenticate_sync(
    config: &Config,
    session: &AuthSession,
    username: &str,
    timeout: u32,
) -> AuthResponse {
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

    let mut anti_spoof = if config.anti_spoof.enabled {
        match AntiSpoofDetector::new(&config.models, &config.anti_spoof) {
            Ok(d) => Some(d),
            Err(e) => {
                warn!(
                    "Anti-spoofing unavailable at startup, will retry during authentication: {}",
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
    let stop_on_valid_frames = config.recognition.stop_on_valid_frames && required_valid_frames > 0;
    let max_frames = config.video.max_frames;
    let enforce_timeout = timeout > 0;
    let enforce_frame_limit = max_frames > 0;

    let start_time = Instant::now();
    let timeout_duration = Duration::from_secs(timeout as u64);

    let mut consecutive_matches = 0u32;
    let mut max_consecutive_matches = 0u32;
    let mut valid_frame_count = 0u32;
    let mut frame_count = 0u32;
    let mut best_match_info: Option<(f64, String)> = None;
    let mut anti_spoof_errors = 0u32;

    info!(
        "Starting face recognition (timeout: {}s, max_frames: {}, valid_frames: {}, consecutive_match_frames: {}, stop_on_valid_frames: {})",
        timeout,
        max_frames,
        required_valid_frames,
        consecutive_match_frames,
        stop_on_valid_frames
    );

    while (!enforce_timeout || start_time.elapsed() < timeout_duration)
        && (!enforce_frame_limit || frame_count < max_frames)
    {
        if session.is_cancelled() {
            return AuthResponse::failure(MANUAL_INTERRUPT_MESSAGE);
        }

        frame_count += 1;

        // Read frame
        let frame = match camera.read_frame() {
            Ok(f) => f,
            Err(_) => continue,
        };

        if session.is_cancelled() {
            return AuthResponse::failure(MANUAL_INTERRUPT_MESSAGE);
        }

        if config.anti_spoof.enabled
            && anti_spoof.is_none()
            && anti_spoof_errors < MAX_ANTI_SPOOF_ERRORS
        {
            match AntiSpoofDetector::new(&config.models, &config.anti_spoof) {
                Ok(detector) => {
                    anti_spoof = Some(detector);
                    warn!("Anti-spoof detector restarted successfully");
                }
                Err(e) => {
                    anti_spoof_errors += 1;
                    warn!(
                        "Anti-spoof restart failed ({}/{}): {}",
                        anti_spoof_errors, MAX_ANTI_SPOOF_ERRORS, e
                    );
                    if anti_spoof_errors >= MAX_ANTI_SPOOF_ERRORS {
                        return AuthResponse::failure("Anti-spoof error");
                    }
                }
            }
        }

        // Detect face
        let faces = match detector.detect_raw(&frame) {
            Ok(f) => f,
            Err(_) => {
                consecutive_matches = 0;
                continue;
            }
        };

        let face_row =
            match select_primary_face(&frame, &faces, config.recognition.valid_crop_scale) {
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
            match detector.check_liveness(&frame, &face_row, config.recognition.valid_crop_scale) {
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
                    anti_spoof_errors += 1;
                    warn!(
                        "Anti-spoof error ({}/{}), restarting detector: {}",
                        anti_spoof_errors, MAX_ANTI_SPOOF_ERRORS, e
                    );
                    anti_spoof = None;
                    if anti_spoof_errors >= MAX_ANTI_SPOOF_ERRORS {
                        return AuthResponse::failure("Anti-spoof error");
                    }
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

    if required_valid_frames > 0 && valid_frame_count < required_valid_frames {
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
