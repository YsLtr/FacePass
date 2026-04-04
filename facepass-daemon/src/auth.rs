//! Authentication logic

use crate::control::{AuthControl, AuthSession};
use crate::runtime::{snapshot, SharedRuntimeConfig};
use facepass_core::{
    camera::Camera,
    config::Config,
    matching::find_best_match,
    models::{AuthRequest, AuthResponse},
    pipeline::FaceRuntime,
    security::check_security,
    storage::FaceStorage,
};
use log::{debug, info, warn};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MANUAL_INTERRUPT_MESSAGE: &str = "用户手动打断";

pub async fn authenticate(
    runtime_state: &SharedRuntimeConfig,
    control: &Arc<AuthControl>,
    request: &AuthRequest,
) -> AuthResponse {
    let config = snapshot(runtime_state).config;
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
    if let Err(e) = check_security(&config.security) {
        return AuthResponse::failure(e.to_string());
    }

    let storage = match FaceStorage::new(&config.storage.data_dir) {
        Ok(storage) => storage,
        Err(e) => return AuthResponse::failure(format!("Storage error: {}", e)),
    };

    let default_group = match storage.get_default_group(username) {
        Ok(group) => group,
        Err(_) => return AuthResponse::no_face_data(),
    };

    let runtime = match FaceRuntime::new(config) {
        Ok(runtime) => runtime,
        Err(e) => return AuthResponse::failure(format!("Backend initialization error: {}", e)),
    };

    let face_data = match storage.load_face_data_in_group_for_model(
        username,
        &default_group.id,
        runtime.model_id(),
        runtime.embedding_dim(),
    ) {
        Ok(data) => data,
        Err(e) => return AuthResponse::failure(format!("Failed to load faces: {}", e)),
    };

    if face_data.is_empty() {
        return AuthResponse::no_face_data();
    }

    debug!(
        "Loaded {} face(s) for user {} from group {} using model {}",
        face_data.len(),
        username,
        default_group.name,
        runtime.model_id()
    );

    let camera = match Camera::open(&config.video) {
        Ok(camera) => camera,
        Err(e) => return AuthResponse::failure(format!("Camera error: {}", e)),
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

    info!(
        "Starting face recognition (detector: {}, recognizer: {}, timeout: {}s, max_frames: {}, valid_frames: {}, consecutive_match_frames: {}, stop_on_valid_frames: {})",
        config.models.active_detector.as_str(),
        config.models.active_recognizer.as_str(),
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
        let frame = match camera.read_frame() {
            Ok(frame) => frame,
            Err(_) => continue,
        };

        if session.is_cancelled() {
            return AuthResponse::failure(MANUAL_INTERRUPT_MESSAGE);
        }

        let detections = match runtime.detect_faces(&frame) {
            Ok(detections) if !detections.is_empty() => detections,
            _ => {
                consecutive_matches = 0;
                continue;
            }
        };

        let detection = match runtime.select_primary_face(&frame, &detections) {
            Ok(detection) => detection,
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

        match runtime.check_liveness(&frame, &detection) {
            Ok(Some(score)) if score >= config.anti_spoof.threshold => {
                debug!("Liveness check passed (score: {:.3})", score);
            }
            Ok(Some(score)) => {
                debug!("Liveness check failed (score: {:.3})", score);
                consecutive_matches = 0;
                continue;
            }
            Ok(None) => {}
            Err(facepass_core::Error::InvalidFace(reason)) => {
                debug!("Skipping invalid face before liveness: {}", reason);
                consecutive_matches = 0;
                continue;
            }
            Err(e) => {
                warn!("Anti-spoof error: {}", e);
                consecutive_matches = 0;
                continue;
            }
        }

        let embedding = match runtime.extract_embedding(&frame, &detection) {
            Ok(embedding) => embedding,
            Err(e) => {
                warn!("Embedding extraction error: {}", e);
                consecutive_matches = 0;
                continue;
            }
        };

        match find_best_match(&embedding, &face_data, threshold) {
            Ok(Some(m)) if m.passed_threshold => {
                consecutive_matches += 1;
                max_consecutive_matches = max_consecutive_matches.max(consecutive_matches);
                debug!(
                    "Match: {} (similarity: {:.2}%, consecutive: {}, valid_frames: {})",
                    m.face_data.label,
                    m.similarity * 100.0,
                    consecutive_matches,
                    valid_frame_count
                );

                if best_match_info
                    .as_ref()
                    .map(|(score, _)| m.similarity > *score)
                    .unwrap_or(true)
                {
                    best_match_info = Some((m.similarity, m.face_data.label.clone()));
                }
            }
            Ok(Some(_)) | Ok(None) => {
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
