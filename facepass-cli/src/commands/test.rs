//! Test face recognition command

use super::{
    ensure_user_access, resolve_group_for_read, resolve_username, CommandInput, CommandKey,
};
use anyhow::{anyhow, Result};
use facepass_core::{
    camera::Camera,
    config::Config,
    matching::find_best_match,
    models::{DetectionResult, FaceRecord},
    pipeline::FaceRuntime,
    storage::FaceStorage,
};
use opencv::{
    core::{Point, Rect, Scalar},
    highgui, imgproc,
    prelude::*,
};
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::time::{Duration, Instant};

#[derive(Default)]
struct TestStats {
    total_frames: u32,
    frame_errors: u32,
    no_face_frames: u32,
    detected_face_frames: u32,
    invalid_face_frames: u32,
    valid_frames: u32,
    matched_frames: u32,
    unmatched_frames: u32,
    spoof_frames: u32,
    max_consecutive_matches: u32,
}

impl TestStats {
    fn valid_frame_rate(&self) -> f64 {
        if self.total_frames == 0 {
            0.0
        } else {
            (self.valid_frames as f64 / self.total_frames as f64) * 100.0
        }
    }

    fn match_success_rate(&self) -> f64 {
        if self.valid_frames == 0 {
            0.0
        } else {
            (self.matched_frames as f64 / self.valid_frames as f64) * 100.0
        }
    }
}

pub fn run(
    config_path: &str,
    user: Option<String>,
    group: Option<String>,
    face: Vec<String>,
    frames_override: Option<u32>,
    debug: bool,
    view: bool,
) -> Result<()> {
    const WINDOW_NAME: &str = "FacePass Test";

    let resolved = Config::load_with_fallback_and_source(config_path)?;
    let config = resolved.config;
    let config_source = resolved
        .source
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "built-in defaults".to_string());
    let active_preset = resolved.active_preset;

    let storage = FaceStorage::new(&config.storage.data_dir)?;
    let username = resolve_username(&storage, user.as_deref())?;
    ensure_user_access(&username, "test faces for other users")?;
    let group = resolve_group_for_read(&storage, &username, group.as_deref())?;
    let frames = frames_override.unwrap_or(config.video.max_frames);

    if debug {
        println!("Testing face recognition for user: {}", username);
        println!("Testing face group: {} ({})", group.name, group.id);
    }

    let runtime = FaceRuntime::new(&config)?;
    let active_model_id = runtime.model_id().to_string();
    let candidate_faces = if face.is_empty() {
        storage.load_faces_in_group(&username, &group.id)?
    } else {
        storage.resolve_faces_in_group(&username, &group.id, &face)?
    };
    let candidate_faces: Vec<FaceRecord> = candidate_faces
        .into_iter()
        .filter(|record| {
            record.data.model_id == runtime.model_id()
                && record.data.embedding_dim == runtime.embedding_dim()
        })
        .collect();
    let face_data: Vec<_> = candidate_faces
        .iter()
        .map(|record| record.data.clone())
        .collect();

    if face_data.is_empty() {
        let selector_scope = if face.is_empty() {
            format!(
                "No faces registered for user '{}' in group '{}' for active model '{}'.",
                username,
                group.name,
                active_model_id
            )
        } else {
            format!(
                "Selected faces in group '{}' do not match active model '{}'.",
                group.name,
                active_model_id
            )
        };
        return Err(anyhow!("{selector_scope} Use 'facepass add' with the current backend first."));
    }

    println!(
        "Loaded {} registered face(s) from group '{}' for model {}",
        face_data.len(),
        group.name,
        active_model_id
    );
    println!(
        "Match candidates: {}",
        summarize_face_candidates(&candidate_faces)
    );
    println!("Config source: {}", colorize(&config_source, COLOR_CYAN));
    println!("Active preset: {}", colorize(&active_preset, COLOR_CYAN));
    println!(
        "Detector: {} | Recognizer: {}",
        config.models.active_detector.as_str(),
        config.models.active_recognizer.as_str()
    );
    println!("Initializing camera...\n");

    let camera = Camera::open(&config.video)?;
    let actual_width = camera.frame_width().ok();
    let actual_height = camera.frame_height().ok();

    if config.anti_spoof.enabled {
        println!(
            "Anti-spoofing enabled (threshold: {:.2}, mode: {})",
            config.anti_spoof.threshold,
            config.anti_spoof.mode.as_str()
        );
    } else {
        println!("Anti-spoofing disabled");
    }

    println!("Please look at the camera...");
    if let (Some(width), Some(height)) = (actual_width, actual_height) {
        println!(
            "Camera resolution: actual {}x{}, requested {}x{}",
            width, height, config.video.frame_width, config.video.frame_height
        );
    }
    let enforce_frame_limit = !debug && frames > 0;
    let enforce_timeout = !debug && config.video.timeout > 0;
    println!(
        "Frame limit: {}{}",
        if frames == 0 {
            "unlimited".to_string()
        } else {
            frames.to_string()
        },
        if debug { " (ignored in debug mode)" } else { "" }
    );
    println!(
        "Timeout: {}{}",
        if config.video.timeout == 0 {
            "unlimited".to_string()
        } else {
            format!("{}s", config.video.timeout)
        },
        if debug { " (ignored in debug mode)" } else { "" }
    );
    if view {
        println!("Press Enter/Esc in the window, or q in the terminal.\n");
    } else if debug {
        println!("Press q to stop.\n");
    } else {
        println!("Press Ctrl+C to stop.\n");
    }

    let mut stats = TestStats::default();
    let mut consecutive_matches = 0u32;
    let threshold = config.recognition.similarity_threshold;
    let required_valid_frames = config.recognition.valid_frames;
    let consecutive_match_frames = config.recognition.consecutive_match_frames;
    let stop_on_valid_frames =
        config.recognition.stop_on_valid_frames && !debug && required_valid_frames > 0;
    let timeout_duration = Duration::from_secs(config.video.timeout as u64);
    let started_at = Instant::now();
    let mut stop_reason = "user_stopped".to_string();
    let mut last_status_width = 0usize;
    let mut user_requested_stop = false;
    let mut command_input = if view || debug {
        Some(CommandInput::capture_single_keys()?)
    } else {
        None
    };

    let loop_result = (|| -> Result<()> {
        let mut frame_idx = 0u32;
        loop {
            loop {
                let command = match command_input.as_ref() {
                    Some(command_input) => command_input.poll_key()?,
                    None => None,
                };

                let Some(command) = command else {
                    break;
                };

                if matches!(command, CommandKey::Quit) {
                    user_requested_stop = true;
                    break;
                }
            }
            if user_requested_stop {
                stop_reason = "user_stopped".to_string();
                break;
            }

            if enforce_timeout && started_at.elapsed() >= timeout_duration {
                stop_reason = "timeout_reached".to_string();
                break;
            }

            if enforce_frame_limit && frame_idx >= frames {
                stop_reason = "frame_limit_reached".to_string();
                break;
            }

            frame_idx += 1;
            stats.total_frames += 1;

            let frame = match camera.read_frame() {
                Ok(frame) => frame,
                Err(e) => {
                    stats.frame_errors += 1;
                    if debug {
                        print_status_line(
                            &mut last_status_width,
                            &format!("Frame error #{}: {}", stats.frame_errors, shorten_error(&e.to_string())),
                        )?;
                    }
                    continue;
                }
            };

            let detections = match runtime.detect_faces(&frame) {
                Ok(detections) if !detections.is_empty() => detections,
                _ => {
                    stats.no_face_frames += 1;
                    print_status_line(
                        &mut last_status_width,
                        &format!(
                            "Searching for face... {}",
                            format_frame_progress(frame_idx, frames)
                        ),
                    )?;
                    if view {
                        let mut display = frame.try_clone()?;
                        draw_text(
                            &mut display,
                            0,
                            "Searching for face...",
                            Scalar::new(0.0, 0.0, 255.0, 0.0),
                        )?;
                        highgui::imshow(WINDOW_NAME, &display)?;
                        if should_end(highgui::wait_key(1)?) {
                            stop_reason = "user_stopped".to_string();
                            break;
                        }
                    }
                    continue;
                }
            };
            stats.detected_face_frames += 1;

            let detection = match runtime.select_primary_face(&frame, &detections) {
                Ok(detection) => detection,
                Err(facepass_core::Error::InvalidFace(reason)) => {
                    stats.invalid_face_frames += 1;
                    consecutive_matches = 0;
                    print_status_line(
                        &mut last_status_width,
                        &format!(
                            "Invalid face ({}) {}",
                            reason,
                            format_frame_progress(frame_idx, frames)
                        ),
                    )?;
                    if view {
                        let mut display = frame.try_clone()?;
                        draw_faces(&mut display, &detections, None, None)?;
                        draw_required_crops(&mut display, &detections, runtime.valid_crop_scale())?;
                        draw_text(
                            &mut display,
                            0,
                            &format!("Invalid face: {}", reason),
                            Scalar::new(0.0, 0.0, 255.0, 0.0),
                        )?;
                        highgui::imshow(WINDOW_NAME, &display)?;
                        if should_end(highgui::wait_key(1)?) {
                            stop_reason = "user_stopped".to_string();
                            break;
                        }
                    }
                    continue;
                }
                Err(_) => {
                    consecutive_matches = 0;
                    continue;
                }
            };

            stats.valid_frames += 1;
            let confidence = detection.confidence;
            let mut liveness_score: Option<f32> = None;
            let mut liveness_status = "disabled";
            let mut liveness_allowed = true;

            match runtime.check_liveness(&frame, &detection) {
                Ok(Some(score)) if score >= config.anti_spoof.threshold => {
                    liveness_score = Some(score);
                    liveness_status = "pass";
                }
                Ok(Some(score)) => {
                    stats.spoof_frames += 1;
                    liveness_score = Some(score);
                    liveness_status = "spoof";
                    liveness_allowed = false;
                    consecutive_matches = 0;
                    print_status_line(
                        &mut last_status_width,
                        &format!(
                            "Spoof detected #{} (score: {:.3}) {}",
                            stats.spoof_frames,
                            score,
                            format_frame_progress(frame_idx, frames)
                        ),
                    )?;
                }
                Ok(None) => {}
                Err(facepass_core::Error::InvalidFace(reason)) => {
                    liveness_status = "invalid";
                    liveness_allowed = false;
                    stats.invalid_face_frames += 1;
                    consecutive_matches = 0;
                    print_status_line(
                        &mut last_status_width,
                        &format!(
                            "Invalid face ({}) {}",
                            reason,
                            format_frame_progress(frame_idx, frames)
                        ),
                    )?;
                    if view {
                        let mut display = frame.try_clone()?;
                        draw_faces(
                            &mut display,
                            &detections,
                            Some(&detection),
                            Some(liveness_box_color(liveness_status)),
                        )?;
                        draw_required_crops(&mut display, &detections, runtime.valid_crop_scale())?;
                        draw_text(
                            &mut display,
                            0,
                            &format!("Invalid face: {}", reason),
                            Scalar::new(0.0, 0.0, 255.0, 0.0),
                        )?;
                        highgui::imshow(WINDOW_NAME, &display)?;
                        if should_end(highgui::wait_key(1)?) {
                            stop_reason = "user_stopped".to_string();
                            break;
                        }
                    }
                }
                Err(e) => {
                    consecutive_matches = 0;
                    print_status_line(
                        &mut last_status_width,
                        &format!(
                            "Anti-spoof error: {} {}",
                            shorten_error(&e.to_string()),
                            format_frame_progress(frame_idx, frames)
                        ),
                    )?;
                    continue;
                }
            }

            if !liveness_allowed && liveness_status == "invalid" {
                continue;
            }

            let mut match_label: Option<String> = None;
            let mut match_score: Option<f64> = None;

            if liveness_allowed {
                let embedding = match runtime.extract_embedding(&frame, &detection) {
                    Ok(embedding) => embedding,
                    Err(e) => {
                        consecutive_matches = 0;
                        print_status_line(
                            &mut last_status_width,
                            &format!(
                                "Embedding error: {} {}",
                                shorten_error(&e.to_string()),
                                format_frame_progress(frame_idx, frames)
                            ),
                        )?;
                        continue;
                    }
                };

                match find_best_match(&embedding, &face_data, threshold) {
                    Ok(Some(m)) => {
                        match_label = Some(m.face_data.label.clone());
                        match_score = Some(m.similarity);
                        if m.passed_threshold {
                            stats.matched_frames += 1;
                            consecutive_matches += 1;
                            stats.max_consecutive_matches =
                                stats.max_consecutive_matches.max(consecutive_matches);
                            print_status_line(
                                &mut last_status_width,
                                &format!(
                                    "Match #{}: {} sim:{:.2}% det:{:.1}% live:{} {}",
                                    stats.matched_frames,
                                    m.face_data.label,
                                    m.similarity * 100.0,
                                    confidence * 100.0,
                                    format_liveness_text(liveness_status, liveness_score),
                                    format_frame_progress(frame_idx, frames)
                                ),
                            )?;
                        } else {
                            stats.unmatched_frames += 1;
                            consecutive_matches = 0;
                            print_status_line(
                                &mut last_status_width,
                                &format!(
                                    "No match: best={} sim:{:.2}% det:{:.1}% live:{} {}",
                                    m.face_data.label,
                                    m.similarity * 100.0,
                                    confidence * 100.0,
                                    format_liveness_text(liveness_status, liveness_score),
                                    format_frame_progress(frame_idx, frames)
                                ),
                            )?;
                        }
                    }
                    Ok(None) => {
                        stats.unmatched_frames += 1;
                        consecutive_matches = 0;
                    }
                    Err(e) => {
                        consecutive_matches = 0;
                        print_status_line(
                            &mut last_status_width,
                            &format!(
                                "Match error: {} {}",
                                shorten_error(&e.to_string()),
                                format_frame_progress(frame_idx, frames)
                            ),
                        )?;
                    }
                }
            }

            if view {
                let mut display = frame.try_clone()?;
                draw_faces(
                    &mut display,
                    &detections,
                    Some(&detection),
                    Some(liveness_box_color(liveness_status)),
                )?;
                draw_required_crops(&mut display, &detections, runtime.valid_crop_scale())?;
                draw_text(
                    &mut display,
                    0,
                    &format!("Detection: {:.1}%", confidence * 100.0),
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                )?;

                let live_text = match (liveness_status, liveness_score) {
                    ("pass", Some(score)) => format!("Anti-spoof: PASS ({:.3})", score),
                    ("spoof", Some(score)) => format!("Anti-spoof: SPOOF ({:.3})", score),
                    ("invalid", _) => "Anti-spoof: INVALID FACE".to_string(),
                    ("error", _) => "Anti-spoof: ERROR".to_string(),
                    _ => "Anti-spoof: disabled".to_string(),
                };
                let live_color = match liveness_status {
                    "pass" => Scalar::new(0.0, 255.0, 0.0, 0.0),
                    "spoof" | "invalid" | "error" => Scalar::new(0.0, 0.0, 255.0, 0.0),
                    _ => Scalar::new(200.0, 200.0, 200.0, 0.0),
                };
                draw_text(&mut display, 1, &live_text, live_color)?;

                if let (Some(label), Some(score)) = (match_label.as_deref(), match_score) {
                    draw_text(
                        &mut display,
                        2,
                        &format!("Best: {} ({:.2}%)", label, score * 100.0),
                        if score >= threshold {
                            Scalar::new(0.0, 255.0, 0.0, 0.0)
                        } else {
                            Scalar::new(0.0, 255.0, 255.0, 0.0)
                        },
                    )?;
                }

                draw_text(
                    &mut display,
                    3,
                    &format!(
                        "Valid: {}  Match: {}  Streak: {}  Spoof: {}",
                        stats.valid_frames,
                        stats.matched_frames,
                        stats.max_consecutive_matches,
                        stats.spoof_frames
                    ),
                    Scalar::new(200.0, 200.0, 200.0, 0.0),
                )?;

                highgui::imshow(WINDOW_NAME, &display)?;
                if should_end(highgui::wait_key(1)?) {
                    stop_reason = "user_stopped".to_string();
                    break;
                }
            }

            if stop_on_valid_frames && stats.valid_frames >= required_valid_frames {
                stop_reason = "valid_frame_threshold_reached".to_string();
                break;
            }
        }

        Ok(())
    })();

    clear_status_line(&mut last_status_width)?;
    drop(command_input.take());
    drop(runtime);
    drop(camera);
    close_view_window(WINDOW_NAME, view)?;
    loop_result?;

    let valid_frame_threshold_met = stats.valid_frames >= required_valid_frames;
    let consecutive_threshold_met = stats.max_consecutive_matches >= consecutive_match_frames;
    let result_valid =
        (required_valid_frames == 0 || valid_frame_threshold_met) && consecutive_threshold_met;

    println!();
    println!("{}", colorize("Test complete!", COLOR_CYAN_BOLD));
    println!(
        "  Result validity: {}",
        colorize(
            if result_valid { "valid" } else { "invalid" },
            if result_valid {
                COLOR_GREEN_BOLD
            } else {
                COLOR_RED_BOLD
            }
        )
    );
    println!(
        "  Match success rate: {}",
        colorize(
            &format!("{:.1}%", stats.match_success_rate()),
            rate_color(stats.match_success_rate())
        )
    );
    println!(
        "  Valid frame rate: {}",
        colorize(
            &format!("{:.1}%", stats.valid_frame_rate()),
            rate_color(stats.valid_frame_rate())
        )
    );
    println!(
        "  Stop reason: {}",
        colorize(&stop_reason, stop_reason_color(&stop_reason))
    );
    println!("  Config source: {}", colorize(&config_source, COLOR_CYAN));
    println!("  Model ID: {}", colorize(&active_model_id, COLOR_CYAN));
    println!();
    println!("{}", colorize("  Thresholds", COLOR_WHITE_BOLD));
    if required_valid_frames > 0 {
        println!(
            "  Valid frame threshold: {} / {} ({})",
            colorize(
                &stats.valid_frames.to_string(),
                if valid_frame_threshold_met {
                    COLOR_GREEN
                } else {
                    COLOR_YELLOW
                }
            ),
            required_valid_frames,
            if valid_frame_threshold_met {
                "met"
            } else {
                "not met"
            }
        );
    } else {
        println!(
            "  Valid frame threshold: disabled (valid_frames = 0, observed {})",
            stats.valid_frames
        );
    }
    println!(
        "  Consecutive match threshold: {} / {} ({})",
        colorize(
            &stats.max_consecutive_matches.to_string(),
            if consecutive_threshold_met {
                COLOR_GREEN
            } else {
                COLOR_YELLOW
            }
        ),
        consecutive_match_frames,
        if consecutive_threshold_met {
            "met"
        } else {
            "not met"
        }
    );
    println!();
    println!("{}", colorize("  Frame Stats", COLOR_WHITE_BOLD));
    println!("  Total frames: {}", stats.total_frames);
    println!(
        "  Valid frames: {}",
        colorize(&stats.valid_frames.to_string(), COLOR_GREEN)
    );
    println!(
        "  Matched frames: {}",
        colorize(&stats.matched_frames.to_string(), COLOR_GREEN)
    );
    println!(
        "  Max consecutive matched frames: {}",
        colorize(&stats.max_consecutive_matches.to_string(), COLOR_GREEN)
    );
    println!(
        "  Faces detected: {}",
        colorize(&stats.detected_face_frames.to_string(), COLOR_CYAN)
    );
    println!(
        "  Invalid face frames: {}",
        colorize(&stats.invalid_face_frames.to_string(), COLOR_YELLOW)
    );
    println!(
        "  No-face frames: {}",
        colorize(&stats.no_face_frames.to_string(), COLOR_YELLOW)
    );
    println!(
        "  Unmatched valid frames: {}",
        colorize(&stats.unmatched_frames.to_string(), COLOR_YELLOW)
    );
    println!(
        "  Spoof frames: {}",
        colorize(&stats.spoof_frames.to_string(), COLOR_RED)
    );
    println!(
        "  Frame errors: {}",
        colorize(&stats.frame_errors.to_string(), COLOR_RED)
    );

    Ok(())
}

const COLOR_RESET: &str = "\x1b[0m";
const COLOR_RED: &str = "31";
const COLOR_RED_BOLD: &str = "1;31";
const COLOR_GREEN: &str = "32";
const COLOR_GREEN_BOLD: &str = "1;32";
const COLOR_YELLOW: &str = "33";
const COLOR_CYAN: &str = "36";
const COLOR_CYAN_BOLD: &str = "1;36";
const COLOR_WHITE_BOLD: &str = "1;37";

fn colorize(text: &str, color: &str) -> String {
    format!("\x1b[{color}m{text}{COLOR_RESET}")
}

fn rate_color(rate: f64) -> &'static str {
    if rate >= 80.0 {
        COLOR_GREEN_BOLD
    } else if rate >= 50.0 {
        COLOR_YELLOW
    } else {
        COLOR_RED_BOLD
    }
}

fn stop_reason_color(stop_reason: &str) -> &'static str {
    match stop_reason {
        "timeout_reached" | "frame_limit_reached" => COLOR_YELLOW,
        "valid_frame_threshold_reached" => COLOR_GREEN,
        _ => COLOR_CYAN,
    }
}

fn format_frame_progress(current: u32, total: u32) -> String {
    if total == 0 {
        format!("({}/unlimited)", current)
    } else {
        format!("({}/{})", current, total)
    }
}

fn format_liveness_text(status: &str, score: Option<f32>) -> String {
    match (status, score) {
        ("pass", Some(score)) => format!("pass/{score:.3}"),
        ("spoof", Some(score)) => format!("spoof/{score:.3}"),
        ("invalid", _) => "invalid".to_string(),
        ("error", _) => "error".to_string(),
        _ => "off".to_string(),
    }
}

fn shorten_error(message: &str) -> String {
    const MAX_LEN: usize = 60;
    let shortened: String = message.chars().take(MAX_LEN).collect();
    if shortened.chars().count() == message.chars().count() {
        shortened
    } else {
        format!("{shortened}...")
    }
}

fn print_status_line(last_width: &mut usize, message: &str) -> Result<()> {
    let message_width = message.chars().count();
    let clear_padding = " ".repeat(last_width.saturating_sub(message_width));
    print!("\r{}{}", message, clear_padding);
    io::stdout().flush()?;
    *last_width = message_width;
    Ok(())
}

fn clear_status_line(last_width: &mut usize) -> Result<()> {
    if *last_width == 0 {
        return Ok(());
    }

    print!("\r{}\r", " ".repeat(*last_width));
    io::stdout().flush()?;
    *last_width = 0;
    Ok(())
}

fn draw_faces(
    image: &mut Mat,
    faces: &[DetectionResult],
    primary_face: Option<&DetectionResult>,
    primary_color: Option<Scalar>,
) -> Result<()> {
    for face in faces {
        let (x, y, w, h) = face.bbox;
        let rect = Rect::new(x.max(0.0) as i32, y.max(0.0) as i32, w.max(0.0) as i32, h.max(0.0) as i32);
        let color = if is_primary_face(face, primary_face) {
            primary_color.unwrap_or_else(default_face_box_color)
        } else {
            default_face_box_color()
        };
        imgproc::rectangle(image, rect, color, 2, imgproc::LINE_8, 0)?;
    }
    Ok(())
}

fn is_primary_face(face: &DetectionResult, primary_face: Option<&DetectionResult>) -> bool {
    let Some(primary_face) = primary_face else {
        return false;
    };
    let (x, y, w, h) = face.bbox;
    let (px, py, pw, ph) = primary_face.bbox;
    (x - px).abs() <= 0.5 && (y - py).abs() <= 0.5 && (w - pw).abs() <= 0.5 && (h - ph).abs() <= 0.5
}

fn default_face_box_color() -> Scalar {
    Scalar::new(0.0, 255.0, 0.0, 0.0)
}

fn liveness_box_color(status: &str) -> Scalar {
    match status {
        "spoof" | "invalid" | "error" => Scalar::new(0.0, 0.0, 255.0, 0.0),
        _ => default_face_box_color(),
    }
}

fn draw_text(image: &mut Mat, line: i32, text: &str, color: Scalar) -> Result<()> {
    let origin = Point::new(10, 25 + line * 20);
    imgproc::put_text(
        image,
        text,
        origin,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        1,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

fn summarize_face_candidates(candidates: &[FaceRecord]) -> String {
    let mut counts = BTreeMap::new();
    for candidate in candidates {
        *counts.entry(candidate.label.as_str()).or_insert(0usize) += 1;
    }

    let mut parts = Vec::new();
    for (label, count) in counts {
        if count == 1 {
            parts.push(label.to_string());
        } else {
            parts.push(format!("{label} x{count}"));
        }
    }

    parts.join(", ")
}

fn should_end(key: i32) -> bool {
    matches!(key, 27 | 10 | 13 | 113 | 81)
}

fn close_view_window(window_name: &str, view: bool) -> Result<()> {
    let _ = (window_name, view);
    Ok(())
}

fn draw_required_crops(
    image: &mut Mat,
    faces: &[DetectionResult],
    valid_crop_scale: f32,
) -> Result<()> {
    for detection in faces {
        if let Ok((rect, valid)) =
            facepass_core::face_validation::compute_valid_crop_rect(image, detection, valid_crop_scale)
        {
            let color = if valid {
                Scalar::new(0.0, 255.0, 255.0, 0.0)
            } else {
                Scalar::new(0.0, 0.0, 255.0, 0.0)
            };

            draw_dashed_rect(image, rect, color, 8, 6)?;
        }
    }

    Ok(())
}

fn draw_dashed_rect(image: &mut Mat, rect: Rect, color: Scalar, dash: i32, gap: i32) -> Result<()> {
    let x1 = rect.x;
    let y1 = rect.y;
    let x2 = rect.x + rect.width;
    let y2 = rect.y + rect.height;

    let mut x = x1;
    while x < x2 {
        let x_end = (x + dash).min(x2);
        imgproc::line(
            image,
            Point::new(x, y1),
            Point::new(x_end, y1),
            color,
            1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::line(
            image,
            Point::new(x, y2),
            Point::new(x_end, y2),
            color,
            1,
            imgproc::LINE_8,
            0,
        )?;
        x += dash + gap;
    }

    let mut y = y1;
    while y < y2 {
        let y_end = (y + dash).min(y2);
        imgproc::line(
            image,
            Point::new(x1, y),
            Point::new(x1, y_end),
            color,
            1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::line(
            image,
            Point::new(x2, y),
            Point::new(x2, y_end),
            color,
            1,
            imgproc::LINE_8,
            0,
        )?;
        y += dash + gap;
    }

    Ok(())
}
