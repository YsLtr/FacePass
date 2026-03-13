//! Test face recognition command

use super::{CommandInput, CommandKey, get_username};
use anyhow::Result;
use facepass_core::{
    anti_spoofing::AntiSpoofDetector,
    camera::Camera,
    config::Config,
    detection::FaceDetector,
    face_validation::select_primary_face,
    matching::find_best_match,
    recognition::{mat_to_vec, FaceRecognizer},
    storage::FaceStorage,
};
use opencv::{
    core::{Point, Rect, Scalar},
    highgui, imgproc,
    prelude::*,
};
use std::io::{self, Write};
use std::time::{Duration, Instant};

const MAX_ANTI_SPOOF_ERRORS: u32 = 3;

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
    anti_spoof_errors: u32,
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
    frames_override: Option<u32>,
    debug: bool,
    view: bool,
) -> Result<()> {
    const WINDOW_NAME: &str = "FacePass Test";

    let username = get_username(user)?;
    let (config, config_source) = Config::load_with_fallback_and_source(config_path)?;
    let frames = frames_override.unwrap_or(config.video.max_frames);
    let config_source_display = config_source
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "built-in defaults".to_string());

    if debug {
        println!("Testing face recognition for user: {}", username);
    }

    // Load registered faces
    let storage = FaceStorage::new(&config.storage.data_dir)?;
    let face_data = storage.load_face_data(&username)?;

    if face_data.is_empty() {
        return Err(anyhow::anyhow!(
            "No faces registered for user '{}'. Use 'facepass add' first.",
            username
        ));
    }

    println!("Loaded {} registered face(s)", face_data.len());
    println!("Config source: {}", colorize(&config_source_display, COLOR_CYAN));
    println!("Initializing camera...\n");

    // Initialize components
    let camera = Camera::open(&config.video)?;
    let actual_width = camera.frame_width().ok();
    let actual_height = camera.frame_height().ok();
    let detector = FaceDetector::new(&config.models.yunet_path, &config.detection)?;
    let recognizer = FaceRecognizer::new(&config.models.sface_path, &config.recognition)?;
    let mut anti_spoof = if config.anti_spoof.enabled {
        match AntiSpoofDetector::new(&config.models, &config.anti_spoof) {
            Ok(d) => {
                println!(
                    "Anti-spoofing enabled (threshold: {:.2}, mode: {})",
                    config.anti_spoof.threshold,
                    config.anti_spoof.mode.as_str()
                );
                Some(d)
            }
            Err(e) => {
                eprintln!(
                    "Warning: anti-spoofing unavailable at startup, will retry during test: {}",
                    e
                );
                None
            }
        }
    } else {
        println!("Anti-spoofing disabled");
        None
    };

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

            if config.anti_spoof.enabled
                && anti_spoof.is_none()
                && stats.anti_spoof_errors < MAX_ANTI_SPOOF_ERRORS
            {
                match AntiSpoofDetector::new(&config.models, &config.anti_spoof) {
                    Ok(detector) => {
                        anti_spoof = Some(detector);
                        print_status_line(
                            &mut last_status_width,
                            &format!(
                                "Anti-spoof detector restarted {}",
                                format_frame_progress(frame_idx, frames)
                            ),
                        )?;
                    }
                    Err(e) => {
                        stats.anti_spoof_errors += 1;
                        print_status_line(
                            &mut last_status_width,
                            &format!(
                                "Anti-spoof restart failed ({}/{}) {}",
                                stats.anti_spoof_errors,
                                MAX_ANTI_SPOOF_ERRORS,
                                shorten_error(&e.to_string())
                            ),
                        )?;
                        if stats.anti_spoof_errors >= MAX_ANTI_SPOOF_ERRORS {
                            stop_reason = "anti_spoof_error".to_string();
                            break;
                        }
                    }
                }
            }

            // Read frame
            let frame = match camera.read_frame() {
                Ok(f) => f,
                Err(e) => {
                    stats.frame_errors += 1;
                    if debug {
                        print_status_line(
                            &mut last_status_width,
                            &format!(
                                "Frame error #{}: {}",
                                stats.frame_errors,
                                shorten_error(&e.to_string())
                            ),
                        )?;
                    }
                    continue;
                }
            };

            // Detect face
            let faces = match detector.detect_raw(&frame) {
                Ok(f) => f,
                Err(_) => {
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
                        let key = highgui::wait_key(1)?;
                        if should_end(key) {
                            stop_reason = "user_stopped".to_string();
                            break;
                        }
                    }
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
                    stats.detected_face_frames += 1;
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
                        draw_faces(&mut display, &faces)?;
                        draw_required_crops(
                            &mut display,
                            &faces,
                            config.recognition.valid_crop_scale,
                        )?;
                        draw_text(
                            &mut display,
                            0,
                            &format!("Invalid face: {}", reason),
                            Scalar::new(0.0, 0.0, 255.0, 0.0),
                        )?;
                        highgui::imshow(WINDOW_NAME, &display)?;
                        let key = highgui::wait_key(1)?;
                        if should_end(key) {
                            stop_reason = "user_stopped".to_string();
                            break;
                        }
                    }
                    continue;
                }
                Err(_) => continue,
            };
            stats.detected_face_frames += 1;
            stats.valid_frames += 1;
            let confidence = *face_row.at_2d::<f32>(0, 14)?;

            let mut liveness_score: Option<f32> = None;
            let mut liveness_status = "disabled";
            let mut liveness_allowed = true;
            let mut fatal_anti_spoof_error = false;
            if let Some(ref anti_spoof_detector) = anti_spoof {
                match anti_spoof_detector.check_liveness(
                    &frame,
                    &face_row,
                    config.recognition.valid_crop_scale,
                ) {
                    Ok(score) if score >= config.anti_spoof.threshold => {
                        liveness_score = Some(score);
                        liveness_status = "pass";
                    }
                    Ok(score) => {
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
                            draw_faces(&mut display, &faces)?;
                            draw_required_crops(
                                &mut display,
                                &faces,
                                config.recognition.valid_crop_scale,
                            )?;
                            draw_text(
                                &mut display,
                                0,
                                &format!("Invalid face: {}", reason),
                                Scalar::new(0.0, 0.0, 255.0, 0.0),
                            )?;
                            highgui::imshow(WINDOW_NAME, &display)?;
                            let key = highgui::wait_key(1)?;
                            if should_end(key) {
                                stop_reason = "user_stopped".to_string();
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        stats.anti_spoof_errors += 1;
                        liveness_status = "error";
                        liveness_allowed = false;
                        consecutive_matches = 0;
                        print_status_line(
                            &mut last_status_width,
                            &format!(
                                "Anti-spoof error ({}/{}): {} {}",
                                stats.anti_spoof_errors,
                                MAX_ANTI_SPOOF_ERRORS,
                                shorten_error(&e.to_string()),
                                format_frame_progress(frame_idx, frames)
                            ),
                        )?;
                        anti_spoof = None;
                        if stats.anti_spoof_errors >= MAX_ANTI_SPOOF_ERRORS {
                            stop_reason = "anti_spoof_error".to_string();
                            fatal_anti_spoof_error = true;
                        }
                    }
                }
            }

            if fatal_anti_spoof_error {
                break;
            }

            if !liveness_allowed && liveness_status == "invalid" {
                continue;
            }

            // Match against registered faces
            let mut match_label: Option<String> = None;
            let mut match_score: Option<f64> = None;
            if liveness_allowed {
                let aligned = recognizer.align_crop(&frame, &face_row)?;
                let feature_mat = recognizer.extract_feature(&aligned)?;
                let feature = mat_to_vec(&feature_mat)?;

                match find_best_match(&feature, &face_data, threshold) {
                    Ok(Some(m)) => {
                        stats.matched_frames += 1;
                        consecutive_matches += 1;
                        stats.max_consecutive_matches =
                            stats.max_consecutive_matches.max(consecutive_matches);
                        match_label = Some(m.face_data.label.clone());
                        match_score = Some(m.similarity);
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
                    }
                    Ok(None) => {
                        stats.unmatched_frames += 1;
                        consecutive_matches = 0;
                        print_status_line(
                            &mut last_status_width,
                            &format!(
                                "No match (< {:.0}%) det:{:.1}% live:{} {}",
                                threshold * 100.0,
                                confidence * 100.0,
                                format_liveness_text(liveness_status, liveness_score),
                                format_frame_progress(frame_idx, frames)
                            ),
                        )?;
                    }
                    Err(e) => {
                        consecutive_matches = 0;
                        if debug {
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
            }

            if view {
                let mut display = frame.try_clone()?;
                draw_faces(&mut display, &faces)?;
                draw_required_crops(&mut display, &faces, config.recognition.valid_crop_scale)?;
                draw_text(
                    &mut display,
                    0,
                    &format!("Detection: {:.1}%", confidence * 100.0),
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                )?;

                let live_text = match (liveness_status, liveness_score) {
                    ("pass", Some(s)) => format!("Anti-spoof: PASS ({:.3})", s),
                    ("spoof", Some(s)) => format!("Anti-spoof: SPOOF ({:.3})", s),
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

                let match_text = match (match_label, match_score) {
                    (Some(label), Some(score)) => {
                        format!("Match: {} ({:.2}%)", label, score * 100.0)
                    }
                    _ => format!("Match: none (threshold {:.0}%)", threshold * 100.0),
                };
                draw_text(
                    &mut display,
                    2,
                    &match_text,
                    Scalar::new(255.0, 255.0, 255.0, 0.0),
                )?;

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
                draw_text(
                    &mut display,
                    4,
                    &format!(
                        "Total: {}  Invalid: {}  NoFace: {}  Err: {}",
                        stats.total_frames,
                        stats.invalid_face_frames,
                        stats.no_face_frames,
                        stats.anti_spoof_errors + stats.frame_errors
                    ),
                    Scalar::new(180.0, 180.0, 180.0, 0.0),
                )?;

                highgui::imshow(WINDOW_NAME, &display)?;
                let key = highgui::wait_key(1)?;
                if should_end(key) {
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
    drop(anti_spoof);
    drop(recognizer);
    drop(detector);
    drop(camera);
    close_view_window(WINDOW_NAME, view)?;
    loop_result?;

    let valid_frame_threshold_met = stats.valid_frames >= required_valid_frames;
    let consecutive_threshold_met =
        stats.max_consecutive_matches >= consecutive_match_frames;
    let anti_spoof_failed = stop_reason == "anti_spoof_error";
    let valid_frame_requirement_enabled = required_valid_frames > 0;
    let result_valid = (!valid_frame_requirement_enabled || valid_frame_threshold_met)
        && consecutive_threshold_met
        && !anti_spoof_failed;

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
    println!("  Config source: {}", colorize(&config_source_display, COLOR_CYAN));
    println!();
    println!("{}", colorize("  Thresholds", COLOR_WHITE_BOLD));
    if valid_frame_requirement_enabled {
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
            if valid_frame_threshold_met { "met" } else { "not met" }
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
    println!(
        "  Stop on valid frames: {}",
        if required_valid_frames == 0 {
            "false (ignored because valid_frames = 0)".to_string()
        } else if debug {
            "false (ignored in debug mode)".to_string()
        } else {
            stop_on_valid_frames.to_string()
        }
    );
    println!(
        "  Timeout: {}{}",
        if config.video.timeout == 0 {
            "unlimited".to_string()
        } else {
            format!("{}s", config.video.timeout)
        },
        if debug { " (ignored in debug mode)" } else { "" }
    );
    println!(
        "  Frame limit: {}{}",
        if frames == 0 {
            "unlimited".to_string()
        } else {
            frames.to_string()
        },
        if debug { " (ignored in debug mode)" } else { "" }
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
    println!(
        "  Anti-spoof errors: {} / {}",
        colorize(&stats.anti_spoof_errors.to_string(), COLOR_RED),
        MAX_ANTI_SPOOF_ERRORS
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
        "anti_spoof_error" => COLOR_RED_BOLD,
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

fn draw_faces(image: &mut Mat, faces: &Mat) -> Result<()> {
    let rows = faces.rows();
    for i in 0..rows {
        let x = *faces.at_2d::<f32>(i, 0)? as i32;
        let y = *faces.at_2d::<f32>(i, 1)? as i32;
        let w = *faces.at_2d::<f32>(i, 2)? as i32;
        let h = *faces.at_2d::<f32>(i, 3)? as i32;
        let rect = Rect::new(x.max(0), y.max(0), w.max(0), h.max(0));
        imgproc::rectangle(
            image,
            rect,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
    }
    Ok(())
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

fn should_end(key: i32) -> bool {
    matches!(key, 27 | 10 | 13 | 113 | 81)
}

fn close_view_window(window_name: &str, view: bool) -> Result<()> {
    let _ = (window_name, view);
    Ok(())
}

fn draw_required_crops(image: &mut Mat, faces: &Mat, valid_crop_scale: f32) -> Result<()> {
    for row_idx in 0..faces.rows() {
        let face_row = faces.row(row_idx)?.try_clone()?;
        if let Ok((rect, valid)) =
            facepass_core::face_validation::compute_valid_crop_rect(image, &face_row, valid_crop_scale)
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
