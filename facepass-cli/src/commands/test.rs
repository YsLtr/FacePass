//! Test face recognition command

use super::get_username;
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
    verbose: bool,
    debug: bool,
) -> Result<()> {
    let username = get_username(user)?;
    let config = Config::load_with_fallback(config_path)?;
    let frames = frames_override.unwrap_or(config.video.max_frames);

    if verbose {
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
    if debug {
        println!("Press Enter or Esc to stop.\n");
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

    if debug {
        highgui::named_window("FacePass Test", highgui::WINDOW_AUTOSIZE)?;
    }

    let mut frame_idx = 0u32;
    loop {
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
                    eprintln!("Warning: anti-spoof detector restarted successfully");
                }
                Err(e) => {
                    stats.anti_spoof_errors += 1;
                    eprintln!(
                        "Warning: anti-spoof restart failed ({}/{}): {}",
                        stats.anti_spoof_errors, MAX_ANTI_SPOOF_ERRORS, e
                    );
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
                if verbose {
                    eprintln!("Frame error: {}", e);
                }
                continue;
            }
        };

        // Detect face
        let faces = match detector.detect_raw(&frame) {
            Ok(f) => f,
            Err(_) => {
                stats.no_face_frames += 1;
                if !debug {
                    print!("\rSearching for face... ({}/{})", frame_idx, frames);
                    io::stdout().flush()?;
                } else {
                    let mut display = frame.try_clone()?;
                    draw_text(
                        &mut display,
                        0,
                        "Searching for face...",
                        Scalar::new(0.0, 0.0, 255.0, 0.0),
                    )?;
                    highgui::imshow("FacePass Test", &display)?;
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
                if !debug {
                    print!(
                        "\r! Invalid face ({}) ({}/{})",
                        reason, frame_idx, frames
                    );
                    io::stdout().flush()?;
                } else {
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
                    highgui::imshow("FacePass Test", &display)?;
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
                    if verbose {
                        eprintln!(
                            "Liveness passed on frame {} (score: {:.3})",
                            frame_idx, score
                        );
                    }
                }
                Ok(score) => {
                    stats.spoof_frames += 1;
                    liveness_score = Some(score);
                    liveness_status = "spoof";
                    liveness_allowed = false;
                    consecutive_matches = 0;
                    if !debug {
                        print!(
                            "\r! Spoof detected #{} (score: {:.3}) ({}/{})",
                            stats.spoof_frames, score, frame_idx, frames
                        );
                        io::stdout().flush()?;
                    }
                }
                Err(facepass_core::Error::InvalidFace(reason)) => {
                    liveness_status = "invalid";
                    liveness_allowed = false;
                    stats.invalid_face_frames += 1;
                    consecutive_matches = 0;
                    if !debug {
                        print!(
                            "\r! Invalid face ({}) ({}/{})",
                            reason, frame_idx, frames
                        );
                        io::stdout().flush()?;
                    } else {
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
                        highgui::imshow("FacePass Test", &display)?;
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
                    if verbose {
                        eprintln!("\rAnti-spoof error on frame {}: {}", frame_idx, e);
                    }
                    eprintln!(
                        "Warning: anti-spoof error ({}/{}), restarting detector: {}",
                        stats.anti_spoof_errors, MAX_ANTI_SPOOF_ERRORS, e
                    );
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
                    if !debug {
                        print!(
                            "\r✓ Match #{}: {} (similarity: {:.2}%) ({}/{})",
                            stats.matched_frames,
                            m.face_data.label,
                            m.similarity * 100.0,
                            frame_idx,
                            frames
                        );
                    }
                }
                Ok(None) => {
                    stats.unmatched_frames += 1;
                    consecutive_matches = 0;
                    if !debug {
                        print!(
                            "\r✗ No match (best < {:.0}%) ({}/{})",
                            threshold * 100.0,
                            frame_idx,
                            frames
                        );
                    }
                }
                Err(e) => {
                    consecutive_matches = 0;
                    if verbose {
                        eprintln!("\rMatch error: {}", e);
                    }
                }
            }
        }

        if debug {
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

            highgui::imshow("FacePass Test", &display)?;
            let key = highgui::wait_key(1)?;
            if should_end(key) {
                stop_reason = "user_stopped".to_string();
                break;
            }
        } else {
            io::stdout().flush()?;
        }

        if stop_on_valid_frames && stats.valid_frames >= required_valid_frames {
            stop_reason = "valid_frame_threshold_reached".to_string();
            break;
        }
    }

    if debug {
        highgui::destroy_window("FacePass Test")?;
    }

    let valid_frame_threshold_met = stats.valid_frames >= required_valid_frames;
    let consecutive_threshold_met =
        stats.max_consecutive_matches >= consecutive_match_frames;
    let anti_spoof_failed = stop_reason == "anti_spoof_error";
    let valid_frame_requirement_enabled = required_valid_frames > 0;
    let result_valid = (!valid_frame_requirement_enabled || valid_frame_threshold_met)
        && consecutive_threshold_met
        && !anti_spoof_failed;

    println!("\n");
    println!("Test complete!");
    println!("  Total frames: {}", stats.total_frames);
    println!("  Valid frames: {}", stats.valid_frames);
    println!("  Matched frames: {}", stats.matched_frames);
    println!(
        "  Max consecutive matched frames: {}",
        stats.max_consecutive_matches
    );
    println!("  Faces detected: {}", stats.detected_face_frames);
    println!("  Invalid face frames: {}", stats.invalid_face_frames);
    println!("  No-face frames: {}", stats.no_face_frames);
    println!("  Unmatched valid frames: {}", stats.unmatched_frames);
    println!("  Spoof frames: {}", stats.spoof_frames);
    println!("  Frame errors: {}", stats.frame_errors);
    println!(
        "  Anti-spoof errors: {} / {}",
        stats.anti_spoof_errors, MAX_ANTI_SPOOF_ERRORS
    );
    println!("  Valid frame rate: {:.1}%", stats.valid_frame_rate());
    println!("  Match success rate: {:.1}%", stats.match_success_rate());
    if valid_frame_requirement_enabled {
        println!(
            "  Valid frame threshold: {} / {} ({})",
            stats.valid_frames,
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
        stats.max_consecutive_matches,
        consecutive_match_frames,
        if consecutive_threshold_met {
            "met"
        } else {
            "not met"
        }
    );
    println!(
        "  Result validity: {}",
        if result_valid { "valid" } else { "invalid" }
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
    println!("  End reason: {}", stop_reason);

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
    matches!(key, 27 | 10 | 13)
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
