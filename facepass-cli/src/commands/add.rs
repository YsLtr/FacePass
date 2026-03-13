//! Add face command

use super::{CommandInput, CommandKey, get_username};
use anyhow::{anyhow, Result};
use facepass_core::{
    anti_spoofing::AntiSpoofDetector,
    camera::Camera,
    config::Config,
    detection::FaceDetector,
    face_validation::select_primary_face,
    models::FaceRecord,
    recognition::FaceRecognizer,
    storage::FaceStorage,
};
use opencv::{
    core::{Point, Rect, Scalar},
    highgui, imgproc,
    prelude::*,
};
use std::io::{self, Write};

enum AddLoopOutcome {
    Cancelled,
    Finished,
}

pub fn run(
    config_path: &str,
    user: Option<String>,
    label: Option<String>,
    debug: bool,
    view: bool,
) -> Result<()> {
    const WINDOW_NAME: &str = "FacePass Add";

    let username = get_username(user)?;
    let config = Config::load_with_fallback(config_path)?;

    if debug {
        println!("Adding face for user: {}", username);
    }

    // Check face count limit
    let storage = FaceStorage::new(&config.storage.data_dir)?;
    let current_count = storage.face_count(&username)?;

    if current_count >= config.recognition.max_faces_per_user as usize {
        return Err(anyhow::anyhow!(
            "Maximum face limit reached ({}/{}). Remove some faces first.",
            current_count,
            config.recognition.max_faces_per_user
        ));
    }

    // Get label
    let face_label = if let Some(l) = label {
        l
    } else {
        print!("Enter a label for this face (e.g., 'normal', 'with glasses'): ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let trimmed = input.trim();
        if trimmed.is_empty() {
            format!("Face {}", current_count + 1)
        } else {
            trimmed.to_string()
        }
    };

    println!("Initializing camera...");

    // Initialize components
    let camera = Camera::open(&config.video)?;
    let actual_width = camera.frame_width().ok();
    let actual_height = camera.frame_height().ok();
    let detector = FaceDetector::new(&config.models.yunet_path, &config.detection)?;
    let recognizer = FaceRecognizer::new(&config.models.sface_path, &config.recognition)?;
    let anti_spoof = if config.anti_spoof.enabled {
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
                return Err(anyhow!("Anti-spoof error: {}", e));
            }
        }
    } else {
        None
    };

    println!("Camera opened successfully.");
    if let (Some(width), Some(height)) = (actual_width, actual_height) {
        println!(
            "Camera resolution: actual {}x{}, requested {}x{}",
            width, height, config.video.frame_width, config.video.frame_height
        );
    }
    println!("Please look at the camera and keep your face centered...");
    if view {
        println!("Press Enter/Esc in the window, or a/q in the terminal.\n");
    } else if debug {
        println!("Press a to add, or q to cancel.\n");
    } else {
        println!("Press Ctrl+C to cancel.\n");
    }

    // Try to capture a good face
    let max_attempts = if debug || view || config.video.max_frames == 0 {
        u32::MAX
    } else {
        config.video.max_frames
    };
    let mut attempt = 0;
    let mut best_confidence = 0.0f32;
    let mut best_feature: Option<Vec<f32>> = None;
    let mut capture_requested = false;
    let mut last_status_width = 0usize;
    let mut command_input = if view || debug {
        Some(CommandInput::capture_single_keys()?)
    } else {
        None
    };

    let loop_outcome = (|| -> Result<AddLoopOutcome> {
        while attempt < max_attempts {
            loop {
                let command = match command_input.as_ref() {
                    Some(command_input) => command_input.poll_key()?,
                    None => None,
                };

                let Some(command) = command else {
                    break;
                };

                match command {
                    CommandKey::Quit => {
                        clear_status_line(&mut last_status_width)?;
                        return Ok(AddLoopOutcome::Cancelled);
                    }
                    CommandKey::Action => {
                        if !capture_requested && best_feature.is_none() {
                            print_status_line(
                                &mut last_status_width,
                                &format!(
                                    "Add requested. Waiting for a valid face... ({}/{})",
                                    attempt, max_attempts
                                ),
                            )?;
                        }
                        capture_requested = true;
                    }
                }
            }

            attempt += 1;

            // Read frame
            let frame = match camera.read_frame() {
                Ok(f) => f,
                Err(e) => {
                    if debug {
                        print_status_line(
                            &mut last_status_width,
                            &format!("Frame error: {}", shorten_error(&e.to_string())),
                        )?;
                    }
                    continue;
                }
            };

            // Detect face
            let faces = match detector.detect_raw(&frame) {
                Ok(f) => f,
                Err(_) => {
                    print_status_line(
                        &mut last_status_width,
                        &format!("Searching for face... ({}/{})", attempt, max_attempts),
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
                        if should_abort(highgui::wait_key(1)?) {
                            clear_status_line(&mut last_status_width)?;
                            return Ok(AddLoopOutcome::Cancelled);
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
                    print_status_line(
                        &mut last_status_width,
                        &format!("Invalid face ({}) ({}/{})", reason, attempt, max_attempts),
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
                        if should_abort(highgui::wait_key(1)?) {
                            clear_status_line(&mut last_status_width)?;
                            return Ok(AddLoopOutcome::Cancelled);
                        }
                    }
                    continue;
                }
                Err(_) => continue,
            };

            let confidence = *face_row.at_2d::<f32>(0, 14)?;

            let mut liveness_score: Option<f32> = None;
            let mut liveness_status = "disabled";
            let mut liveness_allowed = true;
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
                        liveness_score = Some(score);
                        liveness_status = "spoof";
                        liveness_allowed = false;
                        print_status_line(
                            &mut last_status_width,
                            &format!(
                                "Spoof detected ({score:.3}) det:{:.1}% ({}/{})",
                                confidence * 100.0,
                                attempt,
                                max_attempts
                            ),
                        )?;
                    }
                    Err(facepass_core::Error::InvalidFace(reason)) => {
                        liveness_status = "invalid";
                        liveness_allowed = false;
                        print_status_line(
                            &mut last_status_width,
                            &format!("Invalid face ({}) ({}/{})", reason, attempt, max_attempts),
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
                            if should_abort(highgui::wait_key(1)?) {
                                clear_status_line(&mut last_status_width)?;
                                return Ok(AddLoopOutcome::Cancelled);
                            }
                        }
                    }
                    Err(e) => {
                        clear_status_line(&mut last_status_width)?;
                        return Err(anyhow!("Anti-spoof error: {}", e));
                    }
                }
            }

            if !liveness_allowed && liveness_status == "invalid" {
                continue;
            }

            if liveness_allowed {
                let aligned = recognizer.align_crop(&frame, &face_row)?;
                let feature = recognizer.extract_feature(&aligned)?;
                let feature_vec = facepass_core::recognition::mat_to_vec(&feature)?;

                if confidence > config.detection.score_threshold && confidence > best_confidence {
                    best_confidence = confidence;
                    best_feature = Some(feature_vec);
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

                draw_text(
                    &mut display,
                    2,
                    "Enter/a capture, Esc/q cancel",
                    Scalar::new(255.0, 255.0, 255.0, 0.0),
                )?;

                highgui::imshow(WINDOW_NAME, &display)?;
                let key = highgui::wait_key(1)?;
                if should_abort(key) {
                    clear_status_line(&mut last_status_width)?;
                    return Ok(AddLoopOutcome::Cancelled);
                }
                if should_capture(key) {
                    if best_feature.is_some() {
                        break;
                    }
                    draw_text(
                        &mut display,
                        3,
                        "No valid face yet",
                        Scalar::new(0.0, 0.0, 255.0, 0.0),
                    )?;
                    highgui::imshow(WINDOW_NAME, &display)?;
                }
            } else if liveness_allowed {
                print_status_line(
                    &mut last_status_width,
                    &format!(
                        "Face detected det:{:.1}% live:{} ({}/{})",
                        confidence * 100.0,
                        format_liveness_text(liveness_status, liveness_score),
                        attempt,
                        max_attempts
                    ),
                )?;

                // Non-debug CLI mode still supports fast capture without extra input.
                if best_feature.is_some() && !debug && !view && confidence > 0.98 {
                    break;
                }
            }

            if capture_requested && best_feature.is_some() {
                break;
            }
        }

        Ok(AddLoopOutcome::Finished)
    })();

    clear_status_line(&mut last_status_width)?;
    drop(command_input.take());
    drop(anti_spoof);
    drop(recognizer);
    drop(detector);
    drop(camera);
    close_view_window(WINDOW_NAME, view)?;

    match loop_outcome? {
        AddLoopOutcome::Cancelled => {
            println!("\nAdd cancelled.");
            return Ok(());
        }
        AddLoopOutcome::Finished => {}
    }

    println!();

    if let Some(feature) = best_feature {
        // Save the face
        let record = FaceRecord::new(&username, &face_label, feature);
        storage.save_face(&record)?;

        println!("\n✓ Face added successfully!");
        println!("  User: {}", username);
        println!("  Label: {}", face_label);
        println!("  Confidence: {:.1}%", best_confidence * 100.0);
        println!("  ID: {}", record.id);

        let new_count = storage.face_count(&username)?;
        println!(
            "  Total faces: {}/{}",
            new_count, config.recognition.max_faces_per_user
        );
    } else {
        return Err(anyhow::anyhow!(
            "Could not capture a good face image. Please try again with better lighting."
        ));
    }

    Ok(())
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

fn should_abort(key: i32) -> bool {
    matches!(key, 27 | 113 | 81)
}

fn should_capture(key: i32) -> bool {
    matches!(key, 10 | 13 | 97 | 65)
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
