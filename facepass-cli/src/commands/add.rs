//! Add face command

use super::get_username;
use anyhow::Result;
use opencv::{
    core::{Point, Rect, Scalar},
    highgui,
    imgproc,
    prelude::*,
};
use facepass_core::{
    anti_spoofing::AntiSpoofDetector,
    camera::Camera,
    config::Config,
    detection::FaceDetector,
    models::FaceRecord,
    recognition::FaceRecognizer,
    storage::FaceStorage,
};
use std::io::{self, Write};

pub fn run(
    config_path: &str,
    user: Option<String>,
    label: Option<String>,
    verbose: bool,
    debug: bool,
) -> Result<()> {
    let username = get_username(user)?;
    let config = Config::load(config_path).unwrap_or_default();

    if verbose {
        println!("Adding face for user: {}", username);
        println!("Config: {:?}", config);
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
    let detector = FaceDetector::new(&config.models.yunet_path, &config.detection)?;
    let recognizer = FaceRecognizer::new(&config.models.sface_path, &config.recognition)?;
    let mut anti_spoof = if debug && config.anti_spoof.enabled {
        match AntiSpoofDetector::new(&config.models.anti_spoof_path, &config.anti_spoof) {
            Ok(d) => {
                println!(
                    "Anti-spoofing enabled (threshold: {:.2}, input: {}x{})",
                    config.anti_spoof.threshold,
                    config.anti_spoof.input_size,
                    config.anti_spoof.input_size
                );
                Some(d)
            }
            Err(e) => {
                eprintln!(
                    "Warning: anti-spoofing unavailable, falling back to face add only: {}",
                    e
                );
                None
            }
        }
    } else {
        None
    };

    println!("Camera opened successfully.");
    println!("Please look at the camera and keep your face centered...");
    if debug {
        println!("Press Enter to capture, Esc to cancel.\n");
    } else {
        println!("Press Ctrl+C to cancel.\n");
    }

    // Try to capture a good face
    let max_attempts = if debug { u32::MAX } else { config.video.max_frames };
    let mut attempt = 0;
    let mut best_confidence = 0.0f32;
    let mut best_feature: Option<Vec<f32>> = None;

    if debug {
        highgui::named_window("FacePass Add", highgui::WINDOW_AUTOSIZE)?;
    }

    while attempt < max_attempts {
        attempt += 1;

        // Read frame
        let frame = match camera.read_frame() {
            Ok(f) => f,
            Err(e) => {
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
                if !debug {
                    print!("\rSearching for face... ({}/{})", attempt, max_attempts);
                    io::stdout().flush()?;
                } else {
                    let mut display = frame.try_clone()?;
                    draw_text(
                        &mut display,
                        0,
                        "Searching for face...",
                        Scalar::new(0.0, 0.0, 255.0, 0.0),
                    )?;
                    highgui::imshow("FacePass Add", &display)?;
                    if should_abort(highgui::wait_key(1)?) {
                        println!("\nAdd cancelled.");
                        highgui::destroy_window("FacePass Add")?;
                        return Ok(());
                    }
                }
                continue;
            }
        };

        // Get first face row
        let face_row = faces.row(0)?.try_clone()?;

        // Get confidence
        let confidence = *faces.at_2d::<f32>(0, 14)?;

        // Align and extract feature
        let aligned = recognizer.align_crop(&frame, &face_row)?;
        let feature = recognizer.extract_feature(&aligned)?;

        // Convert to vec
        let feature_vec = facepass_core::recognition::mat_to_vec(&feature)?;

        // Only use high confidence detections
        if confidence > config.detection.score_threshold && confidence > best_confidence {
            best_confidence = confidence;
            best_feature = Some(feature_vec);
        }

        if debug {
            let mut display = frame.try_clone()?;
            draw_faces(&mut display, &faces)?;
            draw_text(
                &mut display,
                0,
                &format!("Detection: {:.1}%", confidence * 100.0),
                Scalar::new(0.0, 255.0, 0.0, 0.0),
            )?;

            if let Some(ref anti_spoof_detector) = anti_spoof {
                match anti_spoof_detector.check_liveness(&aligned) {
                    Ok(score) if score >= config.anti_spoof.threshold => {
                        draw_text(
                            &mut display,
                            1,
                            &format!("Liveness: PASS ({:.3})", score),
                            Scalar::new(0.0, 255.0, 0.0, 0.0),
                        )?;
                    }
                    Ok(score) => {
                        draw_text(
                            &mut display,
                            1,
                            &format!("Liveness: SPOOF ({:.3})", score),
                            Scalar::new(0.0, 0.0, 255.0, 0.0),
                        )?;
                    }
                    Err(e) => {
                        draw_text(
                            &mut display,
                            1,
                            "Liveness: ERROR",
                            Scalar::new(0.0, 0.0, 255.0, 0.0),
                        )?;
                        eprintln!("Warning: disabling anti-spoofing after error: {}", e);
                        anti_spoof = None;
                    }
                }
            } else {
                draw_text(
                    &mut display,
                    1,
                    "Liveness: disabled",
                    Scalar::new(200.0, 200.0, 200.0, 0.0),
                )?;
            }

            draw_text(
                &mut display,
                2,
                "Press Enter to capture, Esc to cancel",
                Scalar::new(255.0, 255.0, 255.0, 0.0),
            )?;

            highgui::imshow("FacePass Add", &display)?;
            let key = highgui::wait_key(1)?;
            if should_abort(key) {
                println!("\nAdd cancelled.");
                highgui::destroy_window("FacePass Add")?;
                return Ok(());
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
                highgui::imshow("FacePass Add", &display)?;
            }
        } else {
            print!(
                "\rFace detected! Confidence: {:.1}% ({}/{})",
                confidence * 100.0,
                attempt,
                max_attempts
            );
            io::stdout().flush()?;

            // If we have a very good detection, stop early
            if confidence > 0.98 {
                break;
            }
        }
    }

    if debug {
        highgui::destroy_window("FacePass Add")?;
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
    matches!(key, 27)
}

fn should_capture(key: i32) -> bool {
    matches!(key, 10 | 13)
}
