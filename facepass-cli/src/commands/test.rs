//! Test face recognition command

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
    matching::find_best_match,
    recognition::{mat_to_vec, FaceRecognizer},
    storage::FaceStorage,
};
use std::io::{self, Write};

pub fn run(
    config_path: &str,
    user: Option<String>,
    frames: u32,
    verbose: bool,
    debug: bool,
) -> Result<()> {
    let username = get_username(user)?;
    let config = Config::load(config_path).unwrap_or_default();

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
    let detector = FaceDetector::new(&config.models.yunet_path, &config.detection)?;
    let recognizer = FaceRecognizer::new(&config.models.sface_path, &config.recognition)?;
    let mut anti_spoof = if config.anti_spoof.enabled {
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
                    "Warning: anti-spoofing unavailable, falling back to face recognition only: {}",
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
    if debug {
        println!("Press Enter or Esc to stop.\n");
    } else {
        println!("Press Ctrl+C to stop.\n");
    }

    let mut matches = 0;
    let mut attempts = 0;
    let mut detected_faces = 0;
    let mut spoof_frames = 0;
    let mut liveness_errors = 0;
    let threshold = config.recognition.similarity_threshold;

    if debug {
        highgui::named_window("FacePass Test", highgui::WINDOW_AUTOSIZE)?;
    }

    let mut frame_idx = 0u32;
    loop {
        if !debug && frame_idx >= frames {
            break;
        }
        frame_idx += 1;

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
                        break;
                    }
                }
                continue;
            }
        };

        let face_row = faces.row(0)?.try_clone()?;
        let confidence = *faces.at_2d::<f32>(0, 14)?;
        detected_faces += 1;
        attempts += 1;

        // Align and extract feature
        let aligned = recognizer.align_crop(&frame, &face_row)?;

        let mut liveness_score: Option<f32> = None;
        let mut liveness_status = "disabled";
        if let Some(ref anti_spoof_detector) = anti_spoof {
            match anti_spoof_detector.check_liveness(&aligned) {
                Ok(score) if score >= config.anti_spoof.threshold => {
                    liveness_score = Some(score);
                    liveness_status = "pass";
                    if verbose {
                        eprintln!("Liveness passed on frame {} (score: {:.3})", frame_idx, score);
                    }
                }
                Ok(score) => {
                    spoof_frames += 1;
                    liveness_score = Some(score);
                    liveness_status = "spoof";
                    if !debug {
                        print!(
                            "\r! Spoof detected #{} (score: {:.3}) ({}/{})",
                            spoof_frames,
                            score,
                            frame_idx,
                            frames
                        );
                        io::stdout().flush()?;
                    }
                }
                Err(e) => {
                    liveness_errors += 1;
                    liveness_status = "error";
                    if verbose {
                        eprintln!("\rLiveness error on frame {}: {}", frame_idx, e);
                    }
                    // Avoid spamming the terminal if the model is incompatible.
                    eprintln!("Warning: disabling anti-spoofing after error: {}", e);
                    anti_spoof = None;
                }
            }
        }

        let feature_mat = recognizer.extract_feature(&aligned)?;
        let feature = mat_to_vec(&feature_mat)?;

        // Match against registered faces
        let mut match_label: Option<String> = None;
        let mut match_score: Option<f64> = None;
        match find_best_match(&feature, &face_data, threshold) {
            Ok(Some(m)) => {
                matches += 1;
                match_label = Some(m.face_data.label.clone());
                match_score = Some(m.similarity);
                if !debug {
                    print!(
                        "\r✓ Match #{}: {} (similarity: {:.2}%) ({}/{})",
                        matches,
                        m.face_data.label,
                        m.similarity * 100.0,
                        frame_idx,
                        frames
                    );
                }
            }
            Ok(None) => {
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
                if verbose {
                    eprintln!("\rMatch error: {}", e);
                }
            }
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

            let live_text = match (liveness_status, liveness_score) {
                ("pass", Some(s)) => format!("Liveness: PASS ({:.3})", s),
                ("spoof", Some(s)) => format!("Liveness: SPOOF ({:.3})", s),
                ("error", _) => "Liveness: ERROR".to_string(),
                _ => "Liveness: disabled".to_string(),
            };
            let live_color = match liveness_status {
                "pass" => Scalar::new(0.0, 255.0, 0.0, 0.0),
                "spoof" | "error" => Scalar::new(0.0, 0.0, 255.0, 0.0),
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
                    "Attempts: {}  Matches: {}  Spoof: {}  Errors: {}",
                    attempts, matches, spoof_frames, liveness_errors
                ),
                Scalar::new(200.0, 200.0, 200.0, 0.0),
            )?;

            highgui::imshow("FacePass Test", &display)?;
            let key = highgui::wait_key(1)?;
            if should_end(key) {
                break;
            }
        } else {
            io::stdout().flush()?;
        }
    }

    if debug {
        highgui::destroy_window("FacePass Test")?;
    }

    println!("\n");
    println!("Test complete!");
    println!("  Attempts: {}", attempts);
    println!("  Faces detected: {}", detected_faces);
    println!("  Matches: {}", matches);
    println!("  Spoof frames: {}", spoof_frames);
    println!("  Liveness errors: {}", liveness_errors);
    println!(
        "  Success rate: {:.1}%",
        if attempts > 0 {
            (matches as f64 / attempts as f64) * 100.0
        } else {
            0.0
        }
    );

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
