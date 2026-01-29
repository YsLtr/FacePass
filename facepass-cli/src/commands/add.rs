//! Add face command

use super::get_username;
use anyhow::Result;
use opencv::prelude::*;
use facepass_core::{
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

    println!("Camera opened successfully.");
    println!("Please look at the camera and keep your face centered...");
    println!("Press Ctrl+C to cancel.\n");

    // Try to capture a good face
    let max_attempts = config.video.max_frames;
    let mut attempt = 0;
    let mut best_confidence = 0.0f32;
    let mut best_feature: Option<Vec<f32>> = None;

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
                print!("\rSearching for face... ({}/{})", attempt, max_attempts);
                io::stdout().flush()?;
                continue;
            }
        };

        // Get first face row
        let face_row = faces.row(0)?.try_clone()?;

        // Get confidence
        let confidence = *faces.at_2d::<f32>(0, 14)?;

        print!(
            "\rFace detected! Confidence: {:.1}% ({}/{})",
            confidence * 100.0,
            attempt,
            max_attempts
        );
        io::stdout().flush()?;

        // Only use high confidence detections
        if confidence > config.detection.score_threshold && confidence > best_confidence {
            // Align and extract feature
            let aligned = recognizer.align_crop(&frame, &face_row)?;
            let feature = recognizer.extract_feature(&aligned)?;

            // Convert to vec
            let feature_vec = facepass_core::recognition::mat_to_vec(&feature)?;

            best_confidence = confidence;
            best_feature = Some(feature_vec);

            // If we have a very good detection, stop early
            if confidence > 0.98 {
                break;
            }
        }
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
