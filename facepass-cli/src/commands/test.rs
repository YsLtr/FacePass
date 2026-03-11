//! Test face recognition command

use super::get_username;
use anyhow::Result;
use opencv::prelude::*;
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

pub fn run(config_path: &str, user: Option<String>, frames: u32, verbose: bool) -> Result<()> {
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
    println!("Press Ctrl+C to stop.\n");

    let mut matches = 0;
    let mut attempts = 0;
    let mut detected_faces = 0;
    let mut spoof_frames = 0;
    let mut liveness_errors = 0;
    let threshold = config.recognition.similarity_threshold;

    for i in 0..frames {
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
                print!("\rSearching for face... ({}/{})", i + 1, frames);
                io::stdout().flush()?;
                continue;
            }
        };

        let face_row = faces.row(0)?.try_clone()?;
        detected_faces += 1;
        attempts += 1;

        // Align and extract feature
        let aligned = recognizer.align_crop(&frame, &face_row)?;

        if let Some(ref anti_spoof_detector) = anti_spoof {
            match anti_spoof_detector.check_liveness(&aligned) {
                Ok(score) if score >= config.anti_spoof.threshold => {
                    if verbose {
                        eprintln!("Liveness passed on frame {} (score: {:.3})", i + 1, score);
                    }
                }
                Ok(score) => {
                    spoof_frames += 1;
                    print!(
                        "\r! Spoof detected #{} (score: {:.3}) ({}/{})",
                        spoof_frames,
                        score,
                        i + 1,
                        frames
                    );
                    io::stdout().flush()?;
                    continue;
                }
                Err(e) => {
                    liveness_errors += 1;
                    if verbose {
                        eprintln!("\rLiveness error on frame {}: {}", i + 1, e);
                    }
                    // Avoid spamming the terminal if the model is incompatible.
                    eprintln!("Warning: disabling anti-spoofing after error: {}", e);
                    anti_spoof = None;
                    continue;
                }
            }
        }

        let feature_mat = recognizer.extract_feature(&aligned)?;
        let feature = mat_to_vec(&feature_mat)?;

        // Match against registered faces
        match find_best_match(&feature, &face_data, threshold) {
            Ok(Some(m)) => {
                matches += 1;
                print!(
                    "\r✓ Match #{}: {} (similarity: {:.2}%) ({}/{})",
                    matches,
                    m.face_data.label,
                    m.similarity * 100.0,
                    i + 1,
                    frames
                );
            }
            Ok(None) => {
                print!("\r✗ No match (best < {:.0}%) ({}/{})", threshold * 100.0, i + 1, frames);
            }
            Err(e) => {
                if verbose {
                    eprintln!("\rMatch error: {}", e);
                }
            }
        }

        io::stdout().flush()?;
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
