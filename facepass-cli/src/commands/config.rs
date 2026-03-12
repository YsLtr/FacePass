//! Config command

use anyhow::Result;
use facepass_core::config::Config;
use std::process::Command;

pub fn run(config_path: &str, show: bool, set: Option<String>, _verbose: bool) -> Result<()> {
    if show || set.is_none() {
        // Show configuration
        let config = Config::load(config_path).unwrap_or_default();

        println!("FacePass Configuration");
        println!("======================\n");

        println!("[video]");
        println!("  device = \"{}\"", config.video.device);
        println!("  timeout = {}", config.video.timeout);
        println!("  max_frames = {}", config.video.max_frames);
        println!("  frame_width = {}", config.video.frame_width);
        println!("  frame_height = {}", config.video.frame_height);
        println!();

        println!("[detection]");
        println!("  score_threshold = {}", config.detection.score_threshold);
        println!("  nms_threshold = {}", config.detection.nms_threshold);
        println!("  input_width = {}", config.detection.input_width);
        println!("  input_height = {}", config.detection.input_height);
        println!();

        println!("[recognition]");
        println!(
            "  similarity_threshold = {}",
            config.recognition.similarity_threshold
        );
        println!(
            "  max_faces_per_user = {}",
            config.recognition.max_faces_per_user
        );
        println!(
            "  required_matches = {}",
            config.recognition.required_matches
        );
        println!();

        println!("[security]");
        println!("  ignore_ssh = {}", config.security.ignore_ssh);
        println!(
            "  ignore_closed_lid = {}",
            config.security.ignore_closed_lid
        );
        println!(
            "  show_notification = {}",
            config.security.show_notification
        );
        println!();

        println!("[daemon]");
        println!("  socket_path = \"{}\"", config.daemon.socket_path);
        println!("  log_level = \"{}\"", config.daemon.log_level);
        println!();

        println!("[models]");
        println!("  yunet_path = \"{}\"", config.models.yunet_path);
        println!("  sface_path = \"{}\"", config.models.sface_path);
        println!();

        println!("[storage]");
        println!("  data_dir = \"{}\"", config.storage.data_dir);

        println!("\n---");
        println!("Config file: {}", config_path);

        return Ok(());
    }

    if let Some(value) = set {
        // Set a value - for now, just open the editor
        println!("Setting configuration values is not yet implemented.");
        println!("Please edit the config file directly:\n");

        // Try to open with EDITOR or default to nano
        let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nano".to_string());

        println!("  {} {}", editor, config_path);
        println!("\nOr use:");
        println!("  sudo facepass config --show");
        println!("\nTo set: {}", value);

        // Optionally open the editor
        if std::path::Path::new(config_path).exists() {
            let status = Command::new(&editor).arg(config_path).status();

            if let Err(e) = status {
                eprintln!("Failed to open editor: {}", e);
            }
        } else {
            println!("\nConfig file does not exist. Creating default...");
            let config = Config::default();
            config.save(config_path)?;
            println!("Created: {}", config_path);
        }
    }

    Ok(())
}
