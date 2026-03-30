//! Status check command

use anyhow::Result;
use facepass_core::{
    camera::check_camera,
    config::Config,
    security::{is_lid_closed, is_ssh_session},
    storage::FaceStorage,
};
use std::path::Path;

pub fn run(config_path: &str) -> Result<()> {
    println!("FacePass System Status");
    println!("======================\n");

    let (config, config_source) = Config::load_with_fallback_and_source(config_path)?;

    print!("Daemon: ");
    if is_daemon_running(&config.daemon.socket_path) {
        println!("OK Running");
    } else {
        println!("X Not running");
    }

    print!("Socket: ");
    if Path::new(&config.daemon.socket_path).exists() {
        println!("OK {}", config.daemon.socket_path);
    } else {
        println!("X Not found");
    }

    println!("\nModels:");
    println!(
        "  YuNet: {} | input: {}x{} | score: {:.2} | nms: {:.2}",
        if Path::new(&config.models.yunet_path).exists() {
            "OK Found"
        } else {
            "X Missing"
        },
        config.detection.input_width,
        config.detection.input_height,
        config.detection.score_threshold,
        config.detection.nms_threshold
    );
    println!("    {}", config.models.yunet_path);

    println!(
        "  SFace: {} | similarity: {:.2} | max/group: {} | consecutive: {}",
        if Path::new(&config.models.sface_path).exists() {
            "OK Found"
        } else {
            "X Missing"
        },
        config.recognition.similarity_threshold,
        config.recognition.max_faces_per_group,
        config.recognition.consecutive_match_frames
    );
    println!("    {}", config.models.sface_path);

    print!("  Anti-spoofing: ");
    if config.anti_spoof.enabled {
        println!(
            "OK Enabled (threshold: {:.2}, mode: {})",
            config.anti_spoof.threshold,
            config.anti_spoof.mode.as_str()
        );
    } else {
        println!("X Disabled");
    }
    println!(
        "  Valid crop scale (recognition): {:.2}",
        config.recognition.valid_crop_scale
    );

    println!(
        "  Anti-spoof V2: {} | input: {}x{} | scale: {:.1}",
        if Path::new(&config.models.anti_spoof_v2_path).exists() {
            "OK Found"
        } else {
            "X Missing"
        },
        config.anti_spoof.v2_input_size,
        config.anti_spoof.v2_input_size,
        config.anti_spoof.v2_crop_scale
    );
    println!("    {}", config.models.anti_spoof_v2_path);
    println!(
        "  Anti-spoof V1SE: {} | input: {}x{} | scale: {:.1}",
        if Path::new(&config.models.anti_spoof_v1se_path).exists() {
            "OK Found"
        } else {
            "X Missing"
        },
        config.anti_spoof.v1se_input_size,
        config.anti_spoof.v1se_input_size,
        config.anti_spoof.v1se_crop_scale
    );
    println!("    {}", config.models.anti_spoof_v1se_path);

    println!("\nCamera:");
    print!("  Device: ");
    if check_camera(&config.video.device) {
        println!("OK {}", config.video.device);
    } else {
        println!("X Not available: {}", config.video.device);
    }

    println!("\nPAM Module:");
    print!("  Library: ");
    let pam_path = "/usr/lib/security/pam_facepass.so";
    if Path::new(pam_path).exists() {
        println!("OK Installed");
    } else {
        println!("X Not installed");
    }

    println!("\nSecurity:");
    print!("  SSH Session: ");
    if is_ssh_session() {
        println!("Yes (face auth will be skipped)");
    } else {
        println!("No");
    }

    print!("  Laptop Lid: ");
    if is_lid_closed() {
        println!("Closed (face auth will be skipped)");
    } else {
        println!("Open");
    }

    println!("\nRegistered Users:");
    let storage = FaceStorage::new(&config.storage.data_dir)?;
    let users = storage.list_users()?;

    if users.is_empty() {
        println!("  (none)");
    } else {
        for user in &users {
            let metadata = storage.get_metadata(user)?;
            let default_group = metadata
                .default_group()
                .map(|group| group.name.as_str())
                .unwrap_or("(none)");
            println!(
                "  {} ({} group(s), {} face(s), default: {})",
                user,
                metadata.groups.len(),
                metadata.total_face_count(),
                default_group
            );
        }
    }

    println!("\nConfiguration:");
    print!("  Config file: ");
    if let Some(path) = config_source {
        println!("OK {}", path.display());
    } else {
        println!("X Not found (using defaults)");
    }

    print!("  Data directory: ");
    if Path::new(&config.storage.data_dir).exists() {
        println!("OK {}", config.storage.data_dir);
    } else {
        println!("X Not found");
    }

    Ok(())
}

fn is_daemon_running(socket_path: &str) -> bool {
    if !Path::new(socket_path).exists() {
        return false;
    }

    std::os::unix::net::UnixStream::connect(socket_path).is_ok()
}
