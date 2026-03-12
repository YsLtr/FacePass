//! Status check command

use anyhow::Result;
use facepass_core::{
    camera::check_camera,
    config::Config,
    security::{is_lid_closed, is_ssh_session},
    storage::FaceStorage,
};
use std::path::Path;

pub fn run(config_path: &str, _verbose: bool) -> Result<()> {
    println!("FacePass System Status");
    println!("======================\n");

    let config = Config::load(config_path).unwrap_or_default();

    // Check daemon status
    print!("Daemon: ");
    if is_daemon_running(&config.daemon.socket_path) {
        println!("✓ Running");
    } else {
        println!("✗ Not running");
    }

    // Check socket
    print!("Socket: ");
    if Path::new(&config.daemon.socket_path).exists() {
        println!("✓ {}", config.daemon.socket_path);
    } else {
        println!("✗ Not found");
    }

    // Check models
    println!("\nModels:");
    print!("  YuNet: ");
    if Path::new(&config.models.yunet_path).exists() {
        println!("✓ Found");
    } else {
        println!("✗ Not found: {}", config.models.yunet_path);
    }

    print!("  SFace: ");
    if Path::new(&config.models.sface_path).exists() {
        println!("✓ Found");
    } else {
        println!("✗ Not found: {}", config.models.sface_path);
    }

    print!("  Anti-spoofing: ");
    if config.anti_spoof.enabled {
        println!(
            "✓ Enabled (threshold: {:.2}, input: {}x{}, scale: {:.1})",
            config.anti_spoof.threshold,
            config.anti_spoof.input_size,
            config.anti_spoof.input_size,
            config.anti_spoof.crop_scale
        );
    } else {
        println!("✗ Disabled");
    }

    print!("  Anti-spoof model: ");
    if Path::new(&config.models.anti_spoof_path).exists() {
        println!("✓ Found");
    } else if config.anti_spoof.enabled {
        println!(
            "! Not found: {} (will fall back to face recognition only)",
            config.models.anti_spoof_path
        );
    } else {
        println!("✗ Not found: {}", config.models.anti_spoof_path);
    }

    // Check camera
    println!("\nCamera:");
    print!("  Device: ");
    if check_camera(&config.video.device) {
        println!("✓ {}", config.video.device);
    } else {
        println!("✗ Not available: {}", config.video.device);
    }

    // Check PAM module
    println!("\nPAM Module:");
    print!("  Library: ");
    let pam_path = "/usr/lib/security/pam_facepass.so";
    if Path::new(pam_path).exists() {
        println!("✓ Installed");
    } else {
        println!("✗ Not installed");
    }

    // Check security conditions
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

    // Check registered users
    println!("\nRegistered Users:");
    let storage = FaceStorage::new(&config.storage.data_dir)?;
    let users = storage.list_users()?;

    if users.is_empty() {
        println!("  (none)");
    } else {
        for user in &users {
            let count = storage.face_count(user)?;
            println!("  {} ({} face(s))", user, count);
        }
    }

    // Check config file
    println!("\nConfiguration:");
    print!("  Config file: ");
    if Path::new(config_path).exists() {
        println!("✓ {}", config_path);
    } else {
        println!("✗ Not found (using defaults)");
    }

    print!("  Data directory: ");
    if Path::new(&config.storage.data_dir).exists() {
        println!("✓ {}", config.storage.data_dir);
    } else {
        println!("✗ Not found");
    }

    Ok(())
}

fn is_daemon_running(socket_path: &str) -> bool {
    // Check if socket exists and is connectable
    if !Path::new(socket_path).exists() {
        return false;
    }

    // Try to connect
    std::os::unix::net::UnixStream::connect(socket_path).is_ok()
}
