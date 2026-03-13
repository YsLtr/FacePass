//! Enable face authentication (start daemon)

use anyhow::Result;
use std::process::Command;

pub fn run() -> Result<()> {
    println!("Enabling FacePass face authentication...\n");

    // Enable and start the systemd service
    let enable_result = Command::new("systemctl")
        .args(["enable", "facepass.service"])
        .output();

    match enable_result {
        Ok(output) => {
            if output.status.success() {
                println!("✓ Service enabled for automatic start");
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("Warning: Could not enable service: {}", stderr.trim());
            }
        }
        Err(e) => {
            eprintln!("Warning: systemctl not available: {}", e);
        }
    }

    let start_result = Command::new("systemctl")
        .args(["start", "facepass.service"])
        .output();

    match start_result {
        Ok(output) => {
            if output.status.success() {
                println!("✓ Daemon started");
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("Error: Could not start service: {}", stderr.trim());
                return Err(anyhow::anyhow!("Failed to start daemon"));
            }
        }
        Err(e) => {
            eprintln!("Error: systemctl not available: {}", e);
            println!("\nTo start manually, run:");
            println!("  sudo facepass-daemon &");
            return Err(anyhow::anyhow!("systemctl not available"));
        }
    }

    println!("\nFacePass is now enabled!");
    println!("Face authentication will be used for sudo and other PAM-enabled services.");
    println!("\nTo add your face, run:");
    println!("  sudo facepass add");

    Ok(())
}
