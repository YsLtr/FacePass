//! Disable face authentication (stop daemon)

use anyhow::Result;
use std::process::Command;

pub fn run() -> Result<()> {
    println!("Disabling FacePass face authentication...\n");

    // Stop and disable the systemd service
    let stop_result = Command::new("systemctl")
        .args(["stop", "facepass.service"])
        .output();

    match stop_result {
        Ok(output) => {
            if output.status.success() {
                println!("✓ Daemon stopped");
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                // Not an error if service wasn't running
                if !stderr.contains("not loaded") {
                    eprintln!("Warning: {}", stderr.trim());
                }
            }
        }
        Err(e) => {
            eprintln!("Warning: systemctl not available: {}", e);
        }
    }

    let disable_result = Command::new("systemctl")
        .args(["disable", "facepass.service"])
        .output();

    match disable_result {
        Ok(output) => {
            if output.status.success() {
                println!("✓ Service disabled from automatic start");
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stderr.contains("not loaded") {
                    eprintln!("Warning: {}", stderr.trim());
                }
            }
        }
        Err(e) => {
            eprintln!("Warning: systemctl not available: {}", e);
        }
    }

    println!("\nFacePass has been disabled.");
    println!("You will need to use your password for authentication.");
    println!("\nTo re-enable, run:");
    println!("  sudo facepass enable");

    Ok(())
}
