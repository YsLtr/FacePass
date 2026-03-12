//! Camera listing command

use anyhow::Result;
use facepass_core::camera::list_cameras;

pub fn run(_verbose: bool) -> Result<()> {
    println!("Scanning for available cameras...\n");

    let cameras = list_cameras()?;

    if cameras.is_empty() {
        println!("No cameras found.");
        println!("\nMake sure:");
        println!("  1. A webcam is connected");
        println!("  2. You have permission to access /dev/video* devices");
        println!("  3. The v4l2 kernel modules are loaded");
        return Ok(());
    }

    println!("Available cameras:\n");
    for (index, device) in cameras.iter().enumerate() {
        println!("  [{}] {}", index, device);
    }

    println!("\nFound {} camera(s)", cameras.len());
    println!("\nTo use a specific camera, edit the 'device' field in");
    println!("  /etc/facepass/config.toml");

    Ok(())
}
