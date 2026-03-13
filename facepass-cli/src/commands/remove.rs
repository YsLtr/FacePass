//! Remove face command

use super::get_username;
use anyhow::Result;
use facepass_core::{config::Config, storage::FaceStorage};

pub fn run(config_path: &str, user: Option<String>, index: usize) -> Result<()> {
    let username = get_username(user)?;
    let config = Config::load_with_fallback(config_path)?;

    let storage = FaceStorage::new(&config.storage.data_dir)?;
    let faces = storage.load_all_faces(&username)?;

    if faces.is_empty() {
        return Err(anyhow::anyhow!(
            "No faces registered for user '{}'",
            username
        ));
    }

    if index >= faces.len() {
        return Err(anyhow::anyhow!(
            "Invalid index {}. Valid range: 0-{}",
            index,
            faces.len() - 1
        ));
    }

    let face = &faces[index];

    println!("Removing face:");
    println!("  ID: {}", face.id);
    println!("  Label: {}", face.label);

    storage.delete_face(&username, &face.id)?;

    println!("\n✓ Face removed successfully!");

    let remaining = storage.face_count(&username)?;
    println!("  Remaining faces: {}", remaining);

    Ok(())
}
