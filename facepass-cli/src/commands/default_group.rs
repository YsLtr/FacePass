//! Change the default face group for a user

use super::get_username;
use anyhow::Result;
use facepass_core::{config::Config, storage::FaceStorage};

pub fn run(config_path: &str, user: Option<String>, group: String) -> Result<()> {
    let username = get_username(user)?;
    let config = Config::load_with_fallback(config_path)?;
    let storage = FaceStorage::new(&config.storage.data_dir)?;
    let target_group = storage.resolve_group(&username, &group)?;

    storage.set_default_group(&username, &target_group.id)?;

    println!("✓ Default face group updated");
    println!("  User: {}", username);
    println!("  Group: {}", target_group.name);
    println!("  Group ID: {}", target_group.id);

    Ok(())
}
