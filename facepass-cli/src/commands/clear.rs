//! Clear all faces command

use super::get_username;
use anyhow::Result;
use facepass_core::{config::Config, storage::FaceStorage};
use std::io::{self, Write};

pub fn run(config_path: &str, user: Option<String>, force: bool) -> Result<()> {
    let username = get_username(user)?;
    let config = Config::load_with_fallback(config_path)?;

    let storage = FaceStorage::new(&config.storage.data_dir)?;
    let count = storage.face_count(&username)?;

    if count == 0 {
        println!("No faces registered for user '{}'", username);
        return Ok(());
    }

    if !force {
        print!(
            "This will delete all {} face(s) for '{}'. Continue? [y/N]: ",
            count, username
        );
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled.");
            return Ok(());
        }
    }

    storage.delete_all_faces(&username)?;

    println!("✓ All faces cleared for user '{}'", username);

    Ok(())
}
