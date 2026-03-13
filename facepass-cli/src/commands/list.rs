//! List faces command

use super::get_username;
use anyhow::Result;
use facepass_core::{config::Config, storage::FaceStorage};

pub fn run(config_path: &str, user: Option<String>) -> Result<()> {
    let username = get_username(user)?;
    let config = Config::load_with_fallback(config_path)?;

    let storage = FaceStorage::new(&config.storage.data_dir)?;
    let faces = storage.load_all_faces(&username)?;

    if faces.is_empty() {
        println!("No faces registered for user '{}'", username);
        return Ok(());
    }

    println!("Registered faces for '{}':\n", username);
    println!("{:>5}  {:36}  {:20}  {}", "Index", "ID", "Label", "Created");
    println!("{}", "-".repeat(80));

    for (index, face) in faces.iter().enumerate() {
        let created = chrono_format(face.created_at);
        println!("{:>5}  {}  {:20}  {}", index, face.id, face.label, created);
    }

    println!(
        "\nTotal: {} face(s) (max: {})",
        faces.len(),
        config.recognition.max_faces_per_user
    );

    Ok(())
}

fn chrono_format(timestamp: i64) -> String {
    use std::time::{Duration, UNIX_EPOCH};

    let datetime = UNIX_EPOCH + Duration::from_secs(timestamp as u64);
    let datetime: chrono::DateTime<chrono::Local> = datetime.into();
    datetime.format("%Y-%m-%d %H:%M").to_string()
}
