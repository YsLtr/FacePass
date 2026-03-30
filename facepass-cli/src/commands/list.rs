//! List users, face groups, and faces

use super::{get_username, resolve_group_for_read};
use anyhow::Result;
use facepass_core::{config::Config, storage::FaceStorage};

pub fn run(config_path: &str, user: Option<String>, group: Option<String>, depth: u8) -> Result<()> {
    let username = get_username(user)?;
    let config = Config::load_with_fallback(config_path)?;
    let storage = FaceStorage::new(&config.storage.data_dir)?;
    let metadata = storage.get_metadata(&username)?;

    if metadata.groups.is_empty() {
        println!("No face groups registered for user '{}'", username);
        return Ok(());
    }

    let default_group = metadata.default_group().cloned();
    let all_groups = metadata.groups.clone();
    let groups_to_show = if let Some(group_selector) = group.as_deref() {
        let selected = resolve_group_for_read(&storage, &username, Some(group_selector))?;
        all_groups
            .iter()
            .cloned()
            .enumerate()
            .filter(|(_, group)| group.id == selected.id)
            .collect::<Vec<_>>()
    } else {
        all_groups.into_iter().enumerate().collect::<Vec<_>>()
    };

    println!("Face data for '{}':", username);
    println!("  Groups: {}", metadata.groups.len());
    println!("  Total faces: {}", metadata.total_face_count());
    println!(
        "  Default group: {}",
        default_group
            .as_ref()
            .map(|group| group.name.as_str())
            .unwrap_or("(none)")
    );
    println!(
        "  Max faces per group: {}",
        config.recognition.max_faces_per_group
    );

    if depth == 1 {
        return Ok(());
    }

    for (group_index, group) in groups_to_show {
        let is_default = default_group
            .as_ref()
            .map(|default_group| default_group.id == group.id)
            .unwrap_or(false);
        println!(
            "\n[{}] {}{} ({} face(s))",
            group_index,
            group.name,
            if is_default { " [default]" } else { "" },
            group.face_count
        );

        if depth < 3 {
            continue;
        }

        let faces = storage.load_faces_in_group(&username, &group.id)?;
        if faces.is_empty() {
            println!("  (empty)");
            continue;
        }

        println!("  {:>5}  {:36}  {:20}  {}", "Index", "ID", "Label", "Created");
        for (face_index, face) in faces.iter().enumerate() {
            println!(
                "  {:>5}  {}  {:20}  {}",
                face_index,
                face.id,
                face.label,
                chrono_format(face.created_at)
            );
        }
    }

    Ok(())
}

fn chrono_format(timestamp: i64) -> String {
    use std::time::{Duration, UNIX_EPOCH};

    let datetime = UNIX_EPOCH + Duration::from_secs(timestamp as u64);
    let datetime: chrono::DateTime<chrono::Local> = datetime.into();
    datetime.format("%Y-%m-%d %H:%M").to_string()
}
