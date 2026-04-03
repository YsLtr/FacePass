//! List users, face groups, and faces

use super::{ensure_user_access, resolve_group_for_read, resolve_username};
use anyhow::{anyhow, Result};
use facepass_core::{config::Config, storage::FaceStorage};

pub fn run(
    config_path: &str,
    user: Option<String>,
    group: Option<String>,
    all: bool,
    depth: u8,
) -> Result<()> {
    let config = Config::load_with_fallback(config_path)?;
    let storage = FaceStorage::new(&config.storage.data_dir)?;

    if all {
        return list_all_users(&storage, &config, user, group, depth);
    }

    let username = resolve_username(&storage, user.as_deref())?;
    ensure_user_access(&username, "list data for other users")?;
    print_user_tree(&storage, &config, &username, group.as_deref(), depth, None)
}

fn list_all_users(
    storage: &FaceStorage,
    config: &Config,
    user: Option<String>,
    group: Option<String>,
    depth: u8,
) -> Result<()> {
    if unsafe { libc::geteuid() } != 0 {
        return Err(anyhow!(
            "Root privileges required to list data for all users"
        ));
    }
    if user.is_some() || group.is_some() {
        return Err(anyhow!("--all cannot be combined with --user or --group"));
    }

    let users = storage.list_users()?;
    if users.is_empty() {
        println!("No registered users with face data");
        return Ok(());
    }

    let mut total_faces = 0usize;
    for username in &users {
        total_faces += storage.total_face_count(username)?;
    }

    println!("Face data for all registered users:");
    println!("  Users: {}", users.len());
    println!("  Total faces: {}", total_faces);
    println!(
        "  Max faces per group: {}",
        config.recognition.max_faces_per_group
    );

    for (user_index, username) in users.iter().enumerate() {
        println!();
        print_user_tree(storage, config, username, None, depth, Some(user_index))?;
    }

    Ok(())
}

fn print_user_tree(
    storage: &FaceStorage,
    config: &Config,
    username: &str,
    group_selector: Option<&str>,
    depth: u8,
    user_index: Option<usize>,
) -> Result<()> {
    let metadata = storage.get_metadata(username)?;

    if let Some(index) = user_index {
        println!("[{}] {}", index, username);
    } else {
        println!("Face data for '{}':", username);
    }

    if metadata.groups.is_empty() {
        println!("  Groups: 0");
        println!("  Total faces: 0");
        println!("  Default group: (none)");
        println!(
            "  Max faces per group: {}",
            config.recognition.max_faces_per_group
        );
        return Ok(());
    }

    let default_group = metadata.default_group().cloned();
    let all_groups = metadata.groups.clone();
    let groups_to_show = if let Some(selector) = group_selector {
        let selected = resolve_group_for_read(storage, username, Some(selector))?;
        all_groups
            .iter()
            .cloned()
            .enumerate()
            .filter(|(_, group)| group.id == selected.id)
            .collect::<Vec<_>>()
    } else {
        all_groups.into_iter().enumerate().collect::<Vec<_>>()
    };

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
            "\n  [{}] {}{} ({} face(s))",
            group_index,
            group.name,
            if is_default { " [default]" } else { "" },
            group.face_count
        );

        if depth < 3 {
            continue;
        }

        let faces = storage.load_faces_in_group(username, &group.id)?;
        if faces.is_empty() {
            println!("    (empty)");
            continue;
        }

        println!("    {:>8}  {:36}  {:20}  Created", "Index", "ID", "Label");
        for (face_index, face) in faces.iter().enumerate() {
            println!(
                "    {:>8}  {}  {:20}  {}",
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
