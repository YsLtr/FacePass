//! Remove user, group, or face data

use super::{ensure_user_access, resolve_group_for_read, resolve_username};
use anyhow::{anyhow, Result};
use facepass_core::{config::Config, storage::FaceStorage};

pub fn run(
    config_path: &str,
    user: Option<String>,
    group: Option<String>,
    face: Vec<String>,
) -> Result<()> {
    let user_scope_selected = user.is_some();
    let config = Config::load_with_fallback(config_path)?;
    let storage = FaceStorage::new(&config.storage.data_dir)?;
    let username = resolve_username(&storage, user.as_deref())?;
    ensure_user_access(&username, "remove data for other users")?;

    if !user_scope_selected && group.is_none() && face.is_empty() {
        return Err(anyhow!(
            "Nothing to remove. Specify --face, --group, or --user."
        ));
    }

    if user_scope_selected && group.is_none() && face.is_empty() {
        if storage.user_dir(&username).exists() {
            storage.delete_user(&username)?;
            println!("✓ Removed all face data for user '{}'", username);
            return Ok(());
        }

        return Err(anyhow!("No face data stored for user '{}'", username));
    }

    if let Some(group_selector) = group.as_deref() {
        let target_group = storage.resolve_group(&username, group_selector)?;

        if face.is_empty() {
            let was_default = storage
                .get_default_group(&username)
                .ok()
                .map(|default_group| default_group.id == target_group.id)
                .unwrap_or(false);
            storage.delete_group(&username, &target_group.id)?;

            println!("✓ Removed face group '{}'", target_group.name);
            if was_default {
                if let Ok(new_default) = storage.get_default_group(&username) {
                    println!("  New default group: {}", new_default.name);
                } else {
                    println!("  User now has no face groups");
                }
            }
            return Ok(());
        }
    }

    let target_group = resolve_group_for_read(&storage, &username, group.as_deref())?;
    let faces = storage.resolve_faces_in_group(&username, &target_group.id, &face)?;
    if faces.is_empty() {
        return Err(anyhow::anyhow!(
            "No matching faces found in group '{}'",
            target_group.name
        ));
    }

    println!(
        "Removing {} face(s) from group '{}':",
        faces.len(),
        target_group.name
    );
    for matched in &faces {
        println!("  {}  {}", matched.id, matched.label);
    }

    let ids: Vec<_> = faces.iter().map(|matched| matched.id).collect();
    let deleted = storage.delete_faces(&username, &target_group.id, &ids)?;
    let remaining = storage
        .load_faces_in_group(&username, &target_group.id)?
        .len();

    println!("\n✓ Removed {} face(s)", deleted);
    println!("  User: {}", username);
    println!("  Group: {}", target_group.name);
    println!("  Remaining group faces: {}", remaining);

    Ok(())
}
