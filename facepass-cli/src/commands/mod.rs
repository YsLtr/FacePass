//! CLI command modules

pub mod add;
pub mod cameras;
pub mod clear;
pub mod config;
pub mod disable;
pub mod enable;
pub mod list;
pub mod remove;
pub mod status;
pub mod test;

use std::env;

/// Get the target username (from argument or current user)
pub fn get_username(user_arg: Option<String>) -> anyhow::Result<String> {
    if let Some(user) = user_arg {
        return Ok(user);
    }

    // Try SUDO_USER first
    if let Ok(user) = env::var("SUDO_USER") {
        return Ok(user);
    }

    // Try DOAS_USER
    if let Ok(user) = env::var("DOAS_USER") {
        return Ok(user);
    }

    // Fall back to USER
    env::var("USER").or_else(|_| env::var("LOGNAME")).map_err(|_| {
        anyhow::anyhow!("Could not determine username. Please specify with --user")
    })
}
