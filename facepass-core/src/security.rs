//! Security checks module

use crate::config::SecurityConfig;
use crate::error::{Error, Result};
use std::env;
use std::fs;
use std::path::Path;

/// Perform security checks before authentication
pub fn check_security(config: &SecurityConfig) -> Result<()> {
    // Check SSH session
    if config.ignore_ssh && is_ssh_session() {
        return Err(Error::SecurityCheck(
            "Face authentication disabled for SSH sessions".to_string(),
        ));
    }

    // Check laptop lid
    if config.ignore_closed_lid && is_lid_closed() {
        return Err(Error::SecurityCheck(
            "Face authentication disabled when lid is closed".to_string(),
        ));
    }

    Ok(())
}

/// Check if running in an SSH session
pub fn is_ssh_session() -> bool {
    // Check common SSH environment variables
    env::var("SSH_CONNECTION").is_ok()
        || env::var("SSH_CLIENT").is_ok()
        || env::var("SSH_TTY").is_ok()
}

/// Check if laptop lid is closed
pub fn is_lid_closed() -> bool {
    // Check ACPI lid state
    let lid_paths = [
        "/proc/acpi/button/lid/LID/state",
        "/proc/acpi/button/lid/LID0/state",
        "/proc/acpi/button/lid/LID1/state",
    ];

    for path in &lid_paths {
        if let Ok(content) = fs::read_to_string(path) {
            // State file contains "state:      open" or "state:      closed"
            if content.contains("closed") {
                return true;
            }
            if content.contains("open") {
                return false;
            }
        }
    }

    // Also check /sys/class/power_supply for some systems
    if let Ok(entries) = fs::read_dir("/sys/class/power_supply") {
        for entry in entries.flatten() {
            let lid_state = entry.path().join("lid_state");
            if lid_state.exists() {
                if let Ok(content) = fs::read_to_string(&lid_state) {
                    if content.trim() == "closed" {
                        return true;
                    }
                }
            }
        }
    }

    // Default to open if we can't determine the state
    false
}

/// Check if the user exists on the system
pub fn user_exists(username: &str) -> bool {
    // Try to read /etc/passwd
    if let Ok(content) = fs::read_to_string("/etc/passwd") {
        for line in content.lines() {
            if let Some(name) = line.split(':').next() {
                if name == username {
                    return true;
                }
            }
        }
    }

    false
}

/// Get the current username
pub fn get_current_user() -> Option<String> {
    let pkexec_user = env::var("PKEXEC_UID")
        .ok()
        .and_then(|uid_str| uid_str.parse::<u32>().ok())
        .and_then(uid_to_username);
    let real_uid_user = uid_to_username(unsafe { libc::getuid() });

    resolve_current_user_from_sources(
        env::var("SUDO_USER").ok(),
        env::var("DOAS_USER").ok(),
        pkexec_user,
        real_uid_user,
        env::var("USER").ok(),
        env::var("LOGNAME").ok(),
    )
}

/// Convert UID to username
fn uid_to_username(uid: u32) -> Option<String> {
    if let Ok(content) = fs::read_to_string("/etc/passwd") {
        for line in content.lines() {
            let parts: Vec<&str> = line.split(':').collect();
            if parts.len() >= 3 {
                if let Ok(line_uid) = parts[2].parse::<u32>() {
                    if line_uid == uid {
                        return Some(parts[0].to_string());
                    }
                }
            }
        }
    }
    None
}

fn resolve_current_user_from_sources(
    sudo_user: Option<String>,
    doas_user: Option<String>,
    pkexec_user: Option<String>,
    real_uid_user: Option<String>,
    user_env: Option<String>,
    logname_env: Option<String>,
) -> Option<String> {
    sudo_user
        .filter(|user| !user.is_empty())
        .or_else(|| doas_user.filter(|user| !user.is_empty()))
        .or_else(|| pkexec_user.filter(|user| !user.is_empty()))
        .or_else(|| real_uid_user.filter(|user| !user.is_empty()))
        .or_else(|| user_env.filter(|user| !user.is_empty()))
        .or_else(|| logname_env.filter(|user| !user.is_empty()))
}

/// Check if running as root
pub fn is_root() -> bool {
    unsafe { libc::geteuid() == 0 }
}

/// Check if a camera device exists and is accessible
pub fn check_camera_access(device: &str) -> Result<()> {
    let path = Path::new(device);

    if !path.exists() {
        return Err(Error::Camera(format!(
            "Camera device not found: {}",
            device
        )));
    }

    // Check if we can read the device
    match fs::metadata(path) {
        Ok(meta) => {
            use std::os::unix::fs::MetadataExt;
            let mode = meta.mode();

            // Check if it's a character device
            if mode & 0o170000 != 0o020000 {
                return Err(Error::Camera(format!("Not a character device: {}", device)));
            }

            Ok(())
        }
        Err(e) => Err(Error::Camera(format!(
            "Cannot access camera device {}: {}",
            device, e
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssh_detection() {
        // This test will vary based on environment
        let result = is_ssh_session();
        // Just ensure it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_lid_detection() {
        // This test will vary based on hardware
        let result = is_lid_closed();
        // Just ensure it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_user_exists() {
        // Root should always exist
        assert!(user_exists("root"));
        // Random user should not exist
        assert!(!user_exists("nonexistent_user_12345"));
    }

    #[test]
    fn test_get_current_user() {
        let user = get_current_user();
        // Should return some user in normal circumstances
        assert!(user.is_some() || env::var("USER").is_err());
    }

    #[test]
    fn test_resolve_current_user_prefers_invoker_sources() {
        let user = resolve_current_user_from_sources(
            Some("sudo-user".to_string()),
            Some("doas-user".to_string()),
            Some("pkexec-user".to_string()),
            Some("real-user".to_string()),
            Some("env-user".to_string()),
            Some("logname-user".to_string()),
        );

        assert_eq!(user.as_deref(), Some("sudo-user"));
    }

    #[test]
    fn test_resolve_current_user_falls_back_to_real_uid_before_env() {
        let user = resolve_current_user_from_sources(
            None,
            None,
            None,
            Some("real-user".to_string()),
            Some("env-user".to_string()),
            Some("logname-user".to_string()),
        );

        assert_eq!(user.as_deref(), Some("real-user"));
    }
}
