//! CLI command modules

pub mod add;
pub mod cameras;
pub mod cancel;
pub mod config;
pub mod disable;
pub mod enable;
pub mod list;
pub mod remove;
pub mod status;
pub mod test;

use anyhow::{anyhow, Result};
use facepass_core::{models::FaceGroupSummary, security::get_current_user, storage::FaceStorage};
use std::io;

/// Get the current CLI target user based on invoker context.
pub fn current_username() -> Result<String> {
    get_current_user()
        .ok_or_else(|| anyhow!("Could not determine username. Please specify with --user"))
}

fn is_legacy_prefixed_numeric_selector(selector: &str) -> bool {
    selector
        .strip_prefix('@')
        .or_else(|| selector.strip_prefix('#'))
        .map(|rest| !rest.is_empty() && rest.chars().all(|ch| ch.is_ascii_digit()))
        .unwrap_or(false)
}

/// Resolve a user selector (index or exact name), defaulting to the current user.
pub fn resolve_username(storage: &FaceStorage, user_selector: Option<&str>) -> Result<String> {
    match user_selector {
        Some(selector)
            if !selector.is_empty() && selector.chars().all(|ch| ch.is_ascii_digit()) =>
        {
            Ok(storage.resolve_user(selector)?)
        }
        Some(selector) if is_legacy_prefixed_numeric_selector(selector) => Err(anyhow!(
            "Invalid user selector '{}'. Use bare numeric indexes like '{}'",
            selector,
            selector[1..].to_string()
        )),
        Some(selector) => Ok(selector.to_string()),
        None => current_username(),
    }
}

/// Require root privileges when operating on another user's data.
pub fn ensure_user_access(target_user: &str, action: &str) -> Result<()> {
    if unsafe { libc::geteuid() } == 0 {
        return Ok(());
    }

    let current_user = current_username()?;
    if target_user == current_user {
        return Ok(());
    }

    Err(anyhow!("Root privileges required to {}", action))
}

/// Resolve the target group for read/delete operations.
pub fn resolve_group_for_read(
    storage: &FaceStorage,
    username: &str,
    group_selector: Option<&str>,
) -> Result<FaceGroupSummary> {
    match group_selector {
        Some(selector) => Ok(storage.resolve_group(username, selector)?),
        None => Ok(storage.get_default_group(username)?),
    }
}

/// Resolve the target group for add operations, creating it by name if needed.
pub fn resolve_group_for_add(
    storage: &FaceStorage,
    username: &str,
    group_selector: Option<&str>,
) -> Result<FaceGroupSummary> {
    let Some(selector) = group_selector else {
        return Ok(storage.ensure_default_group(username)?);
    };

    if !selector.is_empty() && selector.chars().all(|ch| ch.is_ascii_digit()) {
        return Ok(storage.resolve_group(username, selector)?);
    }
    if is_legacy_prefixed_numeric_selector(selector) {
        return Err(anyhow!(
            "Invalid group selector '{}'. Use bare numeric indexes like '{}'",
            selector,
            selector[1..].to_string()
        ));
    }

    if let Some(group) = storage.find_group_by_name(username, selector)? {
        return Ok(group);
    }

    Ok(storage.create_group(username, selector)?)
}

pub enum CommandKey {
    Quit,
    Action,
}

pub struct CommandInput {
    fd: i32,
    original_termios: Option<libc::termios>,
    original_flags: i32,
}

impl CommandInput {
    pub fn capture_single_keys() -> Result<Self> {
        Self::capture(true)
    }

    fn capture(raw_mode: bool) -> Result<Self> {
        let fd = libc::STDIN_FILENO;
        let original_flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if original_flags < 0 {
            return Err(io::Error::last_os_error().into());
        }

        let original_termios = if raw_mode && unsafe { libc::isatty(fd) } == 1 {
            let mut termios = unsafe { std::mem::zeroed::<libc::termios>() };
            if unsafe { libc::tcgetattr(fd, &mut termios) } != 0 {
                return Err(io::Error::last_os_error().into());
            }

            let mut raw = termios;
            raw.c_lflag &= !(libc::ICANON | libc::ECHO);
            raw.c_cc[libc::VMIN] = 0;
            raw.c_cc[libc::VTIME] = 0;

            if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &raw) } != 0 {
                return Err(io::Error::last_os_error().into());
            }

            Some(termios)
        } else {
            None
        };

        if unsafe { libc::fcntl(fd, libc::F_SETFL, original_flags | libc::O_NONBLOCK) } != 0 {
            if let Some(termios) = original_termios {
                unsafe {
                    libc::tcsetattr(fd, libc::TCSANOW, &termios);
                }
            }
            return Err(io::Error::last_os_error().into());
        }

        Ok(Self {
            fd,
            original_termios,
            original_flags,
        })
    }

    pub fn poll_key(&self) -> Result<Option<CommandKey>> {
        let mut byte = [0u8; 1];
        let read = unsafe { libc::read(self.fd, byte.as_mut_ptr().cast(), 1) };

        if read == 0 {
            return Ok(None);
        }

        if read < 0 {
            let err = io::Error::last_os_error();
            if let Some(code) = err.raw_os_error() {
                if code == libc::EAGAIN || code == libc::EWOULDBLOCK {
                    return Ok(None);
                }
            }
            return Err(err.into());
        }

        let key = match byte[0] {
            b'q' | b'Q' => Some(CommandKey::Quit),
            b'a' | b'A' | b'\n' | b'\r' => Some(CommandKey::Action),
            _ => None,
        };

        Ok(key)
    }
}

impl Drop for CommandInput {
    fn drop(&mut self) {
        unsafe {
            libc::fcntl(self.fd, libc::F_SETFL, self.original_flags);
            if let Some(termios) = self.original_termios {
                libc::tcsetattr(self.fd, libc::TCSANOW, &termios);
            }
        }
    }
}
