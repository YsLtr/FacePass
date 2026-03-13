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
use std::io;

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
    env::var("USER")
        .or_else(|_| env::var("LOGNAME"))
        .map_err(|_| anyhow::anyhow!("Could not determine username. Please specify with --user"))
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
    pub fn capture_single_keys() -> anyhow::Result<Self> {
        let fd = libc::STDIN_FILENO;
        let original_flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        if original_flags < 0 {
            return Err(io::Error::last_os_error().into());
        }

        let original_termios = if unsafe { libc::isatty(fd) } == 1 {
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

    pub fn poll_key(&self) -> anyhow::Result<Option<CommandKey>> {
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
            b'a' | b'A' => Some(CommandKey::Action),
            b'\n' | b'\r' => None,
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
