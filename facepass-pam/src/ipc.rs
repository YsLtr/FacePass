//! IPC client for communicating with facepass-daemon

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

/// Default socket path
const DEFAULT_SOCKET_PATH: &str = "/run/facepass/facepass.sock";

/// Authentication request
#[derive(Serialize)]
struct AuthRequest {
    msg_type: String,
    username: String,
    source: String,
    timeout: u32,
}

/// Authentication response
#[derive(Deserialize)]
pub struct AuthResponse {
    pub success: bool,
    pub message: String,
    pub confidence: Option<f64>,
    pub matched_label: Option<String>,
}

/// Error type for IPC operations
#[derive(Debug)]
pub struct IpcError(pub String);

impl std::fmt::Display for IpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "IPC error: {}", self.0)
    }
}

impl std::error::Error for IpcError {}

/// Authenticate a user via the daemon
pub fn authenticate(username: &str, source: &str, timeout: u32) -> Result<AuthResponse, IpcError> {
    // Connect to the daemon
    let mut stream = UnixStream::connect(DEFAULT_SOCKET_PATH)
        .map_err(|e| IpcError(format!("Failed to connect to daemon: {}", e)))?;

    // Set timeouts
    stream
        .set_read_timeout(if timeout == 0 {
            None
        } else {
            Some(Duration::from_secs((timeout + 2) as u64))
        })
        .map_err(|e| IpcError(format!("Failed to set timeout: {}", e)))?;

    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| IpcError(format!("Failed to set timeout: {}", e)))?;

    // Create request
    let request = AuthRequest {
        msg_type: "auth".to_string(),
        username: username.to_string(),
        source: source.to_string(),
        timeout,
    };

    // Send request
    let request_json = serde_json::to_string(&request)
        .map_err(|e| IpcError(format!("Failed to serialize request: {}", e)))?;

    stream
        .write_all(request_json.as_bytes())
        .map_err(|e| IpcError(format!("Failed to send request: {}", e)))?;

    stream
        .write_all(b"\n")
        .map_err(|e| IpcError(format!("Failed to send newline: {}", e)))?;

    stream
        .flush()
        .map_err(|e| IpcError(format!("Failed to flush: {}", e)))?;

    // Read response
    let mut reader = BufReader::new(stream);
    let mut response_line = String::new();

    reader
        .read_line(&mut response_line)
        .map_err(|e| IpcError(format!("Failed to read response: {}", e)))?;

    // Parse response
    let response: AuthResponse = serde_json::from_str(&response_line)
        .map_err(|e| IpcError(format!("Failed to parse response: {}", e)))?;

    Ok(response)
}
