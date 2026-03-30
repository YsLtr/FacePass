//! Cancel the current face authentication attempt via IPC

use anyhow::{anyhow, Context, Result};
use facepass_core::{
    config::Config,
    models::{AuthResponse, CancelRequest},
};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

pub fn run(config_path: &str) -> Result<()> {
    let config = Config::load_with_fallback(config_path)?;
    let request = CancelRequest::new();

    let mut stream = UnixStream::connect(&config.daemon.socket_path).with_context(|| {
        format!(
            "Failed to connect to daemon at {}",
            config.daemon.socket_path
        )
    })?;

    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;

    let request_json = serde_json::to_string(&request)?;
    stream.write_all(request_json.as_bytes())?;
    stream.write_all(b"\n")?;
    stream.flush()?;

    let mut reader = BufReader::new(stream);
    let mut response_line = String::new();
    reader.read_line(&mut response_line)?;

    if response_line.trim().is_empty() {
        return Err(anyhow!("Daemon returned an empty response"));
    }

    let response: AuthResponse = serde_json::from_str(&response_line)?;
    if response.success {
        println!("{}", response.message);
        Ok(())
    } else {
        Err(anyhow!(response.message))
    }
}
