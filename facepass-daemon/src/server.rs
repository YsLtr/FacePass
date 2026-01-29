//! Unix socket server for the daemon

use crate::auth;
use anyhow::Result;
use facepass_core::config::Config;
use facepass_core::models::{AuthRequest, AuthResponse};
use log::{debug, error, info, warn};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::broadcast;

/// Run the Unix socket server
pub async fn run(config: Arc<Config>, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
    let socket_path = &config.daemon.socket_path;

    // Remove existing socket if it exists
    let _ = std::fs::remove_file(socket_path);

    // Bind to the socket
    let listener = UnixListener::bind(socket_path)?;
    info!("Listening on {}", socket_path);

    // Set socket permissions (allow all users to connect)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o666);
        std::fs::set_permissions(socket_path, perms)?;
    }

    loop {
        tokio::select! {
            // Handle new connections
            result = listener.accept() => {
                match result {
                    Ok((stream, _addr)) => {
                        let config = config.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_client(stream, config).await {
                                error!("Client handler error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        error!("Accept error: {}", e);
                    }
                }
            }
            // Handle shutdown signal
            _ = shutdown.recv() => {
                info!("Server shutting down...");
                break;
            }
        }
    }

    // Cleanup socket
    let _ = std::fs::remove_file(socket_path);

    Ok(())
}

/// Handle a single client connection
async fn handle_client(stream: UnixStream, config: Arc<Config>) -> Result<()> {
    debug!("New client connected");

    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    // Read request (single line JSON)
    match reader.read_line(&mut line).await {
        Ok(0) => {
            debug!("Client disconnected");
            return Ok(());
        }
        Ok(_) => {}
        Err(e) => {
            error!("Read error: {}", e);
            return Err(e.into());
        }
    }

    // Parse request
    let request: AuthRequest = match serde_json::from_str(&line) {
        Ok(r) => r,
        Err(e) => {
            warn!("Invalid request: {}", e);
            let response = AuthResponse::failure("Invalid request format");
            let response_json = serde_json::to_string(&response)?;
            writer.write_all(response_json.as_bytes()).await?;
            writer.write_all(b"\n").await?;
            return Ok(());
        }
    };

    info!(
        "Auth request: user={}, source={}, timeout={}",
        request.username, request.source, request.timeout
    );

    // Process authentication
    let response = auth::authenticate(&config, &request).await;

    // Send response
    let response_json = serde_json::to_string(&response)?;
    writer.write_all(response_json.as_bytes()).await?;
    writer.write_all(b"\n").await?;

    if response.success {
        info!(
            "Auth success: user={}, confidence={:.2}%",
            request.username,
            response.confidence.unwrap_or(0.0) * 100.0
        );
    } else {
        info!("Auth failed: user={}, reason={}", request.username, response.message);
    }

    Ok(())
}
