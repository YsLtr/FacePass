//! Unix socket server for the daemon

use crate::auth;
use crate::control::AuthControl;
use anyhow::Result;
use facepass_core::config::Config;
use facepass_core::models::{AuthRequest, AuthResponse, CancelRequest};
use log::{debug, error, info, warn};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::broadcast;

/// Run the Unix socket server
pub async fn run(
    config: Arc<Config>,
    auth_control: Arc<AuthControl>,
    mut shutdown: broadcast::Receiver<()>,
) -> Result<()> {
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
                        let auth_control = auth_control.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_client(stream, config, auth_control).await {
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
async fn handle_client(
    stream: UnixStream,
    config: Arc<Config>,
    auth_control: Arc<AuthControl>,
) -> Result<()> {
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

    let mut auth_user: Option<String> = None;
    let response = match extract_message_type(&line) {
        Ok(msg_type) if msg_type == "auth" => {
            let request = match serde_json::from_str::<AuthRequest>(&line) {
                Ok(r) => r,
                Err(e) => {
                    warn!("Invalid auth request: {}", e);
                    let response = AuthResponse::failure("Invalid auth request format");
                    send_response(&mut writer, &response).await?;
                    return Ok(());
                }
            };

            if request.msg_type != "auth" {
                AuthResponse::failure("Unsupported message type")
            } else {
                auth_user = Some(request.username.clone());
                info!(
                    "Auth request: user={}, source={}, timeout={}",
                    request.username, request.source, request.timeout
                );
                auth::authenticate(&config, &auth_control, &request).await
            }
        }
        Ok(msg_type) if msg_type == "cancel" => {
            let request: CancelRequest = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    warn!("Invalid cancel request: {}", e);
                    let response = AuthResponse::failure("Invalid cancel request format");
                    send_response(&mut writer, &response).await?;
                    return Ok(());
                }
            };

            if request.msg_type != "cancel" {
                AuthResponse::failure("Unsupported message type")
            } else if auth_control.cancel_current() {
                info!("Cancellation requested for current authentication");
                AuthResponse::message(true, "Cancellation requested")
            } else {
                AuthResponse::failure("No active authentication to cancel")
            }
        }
        Ok(msg_type) => {
            warn!("Unsupported request type: {}", msg_type);
            AuthResponse::failure("Unsupported message type")
        }
        Err(e) => {
            warn!("Invalid request: {}", e);
            AuthResponse::failure("Invalid request format")
        }
    };

    // Send response
    send_response(&mut writer, &response).await?;

    if let Some(username) = auth_user {
        if response.success {
            info!(
                "Auth success: user={}, confidence={:.2}%",
                username,
                response.confidence.unwrap_or(0.0) * 100.0
            );
        } else {
            info!(
                "Auth failed: user={}, reason={}",
                username, response.message
            );
        }
    }

    Ok(())
}

fn extract_message_type(line: &str) -> Result<String> {
    let value: serde_json::Value = serde_json::from_str(line)?;
    value
        .get("msg_type")
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .ok_or_else(|| anyhow::anyhow!("Missing msg_type"))
}

async fn send_response(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    response: &AuthResponse,
) -> Result<()> {
    let response_json = serde_json::to_string(response)?;
    writer.write_all(response_json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    Ok(())
}
