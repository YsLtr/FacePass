//! FacePass Daemon - Face authentication service

mod auth;
mod control;
mod runtime;
mod server;

use anyhow::Result;
use control::AuthControl;
use facepass_core::config::{Config, DEFAULT_RUNTIME_STATE_PATH};
use log::{error, info};
use runtime::{apply_log_level, load_initial_state, reload, snapshot, write_runtime_state};
use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
use signal_hook_tokio::Signals;
use std::sync::Arc;
use tokio::sync::broadcast;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    info!("FacePass daemon starting...");

    // Load configuration
    let runtime_state = Arc::new(std::sync::RwLock::new(load_initial_state()?));
    let config_snapshot = snapshot(&runtime_state);
    let config = config_snapshot.config.clone();

    // Set log level from config
    apply_log_level(&config.daemon.log_level);

    info!("Configuration loaded");
    info!("Socket path: {}", config.daemon.socket_path);
    info!("Data directory: {}", config.storage.data_dir);
    info!("Running preset: {}", config_snapshot.running_preset);

    // Create shutdown channel
    let (shutdown_tx, _) = broadcast::channel::<()>(1);
    let auth_control = Arc::new(AuthControl::default());

    // Setup signal handlers
    let mut signals = Signals::new([SIGHUP, SIGINT, SIGTERM])?;
    let shutdown_tx_signal = shutdown_tx.clone();
    let runtime_state_signal = runtime_state.clone();

    tokio::spawn(async move {
        use futures::StreamExt;
        while let Some(signal) = signals.next().await {
            match signal {
                SIGHUP => match reload(&runtime_state_signal) {
                    Ok(outcome) => {
                        info!(
                            "Configuration reloaded; running preset={}",
                            outcome.running_preset
                        );
                        for field in outcome.deferred_fields {
                            log::warn!(
                                "{} changed but requires a full restart; current runtime value was kept",
                                field
                            );
                        }
                    }
                    Err(e) => {
                        error!("Configuration reload failed: {}", e);
                    }
                },
                SIGINT | SIGTERM => {
                    info!("Received shutdown signal");
                    let _ = shutdown_tx_signal.send(());
                    break;
                }
                _ => {}
            }
        }
    });

    // Ensure runtime directories exist
    ensure_directories(&config)?;

    // Write PID file
    write_pid_file(&config.daemon.pid_file)?;
    write_runtime_state(&config_snapshot)?;

    // Start the server
    let server_result = server::run(runtime_state, auth_control, shutdown_tx.subscribe()).await;

    // Cleanup
    cleanup(&config.daemon.pid_file);

    match server_result {
        Ok(_) => {
            info!("FacePass daemon stopped gracefully");
            Ok(())
        }
        Err(e) => {
            error!("FacePass daemon error: {}", e);
            Err(e)
        }
    }
}

fn ensure_directories(config: &Config) -> Result<()> {
    // Create socket directory
    if let Some(parent) = std::path::Path::new(&config.daemon.socket_path).parent() {
        std::fs::create_dir_all(parent)?;
    }

    if let Some(parent) = std::path::Path::new(DEFAULT_RUNTIME_STATE_PATH).parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Create data directory
    std::fs::create_dir_all(&config.storage.data_dir)?;

    Ok(())
}

fn write_pid_file(path: &str) -> Result<()> {
    let pid = std::process::id();
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, pid.to_string())?;
    info!("PID file written: {}", path);
    Ok(())
}

fn cleanup(pid_file: &str) {
    // Remove PID file
    let _ = std::fs::remove_file(pid_file);
    let _ = std::fs::remove_file(DEFAULT_RUNTIME_STATE_PATH);
}
