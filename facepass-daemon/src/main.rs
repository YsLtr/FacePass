//! FacePass Daemon - Face authentication service

mod auth;
mod control;
mod server;

use anyhow::Result;
use control::AuthControl;
use facepass_core::config::Config;
use log::{error, info};
use signal_hook::consts::{SIGINT, SIGTERM};
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
    let config = Arc::new(Config::load_or_default()?);

    // Set log level from config
    if config.daemon.log_level == "debug" {
        log::set_max_level(log::LevelFilter::Debug);
    } else if config.daemon.log_level == "trace" {
        log::set_max_level(log::LevelFilter::Trace);
    }

    info!("Configuration loaded");
    info!("Socket path: {}", config.daemon.socket_path);
    info!("Data directory: {}", config.storage.data_dir);

    // Create shutdown channel
    let (shutdown_tx, _) = broadcast::channel::<()>(1);
    let auth_control = Arc::new(AuthControl::default());

    // Setup signal handlers
    let mut signals = Signals::new([SIGINT, SIGTERM])?;
    let shutdown_tx_signal = shutdown_tx.clone();

    tokio::spawn(async move {
        use futures::StreamExt;
        while let Some(signal) = signals.next().await {
            match signal {
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

    // Start the server
    let server_result = server::run(config.clone(), auth_control, shutdown_tx.subscribe()).await;

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
}
