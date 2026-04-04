//! Status check command

use anyhow::Result;
use facepass_core::{
    camera::check_camera,
    config::{Config, DaemonRuntimeState, DEFAULT_RUNTIME_STATE_PATH},
    security::{is_lid_closed, is_ssh_session},
};
use std::path::Path;

pub fn run(config_path: &str) -> Result<()> {
    let resolved = Config::load_with_fallback_and_source(config_path)?;
    let config = resolved.config;
    let config_source = resolved.source;
    let config_active_preset = resolved.active_preset;
    let runtime_state = load_runtime_state();

    print_status_report(
        &config,
        config_source.as_deref(),
        &config_active_preset,
        runtime_state.as_ref(),
    )
}

fn print_status_report(
    config: &Config,
    config_source: Option<&std::path::Path>,
    config_active_preset: &str,
    runtime_state: Option<&DaemonRuntimeState>,
) -> Result<()> {
    println!("FacePass System Status");
    println!("======================\n");

    let runtime_socket_path = runtime_state
        .map(|state| state.socket_path.as_str())
        .unwrap_or(&config.daemon.socket_path);

    print!("Daemon: ");
    if is_daemon_running(runtime_socket_path) {
        println!("OK Running");
    } else {
        println!("X Not running");
    }

    print!("Socket: ");
    if Path::new(runtime_socket_path).exists() {
        println!("OK {}", runtime_socket_path);
    } else {
        println!("X Not found");
    }

    println!("\nRecognition Backend:");
    println!("  Active detector: {}", config.models.active_detector.as_str());
    println!(
        "  Active detector path: {}",
        config.models.active_detector_config().path
    );
    println!("  Active recognizer: {}", config.models.active_recognizer.as_str());
    println!(
        "  Active recognizer path: {}",
        config.models.active_recognizer_config().path
    );
    println!(
        "  Active model ID: {} | embedding_dim: {}",
        config.models.active_recognizer_config().model_id,
        config.models.active_recognizer_config().embedding_dim
    );
    println!(
        "  Recognizer preprocess: {} {}x{} {}",
        config.models.active_recognizer_config().preprocess.input_layout.as_str(),
        config.models.active_recognizer_config().preprocess.input_width,
        config.models.active_recognizer_config().preprocess.input_height,
        config.models.active_recognizer_config().preprocess.color_order.as_str()
    );

    println!("\nConfigured Detectors:");
    print_detector_status("YuNet", &config.models.yunet);
    print_detector_status("SCRFD", &config.models.scrfd);

    println!("\nConfigured Recognizers:");
    print_recognizer_status("SFace", &config.models.sface);
    print_recognizer_status("MobileFaceNet", &config.models.mobilefacenet);
    print_recognizer_status("GhostFaceNet", &config.models.ghostfacenet);

    print!("  Anti-spoofing: ");
    if config.anti_spoof.enabled {
        println!(
            "OK Enabled (threshold: {:.2}, mode: {})",
            config.anti_spoof.threshold,
            config.anti_spoof.mode.as_str()
        );
    } else {
        println!("X Disabled");
    }
    println!(
        "  Valid crop scale (recognition): {:.2}",
        config.recognition.valid_crop_scale
    );
    println!(
        "  Anti-spoof V2: {}",
        if Path::new(&config.models.anti_spoof_v2_path).exists() {
            "OK Found"
        } else {
            "X Missing"
        }
    );
    println!("    {}", config.models.anti_spoof_v2_path);
    println!(
        "  Anti-spoof V1SE: {}",
        if Path::new(&config.models.anti_spoof_v1se_path).exists() {
            "OK Found"
        } else {
            "X Missing"
        }
    );
    println!("    {}", config.models.anti_spoof_v1se_path);

    println!("\nCamera:");
    print!("  Device: ");
    if check_camera(&config.video.device) {
        println!("OK {}", config.video.device);
    } else {
        println!("X Not available: {}", config.video.device);
    }

    println!("\nPAM Module:");
    print!("  Library: ");
    let pam_path = "/usr/lib/security/pam_facepass.so";
    if Path::new(pam_path).exists() {
        println!("OK Installed");
    } else {
        println!("X Not installed");
    }

    println!("\nSecurity:");
    print!("  SSH Session: ");
    if is_ssh_session() {
        println!("Yes (face auth will be skipped)");
    } else {
        println!("No");
    }

    print!("  Laptop Lid: ");
    if is_lid_closed() {
        println!("Closed (face auth will be skipped)");
    } else {
        println!("Open");
    }

    println!("\nConfiguration:");
    println!(
        "  Running preset: {}",
        runtime_state
            .filter(|state| is_daemon_running(&state.socket_path))
            .map(|state| state.running_preset.as_str())
            .unwrap_or("(daemon not running)")
    );
    println!("  Config active preset: {}", config_active_preset);
    print!("  Config file: ");
    if let Some(path) = config_source {
        println!("OK {}", path.display());
    } else {
        println!("X Not found (using defaults)");
    }

    print!("  Data directory: ");
    if Path::new(&config.storage.data_dir).exists() {
        println!("OK {}", config.storage.data_dir);
    } else {
        println!("X Not found");
    }

    Ok(())
}

fn print_detector_status(name: &str, detector: &facepass_core::config::DetectorModelConfig) {
    println!(
        "  {}: {} | input: {}x{} | score: {:.2} | nms: {:.2}",
        name,
        if Path::new(&detector.path).exists() {
            "OK Found"
        } else {
            "X Missing"
        },
        detector.input_width,
        detector.input_height,
        detector.score_threshold,
        detector.nms_threshold
    );
    println!("    {}", detector.path);
}

fn print_recognizer_status(name: &str, recognizer: &facepass_core::config::RecognizerModelConfig) {
    println!(
        "  {}: {} | model_id: {} | dim: {} | input: {}x{} {} {}",
        name,
        if Path::new(&recognizer.path).exists() {
            "OK Found"
        } else {
            "X Missing"
        },
        recognizer.model_id,
        recognizer.embedding_dim,
        recognizer.preprocess.input_width,
        recognizer.preprocess.input_height,
        recognizer.preprocess.input_layout.as_str(),
        recognizer.preprocess.color_order.as_str()
    );
    println!("    {}", recognizer.path);
}

fn is_daemon_running(socket_path: &str) -> bool {
    if !Path::new(socket_path).exists() {
        return false;
    }

    std::os::unix::net::UnixStream::connect(socket_path).is_ok()
}

fn load_runtime_state() -> Option<DaemonRuntimeState> {
    DaemonRuntimeState::load(DEFAULT_RUNTIME_STATE_PATH).ok()
}
