//! Config command

use anyhow::{anyhow, Result};
use facepass_core::config::{Config, ConfigFile};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn run(
    config_path: &str,
    show: bool,
    preset: Option<String>,
    list_presets: bool,
    use_preset: Option<String>,
    set: Option<String>,
) -> Result<()> {
    validate_args(
        show,
        preset.as_deref(),
        list_presets,
        use_preset.as_deref(),
        set.as_deref(),
    )?;
    let show_by_default = !list_presets && use_preset.is_none() && set.is_none();

    if list_presets {
        return list_presets_command(config_path);
    }

    if let Some(preset_name) = use_preset {
        return use_preset_command(config_path, &preset_name);
    }

    if let Some(value) = set {
        return set_value_command(config_path, &value);
    }

    let should_show = show || preset.is_some();
    if should_show || show_by_default {
        return show_config_command(config_path, preset.as_deref());
    }

    Ok(())
}

fn validate_args(
    show: bool,
    preset: Option<&str>,
    list_presets: bool,
    use_preset: Option<&str>,
    set: Option<&str>,
) -> Result<()> {
    if show && list_presets {
        return Err(anyhow!("--show cannot be combined with --list-presets"));
    }

    if show && use_preset.is_some() {
        return Err(anyhow!("--show cannot be combined with --use-preset"));
    }

    if show && set.is_some() {
        return Err(anyhow!("--show cannot be combined with --set"));
    }

    if preset.is_some() && list_presets {
        return Err(anyhow!("--preset cannot be combined with --list-presets"));
    }

    if preset.is_some() && use_preset.is_some() {
        return Err(anyhow!("--preset cannot be combined with --use-preset"));
    }

    if preset.is_some() && set.is_some() {
        return Err(anyhow!("--preset cannot be combined with --set"));
    }

    if list_presets && use_preset.is_some() {
        return Err(anyhow!(
            "--list-presets cannot be combined with --use-preset"
        ));
    }

    if list_presets && set.is_some() {
        return Err(anyhow!("--list-presets cannot be combined with --set"));
    }

    if use_preset.is_some() && set.is_some() {
        return Err(anyhow!("--use-preset cannot be combined with --set"));
    }

    Ok(())
}

fn show_config_command(config_path: &str, preset: Option<&str>) -> Result<()> {
    let (config, source, active_preset) = if let Some(preset_name) = preset {
        let (config_file, source) = ConfigFile::load_with_fallback_and_source(config_path)?;
        (
            config_file.resolve_preset(preset_name)?,
            source,
            preset_name.trim().to_string(),
        )
    } else {
        let resolved = Config::load_with_fallback_and_source(config_path)?;
        (resolved.config, resolved.source, resolved.active_preset)
    };

    println!("FacePass Configuration");
    println!("======================\n");

    println!("Active preset = \"{}\"", active_preset);
    println!("Config source = {}", format_source(source.as_deref()));
    println!();

    print_config(&config);
    Ok(())
}

fn list_presets_command(config_path: &str) -> Result<()> {
    let (config_file, source) = ConfigFile::load_with_fallback_and_source(config_path)?;
    let active_preset = config_file.active_preset_name()?;

    println!("FacePass Presets");
    println!("================\n");
    println!("Config source = {}", format_source(source.as_deref()));
    println!();

    for preset in config_file.list_presets() {
        let marker = if preset == active_preset { "*" } else { " " };
        println!("{} {}", marker, preset);
    }

    Ok(())
}

fn use_preset_command(config_path: &str, preset_name: &str) -> Result<()> {
    let (mut config_file, source) = ConfigFile::load_with_fallback_and_source(config_path)?;
    config_file.set_active_preset(preset_name)?;

    let target = resolve_write_target(config_path, source.as_deref());
    config_file.save(&target)?;

    println!("Active preset updated");
    println!("  Preset: {}", config_file.active_preset_name()?);
    println!("  Config file: {}", target.display());

    if unsafe { libc::geteuid() } == 0 {
        request_daemon_reload();
    } else {
        println!("\nDaemon reload requires root privileges.");
        println!("The new preset will apply after the next daemon reload or restart.");
    }

    Ok(())
}

fn set_value_command(config_path: &str, value: &str) -> Result<()> {
    println!("Setting configuration values is not yet implemented.");
    println!("Please edit the config file directly:\n");

    let (config_file, source) = ConfigFile::load_with_fallback_and_source(config_path)?;
    let target = resolve_write_target(config_path, source.as_deref());
    if !target.exists() {
        config_file.save(&target)?;
        println!("Created: {}", target.display());
    }

    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "nano".to_string());

    println!("  {} {}", editor, target.display());
    println!("\nOr use:");
    println!("  sudo facepass config --show");
    println!("\nTo set: {}", value);

    if target.exists() {
        let status = Command::new(&editor).arg(&target).status();
        if let Err(e) = status {
            eprintln!("Failed to open editor: {}", e);
        }
    }

    Ok(())
}

fn request_daemon_reload() {
    match Command::new("systemctl")
        .args(["reload", "facepass.service"])
        .output()
    {
        Ok(output) if output.status.success() => {
            println!("\nDaemon reload requested.");
            println!("Changes should apply immediately, except socket/pid path updates which require a full restart.");
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!(
                "\nWarning: Could not reload facepass.service: {}",
                stderr.trim()
            );
            println!("The new preset will apply after the next daemon restart.");
        }
        Err(e) => {
            eprintln!("\nWarning: systemctl not available: {}", e);
            println!("The new preset will apply after the next daemon restart.");
        }
    }
}

fn resolve_write_target(config_path: &str, source: Option<&Path>) -> PathBuf {
    source
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(config_path))
}

fn format_source(source: Option<&Path>) -> String {
    source
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "built-in defaults".to_string())
}

fn print_config(config: &Config) {
    println!("[video]");
    println!("  device = \"{}\"", config.video.device);
    println!("  timeout = {}", config.video.timeout);
    println!("  max_frames = {}", config.video.max_frames);
    println!("  frame_width = {}", config.video.frame_width);
    println!("  frame_height = {}", config.video.frame_height);
    println!();

    println!("[detection]");
    println!("  score_threshold = {}", config.detection.score_threshold);
    println!("  nms_threshold = {}", config.detection.nms_threshold);
    println!("  input_width = {}", config.detection.input_width);
    println!("  input_height = {}", config.detection.input_height);
    println!();

    println!("[recognition]");
    println!(
        "  similarity_threshold = {}",
        config.recognition.similarity_threshold
    );
    println!(
        "  max_faces_per_group = {}",
        config.recognition.max_faces_per_group
    );
    println!(
        "  consecutive_match_frames = {}",
        config.recognition.consecutive_match_frames
    );
    println!("  valid_frames = {}", config.recognition.valid_frames);
    println!(
        "  stop_on_valid_frames = {}",
        config.recognition.stop_on_valid_frames
    );
    println!(
        "  valid_crop_scale = {}",
        config.recognition.valid_crop_scale
    );
    println!();

    println!("[security]");
    println!("  ignore_ssh = {}", config.security.ignore_ssh);
    println!(
        "  ignore_closed_lid = {}",
        config.security.ignore_closed_lid
    );
    println!(
        "  show_notification = {}",
        config.security.show_notification
    );
    println!();

    println!("[daemon]");
    println!("  socket_path = \"{}\"", config.daemon.socket_path);
    println!("  log_level = \"{}\"", config.daemon.log_level);
    println!("  pid_file = \"{}\"", config.daemon.pid_file);
    println!();

    println!("[models]");
    println!("  yunet_path = \"{}\"", config.models.yunet_path);
    println!("  sface_path = \"{}\"", config.models.sface_path);
    println!(
        "  anti_spoof_v2_path = \"{}\"",
        config.models.anti_spoof_v2_path
    );
    println!(
        "  anti_spoof_v1se_path = \"{}\"",
        config.models.anti_spoof_v1se_path
    );
    println!();

    println!("[anti_spoof]");
    println!("  enabled = {}", config.anti_spoof.enabled);
    println!("  threshold = {}", config.anti_spoof.threshold);
    println!("  mode = \"{}\"", config.anti_spoof.mode.as_str());
    println!("  v2_input_size = {}", config.anti_spoof.v2_input_size);
    println!("  v2_crop_scale = {}", config.anti_spoof.v2_crop_scale);
    println!("  v1se_input_size = {}", config.anti_spoof.v1se_input_size);
    println!("  v1se_crop_scale = {}", config.anti_spoof.v1se_crop_scale);
    println!();

    println!("[storage]");
    println!("  data_dir = \"{}\"", config.storage.data_dir);
}
