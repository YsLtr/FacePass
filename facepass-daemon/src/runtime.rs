use anyhow::Result;
use facepass_core::config::{
    Config, DaemonRuntimeState, ResolvedConfig, DEFAULT_RUNTIME_STATE_PATH,
};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub struct RuntimeConfigState {
    pub config: Config,
    pub source: Option<PathBuf>,
    pub running_preset: String,
}

impl RuntimeConfigState {
    pub fn from_resolved(resolved: ResolvedConfig) -> Self {
        Self {
            config: resolved.config,
            source: resolved.source,
            running_preset: resolved.active_preset,
        }
    }

    pub fn daemon_runtime_state(&self) -> DaemonRuntimeState {
        DaemonRuntimeState {
            running_preset: self.running_preset.clone(),
            socket_path: self.config.daemon.socket_path.clone(),
            pid_file: self.config.daemon.pid_file.clone(),
        }
    }
}

pub type SharedRuntimeConfig = Arc<RwLock<RuntimeConfigState>>;

pub struct ReloadOutcome {
    pub running_preset: String,
    pub deferred_fields: Vec<&'static str>,
}

pub fn load_initial_state() -> Result<RuntimeConfigState> {
    Ok(RuntimeConfigState::from_resolved(
        Config::load_or_default_with_source()?,
    ))
}

pub fn snapshot(shared: &SharedRuntimeConfig) -> RuntimeConfigState {
    shared
        .read()
        .expect("runtime config rwlock poisoned")
        .clone()
}

pub fn write_runtime_state(state: &RuntimeConfigState) -> Result<()> {
    state
        .daemon_runtime_state()
        .save(DEFAULT_RUNTIME_STATE_PATH)
        .map_err(Into::into)
}

pub fn reload(shared: &SharedRuntimeConfig) -> Result<ReloadOutcome> {
    let current = snapshot(shared);
    let resolved = match current.source.as_deref() {
        Some(path) => Config::load_with_source(path)?,
        None => Config::load_or_default_with_source()?,
    };

    let mut new_config = resolved.config;
    let mut deferred_fields = Vec::new();

    if new_config.daemon.socket_path != current.config.daemon.socket_path {
        deferred_fields.push("daemon.socket_path");
        new_config.daemon.socket_path = current.config.daemon.socket_path.clone();
    }

    if new_config.daemon.pid_file != current.config.daemon.pid_file {
        deferred_fields.push("daemon.pid_file");
        new_config.daemon.pid_file = current.config.daemon.pid_file.clone();
    }

    apply_log_level(&new_config.daemon.log_level);

    let new_state = RuntimeConfigState {
        config: new_config,
        source: resolved.source,
        running_preset: resolved.active_preset,
    };

    {
        let mut guard = shared.write().expect("runtime config rwlock poisoned");
        *guard = new_state.clone();
    }

    write_runtime_state(&new_state)?;

    Ok(ReloadOutcome {
        running_preset: new_state.running_preset,
        deferred_fields,
    })
}

pub fn apply_log_level(level: &str) {
    let filter = match level {
        "trace" => log::LevelFilter::Trace,
        "debug" => log::LevelFilter::Debug,
        "warn" => log::LevelFilter::Warn,
        "error" => log::LevelFilter::Error,
        _ => log::LevelFilter::Info,
    };

    log::set_max_level(filter);
}
