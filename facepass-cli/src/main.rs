//! FacePass CLI - Face authentication management tool

use clap::{value_parser, Parser, Subcommand};

mod commands;

/// FacePass - Linux Face Authentication Tool
#[derive(Parser)]
#[command(name = "facepass")]
#[command(author = "ysltr")]
#[command(version)]
#[command(about = "Linux face recognition authentication management tool")]
#[command(
    long_about = "FacePass is a Linux PAM module that provides face recognition \
                        authentication for sudo, polkit, and other privilege escalation tools."
)]
struct Cli {
    /// Path to configuration file
    #[arg(
        short,
        long,
        global = true,
        default_value = "~/.config/facepass/config.toml"
    )]
    config: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Add a new face for a user
    Add {
        /// Username to add face for (requires root for other users)
        #[arg(short, long)]
        user: Option<String>,

        /// Enable debug mode
        #[arg(short = 'd', long)]
        debug: bool,

        /// Show camera window with overlay info
        #[arg(short = 'v', long)]
        view: bool,

        /// Target face group (index or exact name)
        #[arg(short = 'g', long)]
        group: Option<String>,

        /// Label for this face (e.g., "with glasses", "normal")
        #[arg(short, long)]
        label: Option<String>,

        /// Label for this face, equivalent to --label
        #[arg(value_name = "LABEL", conflicts_with = "label")]
        positional_label: Option<String>,
    },

    /// Remove users, groups, or faces depending on selector scope
    Remove {
        /// Username to remove from (requires root for other users)
        #[arg(short, long)]
        user: Option<String>,

        /// Target face group (index or exact name)
        #[arg(short = 'g', long)]
        group: Option<String>,

        /// Face selector(s) within the target group (index or exact label)
        #[arg(short = 'f', long = "face")]
        face: Vec<String>,
    },

    /// List users, groups, or faces
    List {
        /// Username to list for (requires root for other users)
        #[arg(short, long)]
        user: Option<String>,

        /// Limit output to a face group (index or exact name)
        #[arg(short = 'g', long)]
        group: Option<String>,

        /// Display depth: 1=user summary, 2=groups, 3=faces
        #[arg(long, default_value_t = 3, value_parser = value_parser!(u8).range(1..=3))]
        depth: u8,
    },

    /// Test face recognition
    Test {
        /// Username to test against (requires root for other users)
        #[arg(short, long)]
        user: Option<String>,

        /// Enable debug mode
        #[arg(short = 'd', long)]
        debug: bool,

        /// Show camera window with overlay info
        #[arg(short = 'v', long)]
        view: bool,

        /// Target face group (index or exact name)
        #[arg(short = 'g', long)]
        group: Option<String>,

        /// Override the maximum number of frames to test
        #[arg(short, long)]
        frames: Option<u32>,
    },

    /// View or edit configuration
    Config {
        /// Show current configuration
        #[arg(short, long)]
        show: bool,

        /// Set a configuration value (e.g., video.timeout=10)
        #[arg(short = 'S', long)]
        set: Option<String>,
    },

    /// List available camera devices
    Cameras,

    /// Cancel the current face authentication attempt
    Cancel,

    /// Check system status and requirements
    Status {
        /// Show the full status report
        #[arg(short, long)]
        show: bool,

        /// Username to update when using --set-default-group
        #[arg(short, long)]
        user: Option<String>,

        /// Target face group (index or exact name) when using --set-default-group
        #[arg(short = 'g', long)]
        group: Option<String>,

        /// Set the default face group, optionally passing the group selector inline
        #[arg(long = "set-default-group", num_args = 0..=1, value_name = "GROUP")]
        set_default_group: Option<Option<String>>,
    },

    /// Enable face authentication (start daemon)
    Enable,

    /// Disable face authentication (stop daemon)
    Disable,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let cli = Cli::parse();
    let uses_view_runtime = match &cli.command {
        Commands::Add { view, .. } | Commands::Test { view, .. } => *view,
        _ => false,
    };
    let config_path = expand_tilde(&cli.config);
    let is_root = unsafe { libc::geteuid() == 0 };

    let result = match cli.command {
        Commands::Add {
            user,
            debug,
            view,
            group,
            label,
            positional_label,
        } => {
            require_root_if_other_user(&user, is_root, "add faces for other users");
            let label = label.or(positional_label);
            commands::add::run(&config_path, user, group, label, debug, view)
        }
        Commands::Remove { user, group, face } => {
            require_root_if_other_user(&user, is_root, "remove data for other users");
            commands::remove::run(&config_path, user, group, face)
        }
        Commands::List { user, group, depth } => {
            require_root_if_other_user(&user, is_root, "list data for other users");
            commands::list::run(&config_path, user, group, depth)
        }
        Commands::Test {
            user,
            debug,
            view,
            group,
            frames,
        } => {
            require_root_if_other_user(&user, is_root, "test faces for other users");
            commands::test::run(&config_path, user, group, frames, debug, view)
        }
        Commands::Config { show, set } => commands::config::run(&config_path, show, set),
        Commands::Cameras => commands::cameras::run(),
        Commands::Cancel => commands::cancel::run(&config_path),
        Commands::Status {
            show,
            user,
            group,
            set_default_group,
        } => {
            require_root_if_other_user(&user, is_root, "change default groups for other users");
            commands::status::run(&config_path, show, user, group, set_default_group)
        }
        Commands::Enable => {
            if !is_root {
                eprintln!("Error: Root privileges required");
                std::process::exit(1);
            }
            commands::enable::run()
        }
        Commands::Disable => {
            if !is_root {
                eprintln!("Error: Root privileges required");
                std::process::exit(1);
            }
            commands::disable::run()
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        if uses_view_runtime {
            immediate_exit(1);
        }
        std::process::exit(1);
    }

    if uses_view_runtime {
        immediate_exit(0);
    }
}

fn require_root_if_other_user(user: &Option<String>, is_root: bool, action: &str) {
    if is_root {
        return;
    }

    let Some(target_user) = user.as_deref() else {
        return;
    };

    let current_user = match commands::get_username(None) {
        Ok(user) => user,
        Err(_) => {
            eprintln!("Error: Could not determine current username");
            std::process::exit(1);
        }
    };

    if target_user != current_user {
        eprintln!("Error: Root privileges required to {}", action);
        std::process::exit(1);
    }
}

/// Expand ~ to home directory
fn expand_tilde(path: &str) -> String {
    if path.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return format!("{}{}", home.to_string_lossy(), &path[1..]);
        }
    }
    path.to_string()
}

fn immediate_exit(code: i32) -> ! {
    use std::io::Write;

    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    unsafe {
        libc::_exit(code);
    }
}
