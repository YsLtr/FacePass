//! FacePass CLI - Face authentication management tool

use clap::{Parser, Subcommand};

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
    /// Enable debug mode
    #[arg(short = 'd', long, global = true)]
    debug: bool,

    /// Show camera window with overlay info
    #[arg(short = 'v', long, global = true)]
    view: bool,

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
    /// Add a new face for the current user
    Add {
        /// Username to add face for (requires root)
        #[arg(short, long)]
        user: Option<String>,

        /// Label for this face (e.g., "with glasses", "normal")
        #[arg(short, long)]
        label: Option<String>,
    },

    /// Remove a registered face
    Remove {
        /// Face index to remove (use 'list' to see indices)
        index: usize,

        /// Username (requires root for other users)
        #[arg(short, long)]
        user: Option<String>,
    },

    /// List all registered faces for a user
    List {
        /// Username to list faces for (requires root for other users)
        #[arg(short, long)]
        user: Option<String>,
    },

    /// Clear all faces for a user
    Clear {
        /// Username to clear faces for (requires root for other users)
        #[arg(short, long)]
        user: Option<String>,

        /// Skip confirmation prompt
        #[arg(short, long)]
        force: bool,
    },

    /// Test face recognition
    Test {
        /// Username to test against (requires root for other users)
        #[arg(short, long)]
        user: Option<String>,

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
    Status,

    /// Enable face authentication (start daemon)
    Enable,

    /// Disable face authentication (stop daemon)
    Disable,
}

fn main() {
    // Initialize logger
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp(None)
        .init();

    let cli = Cli::parse();
    let uses_view_runtime =
        cli.view && matches!(&cli.command, Commands::Add { .. } | Commands::Test { .. });

    // Expand ~ to home directory
    let config_path = expand_tilde(&cli.config);

    // Check root privileges for most commands
    let is_root = unsafe { libc::geteuid() == 0 };

    let result = match cli.command {
        Commands::Add { user, label } => {
            if user.is_some() && !is_root {
                eprintln!("Error: Root privileges required to add faces for other users");
                std::process::exit(1);
            }
            commands::add::run(&config_path, user, label, cli.debug, cli.view)
        }
        Commands::Remove { index, user } => {
            if user.is_some() && !is_root {
                eprintln!("Error: Root privileges required to remove faces for other users");
                std::process::exit(1);
            }
            commands::remove::run(&config_path, user, index)
        }
        Commands::List { user } => {
            if user.is_some() && !is_root {
                eprintln!("Error: Root privileges required to list faces for other users");
                std::process::exit(1);
            }
            commands::list::run(&config_path, user)
        }
        Commands::Clear { user, force } => {
            if user.is_some() && !is_root {
                eprintln!("Error: Root privileges required to clear faces for other users");
                std::process::exit(1);
            }
            commands::clear::run(&config_path, user, force)
        }
        Commands::Test { user, frames } => {
            if user.is_some() && !is_root {
                eprintln!("Error: Root privileges required to test faces for other users");
                std::process::exit(1);
            }
            commands::test::run(&config_path, user, frames, cli.debug, cli.view)
        }
        Commands::Config { show, set } => commands::config::run(&config_path, show, set),
        Commands::Cameras => commands::cameras::run(),
        Commands::Cancel => commands::cancel::run(&config_path),
        Commands::Status => commands::status::run(&config_path),
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

    // OpenCV HighGUI with the Qt backend is unstable during process teardown here.
    unsafe {
        libc::_exit(code);
    }
}
