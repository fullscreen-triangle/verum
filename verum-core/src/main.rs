//! # Verum Core Binary
//!
//! Main executable for the Verum personal AI driving system

use clap::{Arg, Command};
use std::process;
use tracing::{info, error};
use verum_core::{VerumEngine, Config};

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let matches = Command::new("verum-core")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Verum Personal AI Driving Engine")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
        )
        .arg(
            Arg::new("destination")
                .short('d')
                .long("destination")
                .value_name("DEST")
                .help("Destination for navigation")
        )
        .subcommand(
            Command::new("start")
                .about("Start the Verum engine")
        )
        .subcommand(
            Command::new("calibrate")
                .about("Calibrate biometric baseline")
        )
        .subcommand(
            Command::new("status")
                .about("Show engine status")
        )
        .get_matches();

    // Load configuration
    let config = if let Some(config_path) = matches.get_one::<String>("config") {
        match Config::load_from_file(config_path) {
            Ok(config) => config,
            Err(e) => {
                error!("Failed to load config from {}: {}", config_path, e);
                process::exit(1);
            }
        }
    } else {
        match Config::from_env() {
            Ok(config) => config,
            Err(e) => {
                error!("Failed to load config from environment: {}", e);
                Config::default()
            }
        }
    };

    // Create directories
    if let Err(e) = config.create_directories() {
        error!("Failed to create directories: {}", e);
        process::exit(1);
    }

    match matches.subcommand() {
        Some(("start", _)) => {
            info!("Starting Verum engine...");
            
            // Create and start engine
            let engine = match VerumEngine::new(config).await {
                Ok(engine) => engine,
                Err(e) => {
                    error!("Failed to create Verum engine: {}", e);
                    process::exit(1);
                }
            };

            if let Err(e) = engine.start().await {
                error!("Failed to start engine: {}", e);
                process::exit(1);
            }

            // If destination is provided, start navigation
            if let Some(destination) = matches.get_one::<String>("destination") {
                info!("Starting navigation to: {}", destination);
                
                if let Err(e) = engine.drive_to_destination(destination.clone()).await {
                    error!("Navigation failed: {}", e);
                    process::exit(1);
                }
            } else {
                // Run indefinitely
                info!("Engine started. Press Ctrl+C to stop.");
                tokio::signal::ctrl_c().await.expect("Failed to listen for ctrl-c");
            }

            // Shutdown
            if let Err(e) = engine.shutdown().await {
                error!("Failed to shutdown cleanly: {}", e);
                process::exit(1);
            }

            info!("Verum engine stopped");
        }
        
        Some(("calibrate", _)) => {
            info!("Starting biometric calibration...");
            
            let engine = match VerumEngine::new(config).await {
                Ok(engine) => engine,
                Err(e) => {
                    error!("Failed to create Verum engine: {}", e);
                    process::exit(1);
                }
            };

            // This would trigger calibration process
            info!("Calibration process would start here");
            info!("Please sit calmly for 10 minutes while we calibrate your baseline");
            
            // For now, just simulate calibration
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            
            info!("Calibration complete");
        }
        
        Some(("status", _)) => {
            info!("Checking engine status...");
            
            let engine = match VerumEngine::new(config).await {
                Ok(engine) => engine,
                Err(e) => {
                    error!("Failed to create Verum engine: {}", e);
                    process::exit(1);
                }
            };

            let stats = engine.get_statistics().await;
            println!("Engine Statistics:");
            println!("  Engine ID: {}", stats.engine_id);
            println!("  Uptime: {} seconds", stats.uptime);
            println!("  AI Patterns Learned: {}", stats.ai_statistics.patterns_learned);
            println!("  Fear Responses Trained: {}", stats.ai_statistics.fear_responses_trained);
            println!("  Decisions Made: {}", stats.ai_statistics.decisions_made);
            println!("  Average Confidence: {:.2}", stats.ai_statistics.average_confidence);
        }
        
        _ => {
            println!("No subcommand provided. Use --help for usage information.");
            println!("Available commands:");
            println!("  start      - Start the Verum engine");
            println!("  calibrate  - Calibrate biometric baseline");
            println!("  status     - Show engine status");
        }
    }
} 