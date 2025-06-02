//! Ruzende - Interactive Gusheshe Hybrid Resolution Engine Demo
//!
//! This executable demonstrates the capabilities of the Gusheshe hybrid resolution engine
//! for real-time decision making in autonomous driving contexts.

use clap::{Parser, Subcommand};
use gusheshe::{
    Engine, EngineConfig, Point, PointBuilder, Certificate, CertificateBuilder,
    types::{Action, Confidence, ProcessingMode, ExecutionContext},
    point::PointCategory,
    resolution::ResolutionStrategy,
    certificate::{CertificatePattern, ResolutionLogic},
};
use std::time::Duration;
use tracing::{info, warn, error};

#[derive(Parser)]
#[command(name = "ruzende")]
#[command(about = "Gusheshe Hybrid Resolution Engine - Real-time Decision Making")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Engine timeout in milliseconds
    #[arg(short, long, default_value = "100")]
    timeout: u64,
    
    /// Confidence threshold (0.0-1.0)
    #[arg(short, long, default_value = "0.65")]
    confidence_threshold: f64,
}

#[derive(Subcommand)]
enum Commands {
    /// Resolve a single point interactively
    Resolve {
        /// The point content to resolve
        content: String,
        
        /// Initial confidence (0.0-1.0)
        #[arg(short, long, default_value = "0.8")]
        confidence: f64,
        
        /// Point category
        #[arg(short, long, default_value = "observation")]
        category: String,
        
        /// Resolution strategy
        #[arg(short, long, default_value = "adaptive")]
        strategy: String,
    },
    
    /// Run interactive resolution session
    Interactive,
    
    /// Run benchmark scenarios
    Benchmark {
        /// Number of iterations
        #[arg(short, long, default_value = "1000")]
        iterations: usize,
        
        /// Number of concurrent resolutions
        #[arg(short, long, default_value = "10")]
        concurrent: usize,
    },
    
    /// Demonstrate certificate usage
    Certificate {
        /// Certificate action type
        #[arg(short, long, default_value = "demo")]
        action: String,
    },
    
    /// Show engine metrics and diagnostics
    Metrics,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Initialize tracing
    let level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("ruzende={},gusheshe={}", level, level))
        .init();
    
    info!("🚀 Ruzende - Gusheshe Hybrid Resolution Engine");
    info!("Version: {}", gusheshe::VERSION);
    
    // Configure engine
    let config = EngineConfig {
        default_timeout: Duration::from_millis(cli.timeout),
        default_confidence_threshold: cli.confidence_threshold,
        default_strategy: ResolutionStrategy::Adaptive,
        max_concurrent_resolutions: 50,
        enable_logical: true,
        enable_fuzzy: true,
        enable_bayesian: true,
        emergency_timeout: Duration::from_millis(10),
        emergency_confidence_threshold: 0.5,
    };
    
    let engine = Engine::with_config(config);
    
    match cli.command {
        Commands::Resolve { content, confidence, category, strategy } => {
            resolve_single_point(&engine, content, confidence, category, strategy).await?;
        },
        Commands::Interactive => {
            run_interactive_session(&engine).await?;
        },
        Commands::Benchmark { iterations, concurrent } => {
            run_benchmark(&engine, iterations, concurrent).await?;
        },
        Commands::Certificate { action } => {
            demonstrate_certificates(&engine, action).await?;
        },
        Commands::Metrics => {
            show_metrics(&engine).await?;
        },
    }
    
    Ok(())
}

/// Resolve a single point with specified parameters
async fn resolve_single_point(
    engine: &Engine,
    content: String,
    confidence: f64,
    category_str: String,
    strategy_str: String,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("🔍 Resolving point: '{}'", content);
    
    // Parse category
    let category = match category_str.to_lowercase().as_str() {
        "observation" => PointCategory::Observation,
        "inference" => PointCategory::Inference,
        "prediction" => PointCategory::Prediction,
        "intention" => PointCategory::Intention,
        "constraint" => PointCategory::Constraint,
        "safety" => PointCategory::Safety,
        "pattern" => PointCategory::Pattern,
        "human" => PointCategory::Human,
        _ => {
            warn!("Unknown category '{}', using Observation", category_str);
            PointCategory::Observation
        }
    };
    
    // Create point
    let point = PointBuilder::new(&content)
        .confidence(confidence)
        .category(category)
        .build();
    
    info!("📊 Point details:");
    info!("  Content: {}", point.content);
    info!("  Confidence: {:.3}", point.confidence.value());
    info!("  Category: {:?}", point.category);
    info!("  ID: {}", point.id);
    
    // Resolve
    let start_time = std::time::Instant::now();
    let result = engine.resolve(point).await;
    let processing_time = start_time.elapsed();
    
    match result {
        Ok(outcome) => {
            info!("✅ Resolution successful!");
            info!("🎯 Action: {:?}", outcome.action);
            info!("🔥 Confidence: {:.3}", outcome.confidence.value());
            info!("💭 Reasoning: {}", outcome.reasoning);
            info!("⏱️  Processing time: {:?}", processing_time);
            info!("🔄 Used fallback: {}", outcome.used_fallback);
            
            // Evidence summary
            let evidence = &outcome.evidence_summary;
            info!("📈 Evidence Summary:");
            info!("  Affirmations: {} (strength: {:.3})", evidence.affirmation_count, evidence.affirmation_strength);
            info!("  Contentions: {} (strength: {:.3})", evidence.contention_count, evidence.contention_strength);
            info!("  Evidence quality: {:.3}", evidence.evidence_quality);
            
            if let Some(ref strongest) = evidence.strongest_affirmation {
                info!("  Strongest support: {}", strongest);
            }
            if let Some(ref strongest) = evidence.strongest_contention {
                info!("  Strongest challenge: {}", strongest);
            }
        },
        Err(e) => {
            error!("❌ Resolution failed: {}", e);
            error!("⏱️  Processing time: {:?}", processing_time);
        }
    }
    
    Ok(())
}

/// Run an interactive resolution session
async fn run_interactive_session(engine: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    info!("🎮 Starting interactive session");
    info!("Type points to resolve, or 'quit' to exit");
    
    loop {
        print!("\n🤖 Enter point content (or 'quit'): ");
        std::io::Write::flush(&mut std::io::stdout())?;
        
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        if input.to_lowercase() == "quit" {
            info!("👋 Goodbye!");
            break;
        }
        
        // Quick resolution with default parameters
        let point = Point::new(input, 0.8);
        
        match engine.resolve_with_timeout(point, Duration::from_millis(100)).await {
            Ok(outcome) => {
                println!("✅ Action: {:?}", outcome.action);
                println!("🔥 Confidence: {:.3}", outcome.confidence.value());
                println!("💭 Reasoning: {}", outcome.reasoning);
            },
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
    }
    
    Ok(())
}

/// Run benchmark scenarios
async fn run_benchmark(
    engine: &Engine,
    iterations: usize,
    concurrent: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("🏃 Running benchmark: {} iterations, {} concurrent", iterations, concurrent);
    
    let test_points = vec![
        ("safe to merge left", 0.8, PointCategory::Safety),
        ("gap detected ahead", 0.9, PointCategory::Observation),
        ("vehicle approaching fast", 0.7, PointCategory::Prediction),
        ("emergency brake required", 0.95, PointCategory::Safety),
        ("lane change recommended", 0.6, PointCategory::Inference),
        ("traffic light turning red", 0.85, PointCategory::Observation),
        ("pedestrian crossing street", 0.9, PointCategory::Safety),
        ("weather conditions poor", 0.7, PointCategory::Pattern),
    ];
    
    let start_time = std::time::Instant::now();
    let mut successful = 0;
    let mut failed = 0;
    
    for batch in 0..(iterations / concurrent) {
        let futures: Vec<_> = (0..concurrent)
            .map(|i| {
                let test_point = &test_points[(batch * concurrent + i) % test_points.len()];
                let point = PointBuilder::new(test_point.0)
                    .confidence(test_point.1)
                    .category(test_point.2.clone())
                    .build();
                engine.resolve(point)
            })
            .collect();
        
        let results = futures::future::join_all(futures).await;
        
        for result in results {
            if result.is_ok() {
                successful += 1;
            } else {
                failed += 1;
            }
        }
        
        if batch % 10 == 0 {
            info!("Progress: {}/{} batches completed", batch, iterations / concurrent);
        }
    }
    
    let total_time = start_time.elapsed();
    let metrics = engine.get_metrics().await;
    
    info!("📊 Benchmark Results:");
    info!("  Total resolutions: {}", successful + failed);
    info!("  Successful: {} ({:.1}%)", successful, (successful as f64 / (successful + failed) as f64) * 100.0);
    info!("  Failed: {} ({:.1}%)", failed, (failed as f64 / (successful + failed) as f64) * 100.0);
    info!("  Total time: {:?}", total_time);
    info!("  Average time per resolution: {:?}", total_time / (successful + failed) as u32);
    info!("  Resolutions per second: {:.1}", (successful + failed) as f64 / total_time.as_secs_f64());
    
    info!("📈 Engine Metrics:");
    info!("  Emergency fallbacks: {}", metrics.emergency_fallbacks);
    info!("  Cache hits: {} / {} ({:.1}%)", 
          metrics.cache_hits, 
          metrics.cache_hits + metrics.cache_misses,
          (metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64) * 100.0);
    info!("  Average resolution time: {:?}", metrics.average_resolution_time);
    
    Ok(())
}

/// Demonstrate certificate usage
async fn demonstrate_certificates(
    engine: &Engine,
    action: String,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("📜 Demonstrating certificate system");
    
    // Create some example certificates
    let safety_cert = CertificateBuilder::new("emergency_brake_cert")
        .pattern(CertificatePattern::And(vec![
            CertificatePattern::Category(PointCategory::Safety),
            CertificatePattern::ConfidenceRange { min: 0.8, max: 1.0 },
        ]))
        .resolution_logic(ResolutionLogic::DirectAction(
            Action::Emergency(gusheshe::types::EmergencyAction::EmergencyBrake)
        ))
        .validity_duration(Duration::from_secs(300))
        .build()?;
    
    let lane_change_cert = CertificateBuilder::new("lane_change_cert")
        .pattern(CertificatePattern::Regex("lane".to_string()))
        .resolution_logic(ResolutionLogic::ConfidenceLookup(vec![
            (0.0, 0.3, Action::Maintain),
            (0.3, 0.7, Action::Execute(gusheshe::types::DrivingAction::AdjustSpeed { 
                delta_mph: -2, 
                urgency: gusheshe::types::Urgency::Low 
            })),
            (0.7, 1.0, Action::Execute(gusheshe::types::DrivingAction::ChangeLane { 
                direction: gusheshe::types::LaneDirection::Left, 
                urgency: gusheshe::types::Urgency::Medium 
            })),
        ]))
        .build()?;
    
    info!("✅ Created {} certificates", 2);
    
    // Test points
    let test_points = vec![
        PointBuilder::new("emergency brake needed")
            .confidence(0.95)
            .category(PointCategory::Safety)
            .build(),
        PointBuilder::new("safe to change lane")
            .confidence(0.85)
            .category(PointCategory::Inference)
            .build(),
        PointBuilder::new("maintain current speed")
            .confidence(0.6)
            .category(PointCategory::Observation)
            .build(),
    ];
    
    for (i, point) in test_points.iter().enumerate() {
        info!("\n🧪 Testing point {}: '{}'", i + 1, point.content);
        
        // Check which certificates can handle this point
        let can_handle_safety = safety_cert.can_handle(point);
        let can_handle_lane = lane_change_cert.can_handle(point);
        
        info!("  Safety cert can handle: {}", can_handle_safety);
        info!("  Lane cert can handle: {}", can_handle_lane);
        
        // Apply appropriate certificate
        let context = ExecutionContext::new(Duration::from_millis(50), Confidence::new(0.6));
        
        if can_handle_safety {
            match safety_cert.apply(point, &context) {
                Ok(outcome) => {
                    info!("  ✅ Safety certificate applied: {:?}", outcome.action);
                },
                Err(e) => {
                    warn!("  ⚠️ Safety certificate failed: {}", e);
                }
            }
        } else if can_handle_lane {
            match lane_change_cert.apply(point, &context) {
                Ok(outcome) => {
                    info!("  ✅ Lane certificate applied: {:?}", outcome.action);
                },
                Err(e) => {
                    warn!("  ⚠️ Lane certificate failed: {}", e);
                }
            }
        } else {
            // Fall back to engine resolution
            match engine.resolve_with_timeout(point.clone(), Duration::from_millis(100)).await {
                Ok(outcome) => {
                    info!("  🔄 Engine fallback: {:?}", outcome.action);
                },
                Err(e) => {
                    warn!("  ❌ Engine resolution failed: {}", e);
                }
            }
        }
    }
    
    Ok(())
}

/// Show engine metrics and diagnostics
async fn show_metrics(engine: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    info!("📊 Engine Metrics and Diagnostics");
    
    let metrics = engine.get_metrics().await;
    
    info!("🔢 Resolution Statistics:");
    info!("  Total resolutions: {}", metrics.total_resolutions);
    info!("  Successful: {}", metrics.successful_resolutions);
    info!("  Failed: {}", metrics.failed_resolutions);
    info!("  Emergency fallbacks: {}", metrics.emergency_fallbacks);
    
    if metrics.total_resolutions > 0 {
        let success_rate = (metrics.successful_resolutions as f64 / metrics.total_resolutions as f64) * 100.0;
        let fallback_rate = (metrics.emergency_fallbacks as f64 / metrics.total_resolutions as f64) * 100.0;
        info!("  Success rate: {:.1}%", success_rate);
        info!("  Fallback rate: {:.1}%", fallback_rate);
    }
    
    info!("⚡ Performance Metrics:");
    info!("  Average resolution time: {:?}", metrics.average_resolution_time);
    info!("  Cache hits: {}", metrics.cache_hits);
    info!("  Cache misses: {}", metrics.cache_misses);
    
    if metrics.cache_hits + metrics.cache_misses > 0 {
        let cache_hit_rate = (metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64) * 100.0;
        info!("  Cache hit rate: {:.1}%", cache_hit_rate);
    }
    
    info!("🔧 System Information:");
    info!("  Gusheshe version: {}", gusheshe::VERSION);
    info!("  Default timeout: 100ms");
    info!("  Default confidence threshold: 0.65");
    
    // Test a few quick resolutions to get fresh metrics
    info!("\n🧪 Running quick diagnostic test...");
    let test_points = vec![
        Point::new("diagnostic test 1", 0.8),
        Point::new("diagnostic test 2", 0.6),
        Point::new("diagnostic test 3", 0.9),
    ];
    
    let start_time = std::time::Instant::now();
    for point in test_points {
        let _ = engine.resolve_with_timeout(point, Duration::from_millis(50)).await;
    }
    let test_time = start_time.elapsed();
    
    info!("  Diagnostic test completed in: {:?}", test_time);
    info!("✅ System operational");
    
    Ok(())
} 