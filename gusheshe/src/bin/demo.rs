//! Gusheshe Demo - Interactive Hybrid Resolution Engine Demonstration
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
#[command(name = "gusheshe-demo")]
#[command(about = "Gusheshe Hybrid Resolution Engine - Real-time Decision Making Demo")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Resolve a single decision point
    Resolve {
        /// The point content to resolve
        content: String,
        /// Confidence level (0.0-1.0)
        #[arg(short, long, default_value = "0.8")]
        confidence: f64,
        /// Point category
        #[arg(short = 't', long, default_value = "observation")]
        category: String,
        /// Processing mode
        #[arg(short, long, default_value = "standard")]
        mode: String,
    },
    /// Start interactive session
    Interactive,
    /// Run performance benchmarks
    Benchmark {
        /// Number of iterations
        #[arg(short, long, default_value = "1000")]
        iterations: usize,
        /// Concurrent resolution count
        #[arg(short, long, default_value = "10")]
        concurrent: usize,
    },
    /// Demonstrate certificate usage
    Certificate,
    /// Show engine metrics
    Metrics,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    // Create engine with default configuration
    let config = EngineConfig::default();
    let engine = Engine::new(config);
    
    match cli.command {
        Commands::Resolve { content, confidence, category, mode } => {
            resolve_point(&engine, &content, confidence, &category, &mode).await?;
        }
        Commands::Interactive => {
            interactive_session(&engine).await?;
        }
        Commands::Benchmark { iterations, concurrent } => {
            run_benchmark(&engine, iterations, concurrent).await?;
        }
        Commands::Certificate => {
            demonstrate_certificates(&engine).await?;
        }
        Commands::Metrics => {
            show_metrics(&engine).await?;
        }
    }
    
    Ok(())
}

async fn resolve_point(
    engine: &Engine,
    content: &str,
    confidence: f64,
    category: &str,
    mode: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Resolving point: {}", content);
    
    let point_category = match category {
        "observation" => PointCategory::Observation,
        "inference" => PointCategory::Inference,
        "prediction" => PointCategory::Prediction,
        "safety" => PointCategory::Safety,
        "performance" => PointCategory::Performance,
        "maintenance" => PointCategory::Maintenance,
        _ => PointCategory::Observation,
    };
    
    let processing_mode = match mode {
        "fast" => ProcessingMode::Fast,
        "thorough" => ProcessingMode::Thorough,
        "emergency" => ProcessingMode::Emergency,
        _ => ProcessingMode::Standard,
    };
    
    let point = PointBuilder::new()
        .content(content.to_string())
        .confidence(Confidence::new(confidence))
        .category(point_category)
        .build();
    
    let context = ExecutionContext::new()
        .with_mode(processing_mode)
        .with_timeout(Duration::from_millis(100));
    
    let outcome = engine.resolve_with_context(point, context).await?;
    
    println!("\n=== Resolution Outcome ===");
    println!("Action: {:?}", outcome.action);
    println!("Confidence: {:.3}", outcome.confidence.value());
    println!("Strategy: {:?}", outcome.strategy);
    println!("Processing Time: {:?}", outcome.processing_time);
    println!("Reasoning: {}", outcome.reasoning);
    
    if !outcome.supporting_evidence.is_empty() {
        println!("\nSupporting Evidence:");
        for evidence in &outcome.supporting_evidence {
            println!("  - {}", evidence);
        }
    }
    
    if !outcome.challenging_evidence.is_empty() {
        println!("\nChallenging Evidence:");
        for evidence in &outcome.challenging_evidence {
            println!("  - {}", evidence);
        }
    }
    
    Ok(())
}

async fn interactive_session(engine: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Gusheshe Interactive Session ===");
    println!("Enter decision points to resolve. Type 'quit' to exit.\n");
    
    loop {
        print!("gusheshe> ");
        use std::io::{self, Write};
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input == "quit" || input == "exit" {
            break;
        }
        
        if input.is_empty() {
            continue;
        }
        
        let point = PointBuilder::new()
            .content(input.to_string())
            .confidence(Confidence::new(0.8))
            .category(PointCategory::Observation)
            .build();
        
        match engine.resolve(point).await {
            Ok(outcome) => {
                println!("Action: {:?} (confidence: {:.3})", 
                    outcome.action, outcome.confidence.value());
                println!("Reasoning: {}\n", outcome.reasoning);
            }
            Err(e) => {
                eprintln!("Error: {}\n", e);
            }
        }
    }
    
    println!("Session ended.");
    Ok(())
}

async fn run_benchmark(
    engine: &Engine,
    iterations: usize,
    concurrent: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    
    println!("=== Gusheshe Performance Benchmark ===");
    println!("Iterations: {}, Concurrent: {}\n", iterations, concurrent);
    
    let test_points = vec![
        "vehicle approaching fast",
        "pedestrian crossing ahead",
        "traffic light changing",
        "emergency vehicle behind",
        "construction zone detected",
        "weather conditions changing",
        "road surface degraded",
        "low fuel warning",
    ];
    
    let start = Instant::now();
    let mut handles = Vec::new();
    
    for i in 0..concurrent {
        let engine = engine.clone();
        let test_points = test_points.clone();
        let batch_size = iterations / concurrent;
        
        let handle = tokio::spawn(async move {
            let mut local_times = Vec::new();
            
            for j in 0..batch_size {
                let content = &test_points[j % test_points.len()];
                let point = PointBuilder::new()
                    .content(content.to_string())
                    .confidence(Confidence::new(0.8))
                    .category(PointCategory::Observation)
                    .build();
                
                let start_time = Instant::now();
                let _outcome = engine.resolve(point).await.unwrap();
                local_times.push(start_time.elapsed());
            }
            
            local_times
        });
        
        handles.push(handle);
    }
    
    let mut all_times = Vec::new();
    for handle in handles {
        let times = handle.await?;
        all_times.extend(times);
    }
    
    let total_time = start.elapsed();
    
    // Calculate statistics
    all_times.sort();
    let count = all_times.len();
    let mean = all_times.iter().sum::<Duration>() / count as u32;
    let median = all_times[count / 2];
    let p95 = all_times[(count as f64 * 0.95) as usize];
    let p99 = all_times[(count as f64 * 0.99) as usize];
    
    println!("Total time: {:?}", total_time);
    println!("Throughput: {:.2} resolutions/sec", count as f64 / total_time.as_secs_f64());
    println!("Mean latency: {:?}", mean);
    println!("Median latency: {:?}", median);
    println!("95th percentile: {:?}", p95);
    println!("99th percentile: {:?}", p99);
    
    Ok(())
}

async fn demonstrate_certificates(engine: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Certificate Demonstration ===\n");
    
    // Create a sample certificate for emergency scenarios
    let emergency_cert = CertificateBuilder::new()
        .name("emergency_brake")
        .pattern(CertificatePattern::ContentMatch("emergency".to_string()))
        .logic(ResolutionLogic::DirectAction(Action::EmergencyBrake))
        .confidence_threshold(0.9)
        .build();
    
    engine.register_certificate(emergency_cert).await?;
    
    println!("Registered emergency brake certificate");
    
    // Test with emergency scenario
    let emergency_point = PointBuilder::new()
        .content("emergency vehicle approaching")
        .confidence(Confidence::new(0.95))
        .category(PointCategory::Safety)
        .build();
    
    let outcome = engine.resolve(emergency_point).await?;
    
    println!("Emergency scenario resolved:");
    println!("Action: {:?}", outcome.action);
    println!("Used certificate: {}", outcome.used_certificate);
    
    Ok(())
}

async fn show_metrics(engine: &Engine) -> Result<(), Box<dyn std::error::Error>> {
    let metrics = engine.get_metrics().await?;
    
    println!("=== Engine Metrics ===");
    println!("Total resolutions: {}", metrics.total_resolutions);
    println!("Success rate: {:.2}%", metrics.success_rate * 100.0);
    println!("Average processing time: {:?}", metrics.average_processing_time);
    println!("Cache hit rate: {:.2}%", metrics.cache_hit_rate * 100.0);
    println!("Active strategies: {:?}", metrics.active_strategies);
    
    Ok(())
} 