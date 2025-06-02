//! # Gusheshe - Hybrid Resolution Engine
//!
//! Gusheshe is a real-time hybrid resolution engine that combines logical programming,
//! fuzzy logic, and Bayesian inference for fast, verifiable decision making.
//!
//! ## Core Concepts
//!
//! - **Points**: Semantic units with uncertainty and confidence levels
//! - **Resolutions**: Debate platforms that process affirmations and contentions
//! - **Certificates**: Pre-compiled, verifiable execution units
//! - **Hybrid Processing**: Dynamic switching between deterministic and probabilistic modes
//!
//! ## Example Usage
//!
//! ```rust
//! use gusheshe::{Engine, Certificate, Point, Resolution};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), gusheshe::Error> {
//!     let engine = Engine::new();
//!     
//!     let point = Point::new("safe_gap_detected", 0.85);
//!     let resolution = engine.resolve_with_timeout(point, std::time::Duration::from_millis(100)).await?;
//!     
//!     println!("Decision: {:?}, Confidence: {}", resolution.action, resolution.confidence);
//!     Ok(())
//! }
//! ```

pub mod certificate;
pub mod engine;
pub mod point;
pub mod resolution;
pub mod logical;
pub mod fuzzy;
pub mod bayesian;
pub mod types;
pub mod error;

// Re-export main types for convenience
pub use certificate::{Certificate, CertificateBuilder};
pub use engine::{Engine, EngineConfig};
pub use point::{Point, PointBuilder};
pub use resolution::{Resolution, ResolutionStrategy, ResolutionOutcome};
pub use types::{Evidence, Affirmation, Contention, Confidence};
pub use error::{Error, Result};

/// Current version of the Gusheshe engine
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default timeout for resolution operations (milliseconds)
pub const DEFAULT_TIMEOUT_MS: u64 = 100;

/// Default confidence threshold for accepting resolutions
pub const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.65; 