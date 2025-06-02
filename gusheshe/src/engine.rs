//! Main Gusheshe Engine - orchestrates hybrid resolution processing

use crate::types::{Confidence, ExecutionContext, ProcessingMode};
use crate::point::{Point, PointCategory};
use crate::resolution::{Resolution, ResolutionStrategy, ResolutionOutcome};
use crate::logical::LogicalEngine;
use crate::fuzzy::FuzzyEngine;
use crate::bayesian::BayesianEngine;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

/// Configuration for the Gusheshe engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Default timeout for resolutions
    pub default_timeout: Duration,
    
    /// Default confidence threshold
    pub default_confidence_threshold: f64,
    
    /// Maximum number of concurrent resolutions
    pub max_concurrent_resolutions: usize,
    
    /// Enable/disable each reasoning engine
    pub enable_logical: bool,
    pub enable_fuzzy: bool,
    pub enable_bayesian: bool,
    
    /// Default resolution strategy
    pub default_strategy: ResolutionStrategy,
    
    /// Emergency fallback configuration
    pub emergency_timeout: Duration,
    pub emergency_confidence_threshold: f64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_millis(100),
            default_confidence_threshold: 0.65,
            max_concurrent_resolutions: 10,
            enable_logical: true,
            enable_fuzzy: true,
            enable_bayesian: true,
            default_strategy: ResolutionStrategy::Adaptive,
            emergency_timeout: Duration::from_millis(10),
            emergency_confidence_threshold: 0.5,
        }
    }
}

/// The main Gusheshe hybrid resolution engine
pub struct Engine {
    /// Engine configuration
    config: EngineConfig,
    
    /// Logical reasoning engine
    logical_engine: Arc<LogicalEngine>,
    
    /// Fuzzy logic engine
    fuzzy_engine: Arc<FuzzyEngine>,
    
    /// Bayesian inference engine
    bayesian_engine: Arc<BayesianEngine>,
    
    /// Active resolutions being processed
    active_resolutions: Arc<RwLock<HashMap<Uuid, Arc<Mutex<Resolution>>>>>,
    
    /// Resolution cache for repeated queries
    resolution_cache: Arc<RwLock<HashMap<String, (ResolutionOutcome, Instant)>>>,
    
    /// Performance metrics
    metrics: Arc<Mutex<EngineMetrics>>,
}

/// Performance metrics for the engine
#[derive(Debug, Clone, Default)]
pub struct EngineMetrics {
    pub total_resolutions: u64,
    pub successful_resolutions: u64,
    pub failed_resolutions: u64,
    pub emergency_fallbacks: u64,
    pub average_resolution_time: Duration,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl Engine {
    /// Create a new Gusheshe engine with default configuration
    pub fn new() -> Self {
        Self::with_config(EngineConfig::default())
    }

    /// Create a new Gusheshe engine with custom configuration
    pub fn with_config(config: EngineConfig) -> Self {
        Self {
            logical_engine: Arc::new(LogicalEngine::new()),
            fuzzy_engine: Arc::new(FuzzyEngine::new()),
            bayesian_engine: Arc::new(BayesianEngine::new()),
            active_resolutions: Arc::new(RwLock::new(HashMap::new())),
            resolution_cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(EngineMetrics::default())),
            config,
        }
    }

    /// Resolve a point with default timeout and confidence threshold
    pub async fn resolve(&self, point: Point) -> Result<ResolutionOutcome> {
        let context = ExecutionContext::new(
            self.config.default_timeout,
            Confidence::new(self.config.default_confidence_threshold),
        );
        self.resolve_with_context(point, context).await
    }

    /// Resolve a point with custom timeout
    pub async fn resolve_with_timeout(&self, point: Point, timeout: Duration) -> Result<ResolutionOutcome> {
        let context = ExecutionContext::new(
            timeout,
            Confidence::new(self.config.default_confidence_threshold),
        );
        self.resolve_with_context(point, context).await
    }

    /// Resolve a point with full execution context
    pub async fn resolve_with_context(&self, point: Point, context: ExecutionContext) -> Result<ResolutionOutcome> {
        let start_time = Instant::now();
        
        // Check cache first
        let cache_key = self.create_cache_key(&point);
        if let Some(cached_outcome) = self.check_cache(&cache_key).await {
            self.increment_cache_hits().await;
            return Ok(cached_outcome);
        }
        self.increment_cache_misses().await;

        // Determine resolution strategy
        let strategy = self.determine_strategy(&point, &context);
        
        // Create resolution
        let mut resolution = Resolution::new(&point, strategy, context.timeout);
        
        // Gather evidence using hybrid engines
        self.gather_evidence(&mut resolution, &point, &context).await?;
        
        // Store resolution as active
        let resolution_id = resolution.id;
        let resolution_arc = Arc::new(Mutex::new(resolution));
        {
            let mut active = self.active_resolutions.write().await;
            active.insert(resolution_id, resolution_arc.clone());
        }

        // Resolve
        let outcome = {
            let mut resolution_guard = resolution_arc.lock().await;
            resolution_guard.resolve(&context).await
        };

        // Clean up active resolution
        {
            let mut active = self.active_resolutions.write().await;
            active.remove(&resolution_id);
        }

        // Handle outcome
        match outcome {
            Ok(outcome) => {
                // Cache successful outcome
                self.cache_outcome(&cache_key, &outcome).await;
                
                // Update metrics
                self.update_metrics(true, start_time.elapsed(), outcome.used_fallback).await;
                
                Ok(outcome)
            },
            Err(e) => {
                // Update failure metrics
                self.update_metrics(false, start_time.elapsed(), true).await;
                
                // Try emergency fallback if error requires it
                if e.requires_emergency_fallback() {
                    self.emergency_fallback(&point, &e).await
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Resolve multiple points concurrently
    pub async fn resolve_batch(&self, points: Vec<Point>) -> Vec<Result<ResolutionOutcome>> {
        let futures = points.into_iter().map(|point| self.resolve(point));
        futures::future::join_all(futures).await
    }

    /// Cancel an active resolution
    pub async fn cancel_resolution(&self, resolution_id: Uuid) -> Result<()> {
        let mut active = self.active_resolutions.write().await;
        if active.remove(&resolution_id).is_some() {
            Ok(())
        } else {
            Err(Error::no_resolution(resolution_id.to_string()))
        }
    }

    /// Get current engine metrics
    pub async fn get_metrics(&self) -> EngineMetrics {
        self.metrics.lock().await.clone()
    }

    /// Clear resolution cache
    pub async fn clear_cache(&self) {
        let mut cache = self.resolution_cache.write().await;
        cache.clear();
    }

    /// Determine the appropriate resolution strategy for a point
    fn determine_strategy(&self, point: &Point, context: &ExecutionContext) -> ResolutionStrategy {
        match self.config.default_strategy {
            ResolutionStrategy::Adaptive => {
                // Choose strategy based on point characteristics and context
                match (&point.category, context.remaining_time()) {
                    // Safety-critical points get conservative treatment
                    (PointCategory::Safety, _) => ResolutionStrategy::Conservative,
                    
                    // Time pressure leads to emergency mode
                    (_, remaining) if remaining < Duration::from_millis(20) => ResolutionStrategy::Emergency,
                    
                    // High-confidence observations can use maximum likelihood
                    (PointCategory::Observation, _) if point.confidence.value() > 0.8 => {
                        ResolutionStrategy::MaximumLikelihood
                    },
                    
                    // Complex inferences benefit from Bayesian approach
                    (PointCategory::Inference, _) => ResolutionStrategy::Bayesian,
                    
                    // Predictions are exploratory by nature
                    (PointCategory::Prediction, _) => ResolutionStrategy::Exploratory,
                    
                    // Default to Bayesian for balanced approach
                    _ => ResolutionStrategy::Bayesian,
                }
            },
            strategy => strategy, // Use configured strategy
        }
    }

    /// Gather evidence for a resolution using hybrid engines
    async fn gather_evidence(&self, resolution: &mut Resolution, point: &Point, context: &ExecutionContext) -> Result<()> {
        // Logical engine: Extract rule-based evidence
        if self.config.enable_logical && context.has_time_for(Duration::from_millis(10)) {
            if let Ok(logical_evidence) = self.logical_engine.gather_evidence(point).await {
                for evidence in logical_evidence {
                    resolution.add_affirmation(evidence);
                }
            }
        }

        // Fuzzy engine: Generate fuzzy membership evidence
        if self.config.enable_fuzzy && context.has_time_for(Duration::from_millis(15)) {
            if let Ok(fuzzy_evidence) = self.fuzzy_engine.gather_evidence(point).await {
                for evidence in fuzzy_evidence {
                    resolution.add_affirmation(evidence);
                }
            }
        }

        // Bayesian engine: Generate probabilistic evidence
        if self.config.enable_bayesian && context.has_time_for(Duration::from_millis(20)) {
            if let Ok(bayesian_evidence) = self.bayesian_engine.gather_evidence(point).await {
                for evidence in bayesian_evidence {
                    resolution.add_affirmation(evidence);
                }
            }
        }

        // Also gather contentions (challenging evidence)
        self.gather_contentions(resolution, point, context).await?;

        Ok(())
    }

    /// Gather contentions (challenging evidence) for a resolution
    async fn gather_contentions(&self, resolution: &mut Resolution, point: &Point, context: &ExecutionContext) -> Result<()> {
        // Look for contradictory evidence in the knowledge base
        // This is a simplified implementation - in practice would be more sophisticated
        
        if point.content.contains("safe") && context.has_time_for(Duration::from_millis(5)) {
            // Generate synthetic contention for testing
            let contention_evidence = crate::types::Evidence {
                id: Uuid::new_v4(),
                content: "Potential blind spot detected".to_string(),
                evidence_type: crate::types::EvidenceType::Sensor,
                confidence: Confidence::new(0.3),
                source: "synthetic".to_string(),
                timestamp: Instant::now(),
                validity_window: Duration::from_secs(5),
            };
            
            let contention = crate::types::Contention {
                evidence: contention_evidence,
                impact: 0.4,
                uncertainty: 0.6,
            };
            
            resolution.add_contention(contention);
        }

        Ok(())
    }

    /// Emergency fallback when normal resolution fails
    async fn emergency_fallback(&self, point: &Point, error: &Error) -> Result<ResolutionOutcome> {
        // Simple, fast emergency decision
        let outcome = ResolutionOutcome {
            action: crate::types::Action::Emergency(crate::types::EmergencyAction::EmergencyBrake),
            confidence: Confidence::new(self.config.emergency_confidence_threshold),
            reasoning: format!("Emergency fallback due to: {}", error),
            evidence_summary: crate::resolution::EvidenceSummary {
                affirmation_strength: 0.0,
                contention_strength: 1.0,
                affirmation_count: 0,
                contention_count: 1,
                strongest_affirmation: None,
                strongest_contention: Some("System error".to_string()),
                evidence_quality: 0.1,
            },
            timestamp: Instant::now(),
            processing_time: Duration::from_millis(1), // Very fast
            used_fallback: true,
        };

        self.increment_emergency_fallbacks().await;
        Ok(outcome)
    }

    /// Create a cache key for a point
    fn create_cache_key(&self, point: &Point) -> String {
        format!("{}:{:.3}", point.content, point.confidence.value())
    }

    /// Check if we have a cached outcome for this key
    async fn check_cache(&self, key: &str) -> Option<ResolutionOutcome> {
        let cache = self.resolution_cache.read().await;
        if let Some((outcome, timestamp)) = cache.get(key) {
            // Check if cache entry is still valid (1 second TTL)
            if timestamp.elapsed() < Duration::from_secs(1) {
                return Some(outcome.clone());
            }
        }
        None
    }

    /// Cache a resolution outcome
    async fn cache_outcome(&self, key: &str, outcome: &ResolutionOutcome) {
        let mut cache = self.resolution_cache.write().await;
        cache.insert(key.to_string(), (outcome.clone(), Instant::now()));
        
        // Simple cache cleanup - remove entries older than 1 minute
        let cutoff = Instant::now() - Duration::from_secs(60);
        cache.retain(|_, (_, timestamp)| *timestamp > cutoff);
    }

    /// Update engine metrics
    async fn update_metrics(&self, success: bool, processing_time: Duration, used_fallback: bool) {
        let mut metrics = self.metrics.lock().await;
        metrics.total_resolutions += 1;
        
        if success {
            metrics.successful_resolutions += 1;
        } else {
            metrics.failed_resolutions += 1;
        }
        
        if used_fallback {
            metrics.emergency_fallbacks += 1;
        }
        
        // Update average processing time (simple moving average)
        let total_time = metrics.average_resolution_time.as_nanos() as f64 * (metrics.total_resolutions - 1) as f64;
        let new_average = (total_time + processing_time.as_nanos() as f64) / metrics.total_resolutions as f64;
        metrics.average_resolution_time = Duration::from_nanos(new_average as u64);
    }

    /// Increment cache hit counter
    async fn increment_cache_hits(&self) {
        let mut metrics = self.metrics.lock().await;
        metrics.cache_hits += 1;
    }

    /// Increment cache miss counter
    async fn increment_cache_misses(&self) {
        let mut metrics = self.metrics.lock().await;
        metrics.cache_misses += 1;
    }

    /// Increment emergency fallback counter
    async fn increment_emergency_fallbacks(&self) {
        let mut metrics = self.metrics.lock().await;
        metrics.emergency_fallbacks += 1;
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point::PointBuilder;

    #[tokio::test]
    async fn test_engine_creation() {
        let engine = Engine::new();
        let metrics = engine.get_metrics().await;
        assert_eq!(metrics.total_resolutions, 0);
    }

    #[tokio::test]
    async fn test_basic_resolution() {
        let engine = Engine::new();
        let point = PointBuilder::new("safe to proceed")
            .confidence(0.8)
            .category(PointCategory::Safety)
            .build();

        let result = engine.resolve(point).await;
        assert!(result.is_ok());
        
        let outcome = result.unwrap();
        assert!(outcome.confidence.value() > 0.0);
    }

    #[tokio::test]
    async fn test_batch_resolution() {
        let engine = Engine::new();
        let points = vec![
            Point::new("point 1", 0.8),
            Point::new("point 2", 0.6),
            Point::new("point 3", 0.9),
        ];

        let results = engine.resolve_batch(points).await;
        assert_eq!(results.len(), 3);
        
        for result in results {
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let engine = Engine::new();
        let point = Point::new("cached point", 0.8);

        // First resolution should be cache miss
        let _result1 = engine.resolve(point.clone()).await;
        let metrics1 = engine.get_metrics().await;
        assert_eq!(metrics1.cache_misses, 1);
        assert_eq!(metrics1.cache_hits, 0);

        // Second resolution should be cache hit
        let _result2 = engine.resolve(point).await;
        let metrics2 = engine.get_metrics().await;
        assert_eq!(metrics2.cache_hits, 1);
    }

    #[tokio::test]
    async fn test_timeout_handling() {
        let engine = Engine::new();
        let point = Point::new("timeout test", 0.8);

        // Very short timeout should trigger emergency fallback
        let result = engine.resolve_with_timeout(point, Duration::from_millis(1)).await;
        assert!(result.is_ok());
        
        let outcome = result.unwrap();
        assert!(outcome.used_fallback);
    }
} 