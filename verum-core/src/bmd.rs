//! BMD System - Biological Maxwell Demon pattern recognition and memory curation
//!
//! This module implements the BMD system that selects from curated "good memories"
//! and performs pattern recognition for optimal system states.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

use crate::verum_system::VerumError;
use crate::entropy::OptimizedState;

/// Configuration for BMD system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BMDConfig {
    /// Maximum number of memories to store
    pub max_memories: usize,
    
    /// Quality threshold for memory inclusion
    pub quality_threshold: f64,
    
    /// Pattern matching configuration
    pub pattern_config: PatternConfig,
    
    /// Memory curation configuration
    pub curation_config: CurationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    /// Similarity threshold for pattern matching
    pub similarity_threshold: f64,
    
    /// Maximum patterns to return
    pub max_matches: usize,
    
    /// Pattern matching algorithm parameters
    pub algorithm_params: PatternAlgorithmParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAlgorithmParams {
    /// Weight for frequency domain matching
    pub frequency_weight: f64,
    
    /// Weight for amplitude matching
    pub amplitude_weight: f64,
    
    /// Weight for phase coherence matching
    pub phase_weight: f64,
    
    /// Weight for entropy matching
    pub entropy_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurationConfig {
    /// Quality scoring weights
    pub scoring_weights: ScoringWeights,
    
    /// Memory pruning parameters
    pub pruning_params: PruningParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    /// Weight for system performance outcome
    pub performance_weight: f64,
    
    /// Weight for comfort optimization
    pub comfort_weight: f64,
    
    /// Weight for energy efficiency
    pub efficiency_weight: f64,
    
    /// Weight for safety metrics
    pub safety_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningParams {
    /// Age threshold for memory pruning (in hours)
    pub age_threshold_hours: f64,
    
    /// Minimum quality for memory retention
    pub min_quality_retention: f64,
    
    /// Maximum memory bank size before forced pruning
    pub max_bank_size: usize,
}

/// Curated collection of optimal system states
#[derive(Debug, Clone)]
pub struct GoodMemoryBank {
    memories: HashMap<String, GoodMemory>,
    quality_index: Vec<(String, f64)>, // Sorted by quality score
    next_id: u64,
}

impl GoodMemoryBank {
    pub fn new() -> Self {
        Self {
            memories: HashMap::new(),
            quality_index: Vec::new(),
            next_id: 0,
        }
    }
    
    pub fn add_memory(&mut self, memory: GoodMemory) {
        let id = format!("memory_{}", self.next_id);
        self.next_id += 1;
        
        let quality = memory.quality_score;
        self.memories.insert(id.clone(), memory);
        self.quality_index.push((id, quality));
        
        // Keep index sorted by quality (descending)
        self.quality_index.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    }
    
    pub fn get_best_memories(&self, count: usize) -> Vec<&GoodMemory> {
        self.quality_index
            .iter()
            .take(count)
            .filter_map(|(id, _)| self.memories.get(id))
            .collect()
    }
    
    pub fn prune_low_quality_memories(&mut self) {
        // Remove memories below threshold
        let threshold = 0.6; // Should come from config
        self.quality_index.retain(|(id, quality)| {
            if *quality < threshold {
                self.memories.remove(id);
                false
            } else {
                true
            }
        });
    }
    
    pub fn len(&self) -> usize {
        self.memories.len()
    }
}

/// A curated "good memory" representing an optimal system state
#[derive(Debug, Clone)]
pub struct GoodMemory {
    pub state_snapshot: StateSnapshot,
    pub quality_score: f64,
    pub timestamp: std::time::Instant,
    pub context: MemoryContext,
    pub outcomes: Vec<MemoryOutcome>,
}

impl GoodMemory {
    pub fn from_snapshot(snapshot: &StateSnapshot, quality_score: f64) -> Self {
        Self {
            state_snapshot: snapshot.clone(),
            quality_score,
            timestamp: std::time::Instant::now(),
            context: MemoryContext::default(),
            outcomes: Vec::new(),
        }
    }
}

/// Context information for when the memory was created
#[derive(Debug, Clone)]
pub struct MemoryContext {
    pub road_conditions: String,
    pub weather_conditions: String,
    pub traffic_level: f64,
    pub driving_mode: String,
}

impl Default for MemoryContext {
    fn default() -> Self {
        Self {
            road_conditions: "normal".to_string(),
            weather_conditions: "clear".to_string(),
            traffic_level: 0.5,
            driving_mode: "normal".to_string(),
        }
    }
}

/// Outcome metrics for a memory
#[derive(Debug, Clone)]
pub struct MemoryOutcome {
    pub metric_name: String,
    pub value: f64,
    pub target_value: f64,
    pub improvement: f64,
}

/// State snapshot for pattern matching
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub oscillation_signature: Vec<f64>,
    pub entropy_level: f64,
    pub comfort_metrics: ComfortMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub environmental_context: EnvironmentalContext,
}

impl StateSnapshot {
    pub fn from_context(context: &crate::verum_system::DecisionContext) -> Self {
        Self {
            oscillation_signature: context.oscillations.frequency_components
                .iter()
                .map(|f| f.amplitude)
                .collect(),
            entropy_level: context.optimized_state.original_entropy,
            comfort_metrics: ComfortMetrics::default(),
            performance_metrics: PerformanceMetrics::default(),
            environmental_context: EnvironmentalContext::from_environment(&context.environment),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComfortMetrics {
    pub vibration_level: f64,
    pub noise_level: f64,
    pub temperature_comfort: f64,
    pub seat_comfort: f64,
}

impl Default for ComfortMetrics {
    fn default() -> Self {
        Self {
            vibration_level: 0.0,
            noise_level: 0.0,
            temperature_comfort: 0.8,
            seat_comfort: 0.8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub fuel_efficiency: f64,
    pub response_time: f64,
    pub system_load: f64,
    pub decision_accuracy: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            fuel_efficiency: 0.8,
            response_time: 10.0, // ms
            system_load: 0.3,
            decision_accuracy: 0.9,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnvironmentalContext {
    pub traffic_density: f64,
    pub road_type: String,
    pub weather_conditions: String,
    pub time_of_day: String,
}

impl EnvironmentalContext {
    pub fn from_environment(env: &crate::verum_system::EnvironmentState) -> Self {
        Self {
            traffic_density: env.traffic_density,
            road_type: "highway".to_string(), // Would be derived from road_conditions
            weather_conditions: "clear".to_string(), // Would be derived from weather_effects
            time_of_day: "day".to_string(), // Would be derived from current time
        }
    }
}

/// Pattern recognition engine for finding similar states
pub struct PatternRecognizer {
    config: PatternConfig,
    similarity_cache: Arc<RwLock<HashMap<String, f64>>>,
}

impl PatternRecognizer {
    pub fn new(config: PatternConfig) -> Self {
        Self {
            config,
            similarity_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Find pattern matches in memory bank
    pub async fn find_matches(
        &self,
        state: &OptimizedState,
        memories: &GoodMemoryBank
    ) -> Result<Vec<PatternMatch>, VerumError> {
        let mut matches = Vec::new();
        
        // Get signature from optimized state
        let query_signature = self.extract_pattern_signature(state).await?;
        
        // Compare against all memories
        for memory in memories.get_best_memories(self.config.max_matches * 2) {
            let memory_signature = self.extract_memory_signature(memory).await?;
            let similarity = self.calculate_similarity(&query_signature, &memory_signature).await?;
            
            if similarity >= self.config.similarity_threshold {
                matches.push(PatternMatch {
                    memory_id: format!("memory_{}", memory.timestamp.elapsed().as_nanos()),
                    similarity_score: similarity,
                    confidence: similarity * 0.9, // Slightly conservative
                    matching_features: self.identify_matching_features(&query_signature, &memory_signature).await?,
                });
            }
        }
        
        // Sort by similarity score and take top matches
        matches.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        matches.truncate(self.config.max_matches);
        
        Ok(matches)
    }
    
    async fn extract_pattern_signature(&self, state: &OptimizedState) -> Result<PatternSignature, VerumError> {
        Ok(PatternSignature {
            frequency_spectrum: state.optimized_oscillations.frequency_components
                .iter()
                .map(|f| f.frequency_hz)
                .collect(),
            amplitude_profile: state.optimized_oscillations.frequency_components
                .iter()
                .map(|f| f.amplitude)
                .collect(),
            entropy_level: state.original_entropy,
            coherence_measure: state.control_precision,
        })
    }
    
    async fn extract_memory_signature(&self, memory: &GoodMemory) -> Result<PatternSignature, VerumError> {
        Ok(PatternSignature {
            frequency_spectrum: memory.state_snapshot.oscillation_signature.clone(),
            amplitude_profile: memory.state_snapshot.oscillation_signature.clone(),
            entropy_level: memory.state_snapshot.entropy_level,
            coherence_measure: memory.quality_score,
        })
    }
    
    async fn calculate_similarity(
        &self,
        sig1: &PatternSignature,
        sig2: &PatternSignature
    ) -> Result<f64, VerumError> {
        let params = &self.config.algorithm_params;
        
        // Frequency domain similarity
        let freq_sim = self.calculate_vector_similarity(&sig1.frequency_spectrum, &sig2.frequency_spectrum);
        
        // Amplitude similarity
        let amp_sim = self.calculate_vector_similarity(&sig1.amplitude_profile, &sig2.amplitude_profile);
        
        // Entropy similarity
        let entropy_sim = 1.0 - (sig1.entropy_level - sig2.entropy_level).abs() / (sig1.entropy_level + sig2.entropy_level);
        
        // Coherence similarity
        let coherence_sim = 1.0 - (sig1.coherence_measure - sig2.coherence_measure).abs();
        
        // Weighted combination
        let total_similarity = 
            freq_sim * params.frequency_weight +
            amp_sim * params.amplitude_weight +
            entropy_sim * params.entropy_weight +
            coherence_sim * params.phase_weight;
        
        let total_weight = params.frequency_weight + params.amplitude_weight + 
                          params.entropy_weight + params.phase_weight;
        
        Ok(total_similarity / total_weight)
    }
    
    fn calculate_vector_similarity(&self, v1: &[f64], v2: &[f64]) -> f64 {
        if v1.is_empty() || v2.is_empty() {
            return 0.0;
        }
        
        let min_len = v1.len().min(v2.len());
        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        
        for i in 0..min_len {
            dot_product += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm1.sqrt() * norm2.sqrt())
    }
    
    async fn identify_matching_features(
        &self,
        sig1: &PatternSignature,
        sig2: &PatternSignature
    ) -> Result<Vec<String>, VerumError> {
        let mut features = Vec::new();
        
        if self.calculate_vector_similarity(&sig1.frequency_spectrum, &sig2.frequency_spectrum) > 0.8 {
            features.push("frequency_spectrum".to_string());
        }
        
        if self.calculate_vector_similarity(&sig1.amplitude_profile, &sig2.amplitude_profile) > 0.8 {
            features.push("amplitude_profile".to_string());
        }
        
        if (sig1.entropy_level - sig2.entropy_level).abs() < 0.1 {
            features.push("entropy_level".to_string());
        }
        
        if (sig1.coherence_measure - sig2.coherence_measure).abs() < 0.1 {
            features.push("coherence_measure".to_string());
        }
        
        Ok(features)
    }
}

/// Pattern signature for similarity matching
#[derive(Debug, Clone)]
pub struct PatternSignature {
    pub frequency_spectrum: Vec<f64>,
    pub amplitude_profile: Vec<f64>,
    pub entropy_level: f64,
    pub coherence_measure: f64,
}

/// Result of pattern matching
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub memory_id: String,
    pub similarity_score: f64,
    pub confidence: f64,
    pub matching_features: Vec<String>,
}

/// Memory scoring and curation system
pub struct MemoryCurator {
    config: CurationConfig,
    scoring_history: Arc<RwLock<Vec<(String, f64, std::time::Instant)>>>,
}

impl MemoryCurator {
    pub fn new(config: CurationConfig) -> Self {
        Self {
            config,
            scoring_history: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Score a memory based on multiple criteria
    pub async fn score_memory(&self, snapshot: &StateSnapshot) -> Result<f64, VerumError> {
        let weights = &self.config.scoring_weights;
        
        // Performance scoring (based on metrics)
        let performance_score = 
            snapshot.performance_metrics.fuel_efficiency * 0.3 +
            snapshot.performance_metrics.decision_accuracy * 0.4 +
            (1.0 - snapshot.performance_metrics.system_load) * 0.3;
        
        // Comfort scoring
        let comfort_score = 
            (1.0 - snapshot.comfort_metrics.vibration_level) * 0.4 +
            (1.0 - snapshot.comfort_metrics.noise_level) * 0.3 +
            snapshot.comfort_metrics.temperature_comfort * 0.15 +
            snapshot.comfort_metrics.seat_comfort * 0.15;
        
        // Efficiency scoring (inverse of entropy)
        let efficiency_score = 1.0 / (1.0 + snapshot.entropy_level);
        
        // Safety scoring (placeholder - would use real safety metrics)
        let safety_score = 0.9; // Default high safety
        
        // Weighted combination
        let total_score = 
            performance_score * weights.performance_weight +
            comfort_score * weights.comfort_weight +
            efficiency_score * weights.efficiency_weight +
            safety_score * weights.safety_weight;
        
        let total_weight = weights.performance_weight + weights.comfort_weight + 
                          weights.efficiency_weight + weights.safety_weight;
        
        let final_score = total_score / total_weight;
        
        // Record scoring history
        let mut history = self.scoring_history.write().await;
        history.push((
            format!("score_{}", std::time::Instant::now().elapsed().as_nanos()),
            final_score,
            std::time::Instant::now()
        ));
        
        Ok(final_score)
    }
    
    pub fn get_threshold(&self) -> f64 {
        self.config.pruning_params.min_quality_retention
    }
}

// Default implementations

impl Default for BMDConfig {
    fn default() -> Self {
        Self {
            max_memories: 1000,
            quality_threshold: 0.7,
            pattern_config: PatternConfig::default(),
            curation_config: CurationConfig::default(),
        }
    }
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_matches: 5,
            algorithm_params: PatternAlgorithmParams::default(),
        }
    }
}

impl Default for PatternAlgorithmParams {
    fn default() -> Self {
        Self {
            frequency_weight: 0.3,
            amplitude_weight: 0.3,
            phase_weight: 0.2,
            entropy_weight: 0.2,
        }
    }
}

impl Default for CurationConfig {
    fn default() -> Self {
        Self {
            scoring_weights: ScoringWeights::default(),
            pruning_params: PruningParams::default(),
        }
    }
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            performance_weight: 0.4,
            comfort_weight: 0.3,
            efficiency_weight: 0.2,
            safety_weight: 0.1,
        }
    }
}

impl Default for PruningParams {
    fn default() -> Self {
        Self {
            age_threshold_hours: 168.0, // 1 week
            min_quality_retention: 0.6,
            max_bank_size: 50000,
        }
    }
} 