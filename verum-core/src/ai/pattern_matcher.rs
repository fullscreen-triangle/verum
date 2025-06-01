//! Pattern Matcher
//!
//! Matches current scenarios to learned cross-domain patterns

use crate::utils::{Result, VerumError};
use super::{PersonalAIModel, ScenarioContext, BiometricState, BehavioralPattern, LearningDomain};
use super::decision_engine::DrivingDecision;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Cross-domain pattern for pattern transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDomainPattern {
    pub id: Uuid,
    pub source_domain: LearningDomain,
    pub target_domain: LearningDomain,
    pub similarity_score: f32,
    pub transfer_confidence: f32,
    pub pattern: BehavioralPattern,
}

/// Pattern matcher for finding similar cross-domain patterns
pub struct PatternMatcher {
    model: PersonalAIModel,
    pattern_database: Vec<BehavioralPattern>,
    similarity_threshold: f32,
}

impl PatternMatcher {
    /// Create new pattern matcher
    pub async fn new(model: &PersonalAIModel) -> Result<Self> {
        Ok(Self {
            model: model.clone(),
            pattern_database: Vec::new(),
            similarity_threshold: 0.7,
        })
    }
    
    /// Find similar patterns for a given scenario
    pub async fn find_similar_patterns(
        &self,
        scenario: &ScenarioContext,
        min_similarity: f32,
    ) -> Result<Vec<BehavioralPattern>> {
        let mut similar_patterns = Vec::new();
        
        // Convert scenario to feature vector for comparison
        let scenario_vector = self.scenario_to_vector(scenario);
        
        for pattern in &self.pattern_database {
            let similarity = self.calculate_similarity(&scenario_vector, &pattern.stimulus_vector);
            if similarity >= min_similarity {
                similar_patterns.push(pattern.clone());
            }
        }
        
        // Sort by similarity (highest first)
        similar_patterns.sort_by(|a, b| {
            let sim_a = self.calculate_similarity(&scenario_vector, &a.stimulus_vector);
            let sim_b = self.calculate_similarity(&scenario_vector, &b.stimulus_vector);
            sim_b.partial_cmp(&sim_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(similar_patterns)
    }
    
    /// Add new pattern to database
    pub async fn add_pattern(
        &mut self,
        scenario: ScenarioContext,
        biometric_state: BiometricState,
        action_taken: DrivingDecision,
        success: bool,
    ) -> Result<()> {
        let pattern = BehavioralPattern {
            id: Uuid::new_v4(),
            domain: LearningDomain::Driving,
            scenario_type: scenario.scenario_type.clone(),
            stimulus_vector: self.scenario_to_vector(&scenario),
            response_vector: self.action_to_vector(&action_taken),
            success_rate: if success { 1.0 } else { 0.0 },
            stress_level: (biometric_state.heart_rate - 70.0) / 50.0,
            confidence: 0.8,
        };
        
        self.pattern_database.push(pattern);
        Ok(())
    }
    
    /// Convert scenario to feature vector
    fn scenario_to_vector(&self, scenario: &ScenarioContext) -> Vec<f32> {
        vec![
            scenario.time_pressure,
            scenario.risk_level,
            scenario.goal_urgency,
            scenario.environmental_factors.len() as f32,
        ]
    }
    
    /// Convert action to response vector
    fn action_to_vector(&self, action: &DrivingDecision) -> Vec<f32> {
        match action {
            DrivingDecision::Accelerate(val) => vec![1.0, *val, 0.0, 0.0],
            DrivingDecision::Brake(val) => vec![0.0, *val, 1.0, 0.0],
            DrivingDecision::Steer(val) => vec![0.0, 0.0, 0.0, *val],
            DrivingDecision::Stop => vec![0.0, 1.0, 1.0, 0.0],
            DrivingDecision::EmergencyBrake => vec![0.0, 1.0, 1.0, 1.0],
            DrivingDecision::Yield => vec![0.0, 0.5, 0.8, 0.0],
            DrivingDecision::Proceed => vec![0.5, 0.0, 0.0, 0.0],
            DrivingDecision::ChangeLane(dir) => vec![0.0, 0.0, 0.0, *dir as f32],
        }
    }
    
    /// Calculate similarity between two vectors using cosine similarity
    fn calculate_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }
        
        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }
} 