//! # AI Module
//!
//! Personal Intelligence-Driven Navigation AI System
//!
//! This module implements the core AI functionality for Verum, including:
//! - Personal AI models learned from 5+ years of cross-domain data
//! - Fear response learning and implementation
//! - Real-time decision making for autonomous driving
//! - Cross-domain pattern transfer from other activities

pub mod personal_model;
pub mod fear_response;
pub mod decision_engine;
pub mod pattern_matcher;

// Re-exports
pub use personal_model::PersonalAIModel;
pub use fear_response::{FearResponse, FearLearningSystem};
pub use decision_engine::{DecisionEngine, DrivingDecision};
pub use pattern_matcher::{PatternMatcher, CrossDomainPattern};

use crate::utils::{Result, VerumError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Represents different domains from which patterns can be learned
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LearningDomain {
    Driving,
    Walking,
    Cycling,
    Tennis,
    Basketball,
    Gaming,
    DailyNavigation,
}

/// Cross-domain behavioral pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralPattern {
    pub id: Uuid,
    pub domain: LearningDomain,
    pub scenario_type: String,
    pub stimulus_vector: Vec<f32>,
    pub response_vector: Vec<f32>,
    pub success_rate: f32,
    pub stress_level: f32,
    pub confidence: f32,
}

/// Biometric state during a behavioral response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricState {
    pub heart_rate: f32,
    pub skin_conductance: f32,
    pub muscle_tension: Vec<f32>,
    pub breathing_rate: f32,
    pub eye_tracking: Option<EyeTrackingData>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Eye tracking data for attention analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeTrackingData {
    pub gaze_x: f32,
    pub gaze_y: f32,
    pub pupil_diameter: f32,
    pub blink_rate: f32,
    pub saccade_velocity: f32,
}

/// Scenario context for decision making
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioContext {
    pub scenario_type: String,
    pub environmental_factors: HashMap<String, f32>,
    pub time_pressure: f32,
    pub risk_level: f32,
    pub goal_urgency: f32,
}

/// AI model capabilities and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub domains_trained: Vec<LearningDomain>,
    pub training_duration_months: u32,
    pub pattern_count: usize,
    pub stress_tolerance_range: (f32, f32),
    pub confidence_threshold: f32,
    pub supported_scenarios: Vec<String>,
}

/// Result of a driving decision with confidence metrics
#[derive(Debug, Clone)]
pub struct DecisionResult {
    pub decision: DrivingDecision,
    pub confidence: f32,
    pub reasoning: String,
    pub alternative_actions: Vec<DrivingDecision>,
    pub expected_stress_level: f32,
}

/// AI system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIConfig {
    pub model_path: String,
    pub pattern_similarity_threshold: f32,
    pub fear_response_sensitivity: f32,
    pub decision_timeout_ms: u64,
    pub max_stress_threshold: f32,
    pub learning_rate: f32,
}

impl Default for AIConfig {
    fn default() -> Self {
        Self {
            model_path: "/var/lib/verum/models".to_string(),
            pattern_similarity_threshold: 0.7,
            fear_response_sensitivity: 0.8,
            decision_timeout_ms: 100,
            max_stress_threshold: 0.85,
            learning_rate: 0.001,
        }
    }
}

/// Main AI orchestrator that coordinates all AI subsystems
pub struct AIOrchestrator {
    personal_model: PersonalAIModel,
    fear_system: FearLearningSystem,
    decision_engine: DecisionEngine,
    pattern_matcher: PatternMatcher,
    config: AIConfig,
}

impl AIOrchestrator {
    /// Create new AI orchestrator
    pub async fn new(
        model: PersonalAIModel,
        config: AIConfig,
    ) -> Result<Self> {
        let fear_system = FearLearningSystem::new(&model, &config).await?;
        let pattern_matcher = PatternMatcher::new(&model).await?;
        let decision_engine = DecisionEngine::new(model.clone()).await?;
        
        Ok(Self {
            personal_model: model,
            fear_system,
            decision_engine,
            pattern_matcher,
            config,
        })
    }
    
    /// Process a driving scenario and make a decision
    pub async fn process_scenario(
        &mut self,
        scenario: ScenarioContext,
        current_biometrics: BiometricState,
    ) -> Result<DecisionResult> {
        // Find similar patterns from cross-domain learning
        let similar_patterns = self.pattern_matcher
            .find_similar_patterns(&scenario, 0.7)
            .await?;
        
        // Check fear response
        let fear_level = self.fear_system
            .evaluate_fear_response(&scenario, &current_biometrics)
            .await?;
        
        // Make decision considering all factors
        let decision = self.decision_engine
            .make_contextual_decision(
                scenario,
                current_biometrics,
                similar_patterns,
                fear_level,
            )
            .await?;
        
        Ok(decision)
    }
    
    /// Learn from a new experience
    pub async fn learn_from_experience(
        &mut self,
        scenario: ScenarioContext,
        biometric_state: BiometricState,
        action_taken: DrivingDecision,
        outcome_success: bool,
    ) -> Result<()> {
        // Update fear learning system
        self.fear_system
            .update_fear_response(scenario.clone(), biometric_state.clone(), outcome_success)
            .await?;
        
        // Update pattern matcher
        self.pattern_matcher
            .add_pattern(scenario, biometric_state, action_taken, outcome_success)
            .await?;
        
        // Update decision engine
        self.decision_engine
            .update_from_outcome(action_taken, outcome_success)
            .await?;
        
        Ok(())
    }
    
    /// Get current AI system statistics
    pub fn get_statistics(&self) -> AIStatistics {
        AIStatistics {
            patterns_learned: self.pattern_matcher.pattern_count(),
            fear_responses_trained: self.fear_system.response_count(),
            decisions_made: self.decision_engine.decision_count(),
            average_confidence: self.decision_engine.average_confidence(),
            stress_tolerance_range: self.personal_model.stress_tolerance_range(),
        }
    }
}

/// AI system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIStatistics {
    pub patterns_learned: usize,
    pub fear_responses_trained: usize,
    pub decisions_made: u64,
    pub average_confidence: f32,
    pub stress_tolerance_range: (f32, f32),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_learning_domain_serialization() {
        let domain = LearningDomain::Tennis;
        let serialized = serde_json::to_string(&domain).unwrap();
        let deserialized: LearningDomain = serde_json::from_str(&serialized).unwrap();
        assert_eq!(domain, deserialized);
    }
    
    #[test]
    fn test_ai_config_default() {
        let config = AIConfig::default();
        assert!(config.pattern_similarity_threshold > 0.0);
        assert!(config.fear_response_sensitivity > 0.0);
        assert!(config.decision_timeout_ms > 0);
    }
} 