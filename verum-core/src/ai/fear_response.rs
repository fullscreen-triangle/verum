//! Fear Response Learning System
//!
//! Learns and implements fear responses from cross-domain experiences

use crate::utils::{Result, VerumError};
use super::{PersonalAIModel, AIConfig, ScenarioContext, BiometricState};
use serde::{Deserialize, Serialize};

/// Fear response data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FearResponse {
    pub trigger_pattern: Vec<f32>,
    pub intensity: f32,
    pub learned_from_domain: super::LearningDomain,
    pub confidence: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Fear learning system that learns from biometric responses
pub struct FearLearningSystem {
    learned_responses: Vec<FearResponse>,
    sensitivity_threshold: f32,
}

impl FearLearningSystem {
    /// Create new fear learning system
    pub async fn new(model: &PersonalAIModel, config: &AIConfig) -> Result<Self> {
        Ok(Self {
            learned_responses: Vec::new(),
            sensitivity_threshold: config.fear_response_sensitivity,
        })
    }
    
    /// Evaluate fear response for a scenario
    pub async fn evaluate_fear_response(
        &self,
        scenario: &ScenarioContext,
        biometrics: &BiometricState,
    ) -> Result<f32> {
        // Placeholder fear evaluation logic
        let stress_level = (biometrics.heart_rate - 70.0) / 50.0; // Normalize heart rate
        Ok(stress_level.clamp(0.0, 1.0))
    }
    
    /// Update fear response based on experience outcome
    pub async fn update_fear_response(
        &mut self,
        scenario: ScenarioContext,
        biometrics: BiometricState,
        success: bool,
    ) -> Result<()> {
        // Learn from the experience
        if !success && biometrics.heart_rate > 100.0 {
            let fear_response = FearResponse {
                trigger_pattern: vec![scenario.risk_level, scenario.time_pressure],
                intensity: (biometrics.heart_rate - 70.0) / 50.0,
                learned_from_domain: super::LearningDomain::Driving,
                confidence: 0.8,
                timestamp: chrono::Utc::now(),
            };
            self.learned_responses.push(fear_response);
        }
        Ok(())
    }
} 