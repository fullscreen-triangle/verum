//! Decision Engine
//!
//! Makes real-time driving decisions based on personal AI model and current context

use crate::utils::{Result, VerumError};
use super::{PersonalAIModel, ScenarioContext, BiometricState, DecisionResult, BehavioralPattern};
use super::fear_response::FearResponse;
use serde::{Deserialize, Serialize};

/// Driving decision types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DrivingDecision {
    Accelerate(f32),      // acceleration value
    Brake(f32),           // braking force
    Steer(f32),           // steering angle
    ChangeLane(i8),       // lane change direction (-1, 0, 1)
    Stop,
    EmergencyBrake,
    Yield,
    Proceed,
}

/// Decision engine that makes contextual driving decisions
pub struct DecisionEngine {
    model: PersonalAIModel,
    decision_history: Vec<(ScenarioContext, DrivingDecision, f32)>, // scenario, decision, confidence
}

impl DecisionEngine {
    /// Create new decision engine
    pub async fn new(model: PersonalAIModel) -> Result<Self> {
        Ok(Self {
            model,
            decision_history: Vec::new(),
        })
    }
    
    /// Make contextual decision based on all available information
    pub async fn make_contextual_decision(
        &mut self,
        scenario: ScenarioContext,
        biometrics: BiometricState,
        similar_patterns: Vec<BehavioralPattern>,
        fear_level: f32,
    ) -> Result<DecisionResult> {
        // Simple decision logic based on risk level and fear
        let base_confidence = 0.8;
        let confidence = base_confidence * (1.0 - fear_level * 0.5);
        
        let decision = if scenario.risk_level > 0.8 || fear_level > 0.7 {
            DrivingDecision::Brake(0.6)
        } else if scenario.time_pressure > 0.8 {
            DrivingDecision::Accelerate(0.4)
        } else {
            DrivingDecision::Proceed
        };
        
        let reasoning = format!(
            "Decision based on risk_level: {:.2}, fear_level: {:.2}, time_pressure: {:.2}",
            scenario.risk_level, fear_level, scenario.time_pressure
        );
        
        let result = DecisionResult {
            decision: decision.clone(),
            confidence,
            reasoning,
            alternative_actions: vec![DrivingDecision::Stop, DrivingDecision::Yield],
            expected_stress_level: fear_level,
        };
        
        // Record this decision
        self.decision_history.push((scenario, decision, confidence));
        
        Ok(result)
    }
    
    /// Get decision statistics
    pub fn get_decision_count(&self) -> usize {
        self.decision_history.len()
    }
} 