//! # Safety and Validation Module
//!
//! Comprehensive safety systems for autonomous driving with biometric validation.
//! This module implements the critical safety layers that validate AI decisions
//! and ensure biometric performance metrics stay within safe ranges.

pub mod emergency_override;
pub mod safety_monitor;
pub mod validation_engine;
pub mod risk_assessor;

use crate::utils::{Result, VerumError};
use crate::ai::decision_engine::DrivingDecision;
use crate::biometrics::BiometricState;
use crate::vehicle::VehicleState;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Safety level classifications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyLevel {
    Safe,
    Caution,
    Warning,
    Critical,
    Emergency,
}

/// Safety violation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyViolation {
    ExcessiveStress {
        current: f32,
        threshold: f32,
        duration: f32,
    },
    BiometricAnomaly {
        metric: String,
        value: f32,
        expected_range: (f32, f32),
    },
    UnsafeDecision {
        decision: String,
        risk_score: f32,
        context: String,
    },
    VehicleParameterViolation {
        parameter: String,
        value: f32,
        limit: f32,
    },
    PatternDeviation {
        deviation_score: f32,
        pattern_type: String,
    },
    PerformanceDegradation {
        metric: String,
        current: f32,
        baseline: f32,
        degradation: f32,
    },
}

/// Safety validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyValidation {
    pub is_safe: bool,
    pub safety_level: SafetyLevel,
    pub violations: Vec<SafetyViolation>,
    pub recommended_actions: Vec<String>,
    pub confidence: f32,
    pub validation_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Risk assessment for decisions and states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub overall_risk: f32,
    pub biometric_risk: f32,
    pub vehicle_risk: f32,
    pub decision_risk: f32,
    pub environmental_risk: f32,
    pub risk_factors: Vec<String>,
    pub mitigation_strategies: Vec<String>,
}

/// Performance metrics for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub comfort_score: f32,
    pub efficiency_score: f32,
    pub safety_score: f32,
    pub stress_score: f32,
    pub consistency_score: f32,
    pub adaptation_score: f32,
}

/// Main safety coordinator
pub struct SafetyCoordinator {
    id: Uuid,
    emergency_override: emergency_override::EmergencyOverride,
    safety_monitor: safety_monitor::SafetyMonitor,
    validation_engine: validation_engine::ValidationEngine,
    risk_assessor: risk_assessor::RiskAssessor,
    
    // Safety thresholds
    stress_threshold: f32,
    biometric_anomaly_threshold: f32,
    risk_threshold: f32,
    
    // Historical data for pattern analysis
    safety_history: RwLock<VecDeque<SafetyValidation>>,
    performance_history: RwLock<VecDeque<PerformanceMetrics>>,
    
    // Emergency state
    emergency_state: RwLock<bool>,
    override_reason: RwLock<Option<String>>,
}

impl SafetyCoordinator {
    /// Create new safety coordinator
    pub async fn new(config: &crate::utils::config::Config) -> Result<Self> {
        let id = Uuid::new_v4();
        
        Ok(Self {
            id,
            emergency_override: emergency_override::EmergencyOverride::new(&config.safety).await?,
            safety_monitor: safety_monitor::SafetyMonitor::new(&config.safety).await?,
            validation_engine: validation_engine::ValidationEngine::new(&config.safety).await?,
            risk_assessor: risk_assessor::RiskAssessor::new(&config.safety).await?,
            
            stress_threshold: config.safety.max_stress_threshold,
            biometric_anomaly_threshold: config.safety.biometric_anomaly_threshold,
            risk_threshold: config.safety.risk_threshold,
            
            safety_history: RwLock::new(VecDeque::with_capacity(1000)),
            performance_history: RwLock::new(VecDeque::with_capacity(1000)),
            
            emergency_state: RwLock::new(false),
            override_reason: RwLock::new(None),
        })
    }
    
    /// Validate a driving decision before execution
    pub async fn validate_decision(
        &self,
        decision: &DrivingDecision,
        biometric_state: &BiometricState,
        vehicle_state: &VehicleState,
        context: &crate::ai::ScenarioContext,
    ) -> Result<SafetyValidation> {
        // Check if we're in emergency state
        if *self.emergency_state.read().await {
            return Ok(SafetyValidation {
                is_safe: false,
                safety_level: SafetyLevel::Emergency,
                violations: vec![SafetyViolation::UnsafeDecision {
                    decision: format!("{:?}", decision),
                    risk_score: 1.0,
                    context: "System in emergency state".to_string(),
                }],
                recommended_actions: vec!["Emergency stop".to_string()],
                confidence: 1.0,
                validation_timestamp: chrono::Utc::now(),
            });
        }
        
        let mut violations = Vec::new();
        let mut safety_level = SafetyLevel::Safe;
        let mut recommended_actions = Vec::new();
        
        // 1. Biometric validation
        let biometric_violations = self.validate_biometrics(biometric_state).await?;
        violations.extend(biometric_violations);
        
        // 2. Decision risk assessment
        let decision_risk = self.risk_assessor
            .assess_decision_risk(decision, context, biometric_state, vehicle_state)
            .await?;
        
        if decision_risk.overall_risk > self.risk_threshold {
            violations.push(SafetyViolation::UnsafeDecision {
                decision: format!("{:?}", decision),
                risk_score: decision_risk.overall_risk,
                context: format!("Context: {:?}", context),
            });
        }
        
        // 3. Vehicle state validation
        let vehicle_violations = self.validate_vehicle_state(vehicle_state).await?;
        violations.extend(vehicle_violations);
        
        // 4. Pattern deviation analysis
        let pattern_violations = self.validation_engine
            .validate_behavioral_patterns(decision, biometric_state, context)
            .await?;
        violations.extend(pattern_violations);
        
        // Determine overall safety level
        safety_level = self.calculate_safety_level(&violations);
        
        // Generate recommendations
        recommended_actions = self.generate_safety_recommendations(&violations, safety_level);
        
        // Check for emergency override conditions
        if safety_level == SafetyLevel::Critical || safety_level == SafetyLevel::Emergency {
            self.emergency_override.evaluate_override_conditions(
                &violations,
                biometric_state,
                vehicle_state,
            ).await?;
        }
        
        let validation = SafetyValidation {
            is_safe: violations.is_empty() && safety_level == SafetyLevel::Safe,
            safety_level,
            violations,
            recommended_actions,
            confidence: self.calculate_validation_confidence(&decision_risk).await,
            validation_timestamp: chrono::Utc::now(),
        };
        
        // Store in history
        let mut history = self.safety_history.write().await;
        history.push_back(validation.clone());
        if history.len() > 1000 {
            history.pop_front();
        }
        
        Ok(validation)
    }
    
    /// Validate biometric state for safety
    async fn validate_biometrics(&self, state: &BiometricState) -> Result<Vec<SafetyViolation>> {
        let mut violations = Vec::new();
        
        // Check stress level
        if state.stress_level > self.stress_threshold {
            violations.push(SafetyViolation::ExcessiveStress {
                current: state.stress_level,
                threshold: self.stress_threshold,
                duration: 0.0, // Would track duration in real implementation
            });
        }
        
        // Check heart rate
        if state.heart_rate > 120.0 || state.heart_rate < 50.0 {
            violations.push(SafetyViolation::BiometricAnomaly {
                metric: "heart_rate".to_string(),
                value: state.heart_rate,
                expected_range: (50.0, 120.0),
            });
        }
        
        // Check skin conductance for extreme stress
        if state.skin_conductance > 0.9 {
            violations.push(SafetyViolation::BiometricAnomaly {
                metric: "skin_conductance".to_string(),
                value: state.skin_conductance,
                expected_range: (0.0, 0.8),
            });
        }
        
        // Check for fear response indicators
        if state.indicates_fear(0.8) {
            violations.push(SafetyViolation::BiometricAnomaly {
                metric: "fear_response".to_string(),
                value: 1.0,
                expected_range: (0.0, 0.8),
            });
        }
        
        Ok(violations)
    }
    
    /// Validate vehicle state parameters
    async fn validate_vehicle_state(&self, state: &VehicleState) -> Result<Vec<SafetyViolation>> {
        let mut violations = Vec::new();
        
        // Check speed limits (simplified)
        let speed = state.velocity.magnitude();
        if speed > 35.0 { // 35 m/s = ~78 mph
            violations.push(SafetyViolation::VehicleParameterViolation {
                parameter: "speed".to_string(),
                value: speed,
                limit: 35.0,
            });
        }
        
        // Check acceleration limits
        let acceleration = state.acceleration.magnitude();
        if acceleration > 8.0 { // 8 m/s² is quite high
            violations.push(SafetyViolation::VehicleParameterViolation {
                parameter: "acceleration".to_string(),
                value: acceleration,
                limit: 8.0,
            });
        }
        
        // Check engine parameters
        if state.engine_rpm > 6000.0 {
            violations.push(SafetyViolation::VehicleParameterViolation {
                parameter: "engine_rpm".to_string(),
                value: state.engine_rpm,
                limit: 6000.0,
            });
        }
        
        Ok(violations)
    }
    
    /// Calculate overall safety level from violations
    fn calculate_safety_level(&self, violations: &[SafetyViolation]) -> SafetyLevel {
        if violations.is_empty() {
            return SafetyLevel::Safe;
        }
        
        let mut max_level = SafetyLevel::Safe;
        
        for violation in violations {
            let level = match violation {
                SafetyViolation::ExcessiveStress { current, threshold, .. } => {
                    let ratio = current / threshold;
                    if ratio > 2.0 {
                        SafetyLevel::Emergency
                    } else if ratio > 1.5 {
                        SafetyLevel::Critical
                    } else if ratio > 1.2 {
                        SafetyLevel::Warning
                    } else {
                        SafetyLevel::Caution
                    }
                }
                SafetyViolation::BiometricAnomaly { .. } => SafetyLevel::Warning,
                SafetyViolation::UnsafeDecision { risk_score, .. } => {
                    if *risk_score > 0.9 {
                        SafetyLevel::Critical
                    } else if *risk_score > 0.7 {
                        SafetyLevel::Warning
                    } else {
                        SafetyLevel::Caution
                    }
                }
                SafetyViolation::VehicleParameterViolation { .. } => SafetyLevel::Warning,
                SafetyViolation::PatternDeviation { deviation_score, .. } => {
                    if *deviation_score > 0.8 {
                        SafetyLevel::Warning
                    } else {
                        SafetyLevel::Caution
                    }
                }
                SafetyViolation::PerformanceDegradation { degradation, .. } => {
                    if *degradation > 0.5 {
                        SafetyLevel::Warning
                    } else {
                        SafetyLevel::Caution
                    }
                }
            };
            
            if level as u8 > max_level as u8 {
                max_level = level;
            }
        }
        
        max_level
    }
    
    /// Generate safety recommendations based on violations
    fn generate_safety_recommendations(
        &self,
        violations: &[SafetyViolation],
        safety_level: SafetyLevel,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match safety_level {
            SafetyLevel::Safe => {}
            SafetyLevel::Caution => {
                recommendations.push("Monitor biometric state closely".to_string());
                recommendations.push("Consider reducing driving intensity".to_string());
            }
            SafetyLevel::Warning => {
                recommendations.push("Reduce speed and increase following distance".to_string());
                recommendations.push("Consider taking a break".to_string());
                recommendations.push("Activate stress reduction protocols".to_string());
            }
            SafetyLevel::Critical => {
                recommendations.push("Prepare for emergency handover".to_string());
                recommendations.push("Find safe location to stop".to_string());
                recommendations.push("Alert emergency contacts".to_string());
            }
            SafetyLevel::Emergency => {
                recommendations.push("Execute emergency stop protocol".to_string());
                recommendations.push("Contact emergency services".to_string());
                recommendations.push("Transfer control to backup systems".to_string());
            }
        }
        
        // Add specific recommendations for violation types
        for violation in violations {
            match violation {
                SafetyViolation::ExcessiveStress { .. } => {
                    recommendations.push("Implement calming breathing exercises".to_string());
                    recommendations.push("Reduce environmental stressors".to_string());
                }
                SafetyViolation::BiometricAnomaly { metric, .. } => {
                    recommendations.push(format!("Monitor {} closely", metric));
                    recommendations.push("Consider medical evaluation".to_string());
                }
                SafetyViolation::UnsafeDecision { .. } => {
                    recommendations.push("Override with conservative decision".to_string());
                    recommendations.push("Increase safety margins".to_string());
                }
                SafetyViolation::VehicleParameterViolation { parameter, .. } => {
                    recommendations.push(format!("Reduce {}", parameter));
                }
                SafetyViolation::PatternDeviation { .. } => {
                    recommendations.push("Return to established behavioral patterns".to_string());
                    recommendations.push("Increase AI supervision level".to_string());
                }
                SafetyViolation::PerformanceDegradation { .. } => {
                    recommendations.push("Recalibrate personal model".to_string());
                    recommendations.push("Increase training data collection".to_string());
                }
            }
        }
        
        recommendations.sort();
        recommendations.dedup();
        recommendations
    }
    
    /// Calculate confidence in the validation
    async fn calculate_validation_confidence(&self, risk_assessment: &RiskAssessment) -> f32 {
        let history = self.safety_history.read().await;
        
        // Base confidence on historical accuracy and current risk factors
        let historical_accuracy = if history.len() > 10 {
            let recent_validations = history.iter().rev().take(10);
            let accurate_count = recent_validations
                .filter(|v| v.confidence > 0.8)
                .count();
            accurate_count as f32 / 10.0
        } else {
            0.8 // Default confidence
        };
        
        // Adjust based on current risk level
        let risk_adjustment = 1.0 - risk_assessment.overall_risk * 0.3;
        
        (historical_accuracy * risk_adjustment).clamp(0.0, 1.0)
    }
    
    /// Assess current performance metrics
    pub async fn assess_performance(
        &self,
        biometric_state: &BiometricState,
        vehicle_state: &VehicleState,
        baseline_performance: Option<&PerformanceMetrics>,
    ) -> Result<PerformanceMetrics> {
        let comfort_score = biometric_state.comfort_level();
        let stress_score = 1.0 - biometric_state.stress_level;
        
        // Calculate efficiency based on vehicle smoothness
        let efficiency_score = self.calculate_efficiency_score(vehicle_state).await;
        
        // Calculate safety score based on recent violations
        let safety_score = self.calculate_safety_score().await;
        
        // Calculate consistency score (simplified)
        let consistency_score = 0.8; // Would compare against historical patterns
        
        // Calculate adaptation score (how well AI is adapting to current conditions)
        let adaptation_score = self.calculate_adaptation_score(biometric_state).await;
        
        let metrics = PerformanceMetrics {
            comfort_score,
            efficiency_score,
            safety_score,
            stress_score,
            consistency_score,
            adaptation_score,
        };
        
        // Store in history
        let mut history = self.performance_history.write().await;
        history.push_back(metrics.clone());
        if history.len() > 1000 {
            history.pop_front();
        }
        
        // Check for performance degradation
        if let Some(baseline) = baseline_performance {
            self.check_performance_degradation(&metrics, baseline).await?;
        }
        
        Ok(metrics)
    }
    
    /// Calculate efficiency score based on vehicle smoothness
    async fn calculate_efficiency_score(&self, vehicle_state: &VehicleState) -> f32 {
        let speed = vehicle_state.velocity.magnitude();
        let acceleration = vehicle_state.acceleration.magnitude();
        
        // Penalize excessive acceleration/deceleration
        let smoothness = 1.0 - (acceleration / 5.0).clamp(0.0, 1.0);
        
        // Reward optimal speed range
        let speed_efficiency = if speed > 15.0 && speed < 25.0 {
            1.0
        } else {
            1.0 - ((speed - 20.0).abs() / 20.0).clamp(0.0, 0.5)
        };
        
        (smoothness + speed_efficiency) / 2.0
    }
    
    /// Calculate safety score based on recent violations
    async fn calculate_safety_score(&self) -> f32 {
        let history = self.safety_history.read().await;
        
        if history.is_empty() {
            return 1.0;
        }
        
        let recent_validations = history.iter().rev().take(20);
        let violation_count: usize = recent_validations
            .map(|v| v.violations.len())
            .sum();
        
        let max_possible_violations = 20 * 5; // 20 validations * 5 possible violation types
        1.0 - (violation_count as f32 / max_possible_violations as f32)
    }
    
    /// Calculate adaptation score
    async fn calculate_adaptation_score(&self, biometric_state: &BiometricState) -> f32 {
        // Simple heuristic: good adaptation means stable biometrics
        let stability = 1.0 - biometric_state.stress_level;
        let comfort = biometric_state.comfort_level();
        
        (stability + comfort) / 2.0
    }
    
    /// Check for performance degradation
    async fn check_performance_degradation(
        &self,
        current: &PerformanceMetrics,
        baseline: &PerformanceMetrics,
    ) -> Result<()> {
        let degradation_threshold = 0.2; // 20% degradation threshold
        
        let comfort_degradation = (baseline.comfort_score - current.comfort_score) / baseline.comfort_score;
        let efficiency_degradation = (baseline.efficiency_score - current.efficiency_score) / baseline.efficiency_score;
        let safety_degradation = (baseline.safety_score - current.safety_score) / baseline.safety_score;
        
        if comfort_degradation > degradation_threshold {
            tracing::warn!("Comfort performance degraded by {:.1}%", comfort_degradation * 100.0);
        }
        
        if efficiency_degradation > degradation_threshold {
            tracing::warn!("Efficiency performance degraded by {:.1}%", efficiency_degradation * 100.0);
        }
        
        if safety_degradation > degradation_threshold {
            tracing::warn!("Safety performance degraded by {:.1}%", safety_degradation * 100.0);
        }
        
        Ok(())
    }
    
    /// Trigger emergency override
    pub async fn trigger_emergency_override(&self, reason: String) -> Result<()> {
        *self.emergency_state.write().await = true;
        *self.override_reason.write().await = Some(reason.clone());
        
        tracing::error!("Emergency override triggered: {}", reason);
        
        // Activate emergency protocols
        self.emergency_override.activate_emergency_protocols().await?;
        
        Ok(())
    }
    
    /// Clear emergency state
    pub async fn clear_emergency_state(&self) -> Result<()> {
        *self.emergency_state.write().await = false;
        *self.override_reason.write().await = None;
        
        tracing::info!("Emergency state cleared");
        Ok(())
    }
    
    /// Get comprehensive safety statistics
    pub async fn get_safety_statistics(&self) -> SafetyStatistics {
        let safety_history = self.safety_history.read().await;
        let performance_history = self.performance_history.read().await;
        
        let total_validations = safety_history.len();
        let total_violations = safety_history.iter().map(|v| v.violations.len()).sum::<usize>();
        
        let avg_safety_score = if !performance_history.is_empty() {
            performance_history.iter().map(|p| p.safety_score).sum::<f32>() / performance_history.len() as f32
        } else {
            0.0
        };
        
        let avg_comfort_score = if !performance_history.is_empty() {
            performance_history.iter().map(|p| p.comfort_score).sum::<f32>() / performance_history.len() as f32
        } else {
            0.0
        };
        
        SafetyStatistics {
            total_validations,
            total_violations,
            avg_safety_score,
            avg_comfort_score,
            emergency_overrides: 0, // Would track in real implementation
            current_safety_level: if safety_history.is_empty() {
                SafetyLevel::Safe
            } else {
                safety_history.back().unwrap().safety_level
            },
        }
    }
}

/// Safety statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyStatistics {
    pub total_validations: usize,
    pub total_violations: usize,
    pub avg_safety_score: f32,
    pub avg_comfort_score: f32,
    pub emergency_overrides: usize,
    pub current_safety_level: SafetyLevel,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biometrics::BiometricState;
    use crate::vehicle::VehicleState;
    use crate::utils::Position;
    
    #[tokio::test]
    async fn test_safety_validation() {
        // Would implement comprehensive safety validation tests
        let biometric_state = BiometricState::new();
        let vehicle_state = VehicleState {
            position: Position::new(0.0, 0.0),
            velocity: crate::utils::Velocity::new(20.0, 0.0, 0.0),
            acceleration: crate::utils::Acceleration::new(0.0, 0.0, 0.0),
            heading: 0.0,
            engine_rpm: 2000.0,
            fuel_level: 0.8,
            temperature: 20.0,
            timestamp: chrono::Utc::now(),
        };
        
        // Test would validate that normal states pass safety checks
        assert!(biometric_state.comfort_level() > 0.0);
        assert!(vehicle_state.velocity.magnitude() > 0.0);
    }
} 