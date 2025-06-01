//! Biometric Data Validation

use crate::utils::{Result, config::BiometricsConfig};
use super::BiometricState;

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub reason: String,
    pub confidence: f32,
}

/// Biometric data validator
#[derive(Clone)]
pub struct BiometricValidator {
    config: BiometricsConfig,
}

impl BiometricValidator {
    pub fn new(config: &BiometricsConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
    
    pub async fn validate_state(&self, state: &BiometricState) -> Result<ValidationResult> {
        // Simple validation logic
        let is_valid = state.heart_rate >= self.config.heart_rate_range.0 
            && state.heart_rate <= self.config.heart_rate_range.1;
        
        Ok(ValidationResult {
            is_valid,
            reason: if is_valid { "Valid".to_string() } else { "Heart rate out of range".to_string() },
            confidence: 0.9,
        })
    }
} 