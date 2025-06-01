//! Biometric Data Processing

use crate::utils::Result;
use super::BiometricState;

/// Processed biometric data
#[derive(Debug, Clone)]
pub struct ProcessedData {
    pub filtered_state: BiometricState,
    pub confidence: f32,
}

/// Process raw biometric data
pub fn process_biometric_data(raw_state: BiometricState) -> Result<ProcessedData> {
    Ok(ProcessedData {
        filtered_state: raw_state,
        confidence: 0.9,
    })
} 