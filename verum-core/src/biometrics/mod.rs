//! # Biometrics Module
//!
//! Real-time biometric processing and validation for personal AI systems

pub mod sensors;
pub mod processing;
pub mod validator;
pub mod patterns;

// Re-exports
pub use sensors::{BiometricSensors, SensorReading};
pub use processing::{BiometricProcessor, ProcessedData};
pub use validator::{BiometricValidator, ValidationResult};
pub use patterns::{BiometricPattern, PatternMatcher as BiometricPatternMatcher};

use crate::utils::{Result, VerumError};
use crate::utils::config::BiometricsConfig;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Current biometric state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricState {
    pub heart_rate: f32,
    pub skin_conductance: f32,
    pub muscle_tension: Vec<f32>,
    pub breathing_rate: f32,
    pub body_temperature: f32,
    pub blood_pressure: Option<(f32, f32)>, // systolic, diastolic
    pub eye_tracking: Option<EyeTrackingData>,
    pub stress_level: f32, // 0.0 to 1.0
    pub arousal_level: f32, // 0.0 to 1.0
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl BiometricState {
    pub fn new() -> Self {
        Self {
            heart_rate: 70.0,
            skin_conductance: 0.5,
            muscle_tension: vec![0.0; 16], // 16 muscle groups
            breathing_rate: 15.0,
            body_temperature: 36.5,
            blood_pressure: None,
            eye_tracking: None,
            stress_level: 0.3,
            arousal_level: 0.4,
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Calculate overall comfort level (0.0 = very uncomfortable, 1.0 = very comfortable)
    pub fn comfort_level(&self) -> f32 {
        // Normalize metrics and combine them
        let hr_comfort = self.heart_rate_comfort();
        let stress_comfort = 1.0 - self.stress_level;
        let arousal_comfort = 1.0 - (self.arousal_level - 0.5).abs() * 2.0; // optimal around 0.5
        
        (hr_comfort + stress_comfort + arousal_comfort) / 3.0
    }
    
    fn heart_rate_comfort(&self) -> f32 {
        // Optimal heart rate around 60-80 bpm for resting
        let optimal_range = (60.0, 80.0);
        if self.heart_rate >= optimal_range.0 && self.heart_rate <= optimal_range.1 {
            1.0
        } else {
            let distance = if self.heart_rate < optimal_range.0 {
                optimal_range.0 - self.heart_rate
            } else {
                self.heart_rate - optimal_range.1
            };
            (1.0 - distance / 50.0).max(0.0) // Comfort drops with distance from optimal
        }
    }
    
    /// Check if biometric state indicates fear response
    pub fn indicates_fear(&self, threshold: f32) -> bool {
        self.stress_level > threshold || 
        self.heart_rate > 100.0 ||
        self.skin_conductance > 0.8
    }
    
    /// Calculate similarity to another biometric state
    pub fn similarity(&self, other: &BiometricState) -> f32 {
        let hr_sim = 1.0 - (self.heart_rate - other.heart_rate).abs() / 100.0;
        let stress_sim = 1.0 - (self.stress_level - other.stress_level).abs();
        let arousal_sim = 1.0 - (self.arousal_level - other.arousal_level).abs();
        let sc_sim = 1.0 - (self.skin_conductance - other.skin_conductance).abs();
        
        (hr_sim + stress_sim + arousal_sim + sc_sim) / 4.0
    }
}

/// Eye tracking data for attention analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EyeTrackingData {
    pub gaze_x: f32,
    pub gaze_y: f32,
    pub pupil_diameter: f32,
    pub blink_rate: f32,
    pub saccade_velocity: f32,
    pub fixation_duration: f32,
    pub attention_focus: AttentionFocus,
}

/// Attention focus areas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionFocus {
    Road,
    Dashboard,
    Mirrors,
    Passenger,
    Environment,
    Unknown,
}

/// Main biometric processor
pub struct BiometricProcessor {
    id: Uuid,
    config: BiometricsConfig,
    sensors: BiometricSensors,
    validator: BiometricValidator,
    pattern_matcher: BiometricPatternMatcher,
    state_buffer: RwLock<VecDeque<BiometricState>>,
    baseline_state: RwLock<Option<BiometricState>>,
    is_running: RwLock<bool>,
}

impl BiometricProcessor {
    /// Create new biometric processor
    pub async fn new(config: BiometricsConfig) -> Result<Self> {
        let id = Uuid::new_v4();
        let sensors = BiometricSensors::new(&config).await?;
        let validator = BiometricValidator::new(&config);
        let pattern_matcher = BiometricPatternMatcher::new();
        
        Ok(Self {
            id,
            config,
            sensors,
            validator,
            pattern_matcher,
            state_buffer: RwLock::new(VecDeque::with_capacity(config.buffer_size)),
            baseline_state: RwLock::new(None),
            is_running: RwLock::new(false),
        })
    }
    
    /// Start biometric processing
    pub async fn start(&self) -> Result<()> {
        tracing::info!("Starting biometric processor {}", self.id);
        
        // Start sensors
        self.sensors.start().await?;
        
        // Calibrate baseline if needed
        if self.baseline_state.read().await.is_none() {
            self.calibrate_baseline().await?;
        }
        
        // Start processing loop
        *self.is_running.write().await = true;
        
        let processor = self.clone_for_processing();
        tokio::spawn(async move {
            processor.processing_loop().await;
        });
        
        tracing::info!("Biometric processor started");
        Ok(())
    }
    
    /// Stop biometric processing
    pub async fn stop(&self) -> Result<()> {
        tracing::info!("Stopping biometric processor {}", self.id);
        
        *self.is_running.write().await = false;
        self.sensors.stop().await?;
        
        tracing::info!("Biometric processor stopped");
        Ok(())
    }
    
    /// Get current biometric state
    pub async fn get_current_state(&self) -> Result<BiometricState> {
        let buffer = self.state_buffer.read().await;
        buffer.back()
            .cloned()
            .ok_or_else(|| VerumError::BiometricProcessing("No biometric data available".to_string()))
    }
    
    /// Get historical biometric states
    pub async fn get_historical_states(&self, count: usize) -> Result<Vec<BiometricState>> {
        let buffer = self.state_buffer.read().await;
        Ok(buffer.iter()
            .rev()
            .take(count)
            .cloned()
            .collect())
    }
    
    /// Calibrate baseline biometric state
    pub async fn calibrate_baseline(&self) -> Result<()> {
        tracing::info!("Calibrating biometric baseline...");
        
        let duration = std::time::Duration::from_secs(
            self.config.baseline_calibration_duration_mins as u64 * 60
        );
        let start_time = std::time::Instant::now();
        let mut samples = Vec::new();
        
        while start_time.elapsed() < duration {
            if let Ok(reading) = self.sensors.read_all_sensors().await {
                let state = self.reading_to_state(reading);
                samples.push(state);
            }
            
            tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
        }
        
        if samples.is_empty() {
            return Err(VerumError::BiometricProcessing(
                "Failed to collect baseline samples".to_string()
            ));
        }
        
        // Calculate average baseline
        let baseline = self.calculate_average_state(&samples);
        *self.baseline_state.write().await = Some(baseline);
        
        tracing::info!("Biometric baseline calibrated with {} samples", samples.len());
        Ok(())
    }
    
    /// Main processing loop
    async fn processing_loop(&self) {
        let mut interval = tokio::time::interval(
            tokio::time::Duration::from_millis(1000 / self.config.sample_rate_hz as u64)
        );
        
        while *self.is_running.read().await {
            interval.tick().await;
            
            if let Err(e) = self.process_single_reading().await {
                tracing::error!("Biometric processing error: {}", e);
            }
        }
    }
    
    /// Process a single sensor reading
    async fn process_single_reading(&self) -> Result<()> {
        // Read from all sensors
        let reading = self.sensors.read_all_sensors().await?;
        
        // Convert to biometric state
        let mut state = self.reading_to_state(reading);
        
        // Validate the reading
        let validation = self.validator.validate_state(&state).await?;
        if !validation.is_valid {
            tracing::warn!("Invalid biometric reading: {}", validation.reason);
            return Ok(());
        }
        
        // Calculate stress and arousal levels
        if let Some(baseline) = &*self.baseline_state.read().await {
            state.stress_level = self.calculate_stress_level(&state, baseline);
            state.arousal_level = self.calculate_arousal_level(&state, baseline);
        }
        
        // Add to buffer
        let mut buffer = self.state_buffer.write().await;
        buffer.push_back(state);
        
        // Maintain buffer size
        if buffer.len() > self.config.buffer_size {
            buffer.pop_front();
        }
        
        Ok(())
    }
    
    /// Convert sensor reading to biometric state
    fn reading_to_state(&self, reading: SensorReading) -> BiometricState {
        BiometricState {
            heart_rate: reading.heart_rate.unwrap_or(70.0),
            skin_conductance: reading.skin_conductance.unwrap_or(0.5),
            muscle_tension: reading.muscle_tension.unwrap_or_else(|| vec![0.0; 16]),
            breathing_rate: reading.breathing_rate.unwrap_or(15.0),
            body_temperature: reading.body_temperature.unwrap_or(36.5),
            blood_pressure: reading.blood_pressure,
            eye_tracking: reading.eye_tracking,
            stress_level: 0.3, // Will be calculated later
            arousal_level: 0.4, // Will be calculated later
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Calculate stress level relative to baseline
    fn calculate_stress_level(&self, current: &BiometricState, baseline: &BiometricState) -> f32 {
        let hr_stress = (current.heart_rate - baseline.heart_rate) / 50.0;
        let sc_stress = (current.skin_conductance - baseline.skin_conductance) / 0.5;
        
        ((hr_stress + sc_stress) / 2.0).clamp(0.0, 1.0)
    }
    
    /// Calculate arousal level relative to baseline
    fn calculate_arousal_level(&self, current: &BiometricState, baseline: &BiometricState) -> f32 {
        let hr_arousal = (current.heart_rate - baseline.heart_rate).abs() / 30.0;
        let br_arousal = (current.breathing_rate - baseline.breathing_rate).abs() / 10.0;
        
        ((hr_arousal + br_arousal) / 2.0).clamp(0.0, 1.0)
    }
    
    /// Calculate average biometric state from samples
    fn calculate_average_state(&self, samples: &[BiometricState]) -> BiometricState {
        let count = samples.len() as f32;
        
        let avg_hr = samples.iter().map(|s| s.heart_rate).sum::<f32>() / count;
        let avg_sc = samples.iter().map(|s| s.skin_conductance).sum::<f32>() / count;
        let avg_br = samples.iter().map(|s| s.breathing_rate).sum::<f32>() / count;
        let avg_temp = samples.iter().map(|s| s.body_temperature).sum::<f32>() / count;
        
        BiometricState {
            heart_rate: avg_hr,
            skin_conductance: avg_sc,
            muscle_tension: vec![0.0; 16], // TODO: Calculate average
            breathing_rate: avg_br,
            body_temperature: avg_temp,
            blood_pressure: None,
            eye_tracking: None,
            stress_level: 0.3,
            arousal_level: 0.4,
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Clone for processing (avoiding self-referential issues)
    fn clone_for_processing(&self) -> ProcessorHandle {
        ProcessorHandle {
            sensors: self.sensors.clone(),
            validator: self.validator.clone(),
            state_buffer: self.state_buffer.clone(),
            baseline_state: self.baseline_state.clone(),
            is_running: self.is_running.clone(),
            config: self.config.clone(),
        }
    }
}

/// Handle for the processing loop to avoid self-referential issues
#[derive(Clone)]
struct ProcessorHandle {
    sensors: BiometricSensors,
    validator: BiometricValidator,
    state_buffer: std::sync::Arc<RwLock<VecDeque<BiometricState>>>,
    baseline_state: std::sync::Arc<RwLock<Option<BiometricState>>>,
    is_running: std::sync::Arc<RwLock<bool>>,
    config: BiometricsConfig,
}

impl ProcessorHandle {
    async fn processing_loop(&self) {
        // Implementation would be similar to the main processing loop
        // This is a simplified version to show the structure
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_biometric_state_comfort_level() {
        let state = BiometricState {
            heart_rate: 70.0,
            stress_level: 0.2,
            arousal_level: 0.5,
            ..BiometricState::new()
        };
        
        let comfort = state.comfort_level();
        assert!(comfort > 0.5);
    }
    
    #[test]
    fn test_fear_indication() {
        let mut state = BiometricState::new();
        state.stress_level = 0.9;
        
        assert!(state.indicates_fear(0.8));
        assert!(!state.indicates_fear(0.95));
    }
    
    #[test]
    fn test_state_similarity() {
        let state1 = BiometricState::new();
        let mut state2 = BiometricState::new();
        state2.heart_rate = 75.0; // Small difference
        
        let similarity = state1.similarity(&state2);
        assert!(similarity > 0.9);
    }
} 