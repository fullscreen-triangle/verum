//! Entropy Engineering - Tangible entropy control for system optimization
//!
//! This module implements the entropy controller that can manipulate oscillation endpoints
//! to achieve desired system states through thermodynamic control laws.

use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

use crate::verum_system::VerumError;

/// Boltzmann constant for entropy calculations
const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;

/// Configuration for entropy engineering system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyConfig {
    /// Target entropy level for system optimization
    pub target_entropy: f64,
    
    /// PID control gains
    pub control_gains: ControlGains,
    
    /// Comfort optimization settings
    pub comfort_config: ComfortConfig,
    
    /// Endpoint steering parameters
    pub endpoint_config: EndpointConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlGains {
    /// Proportional gain
    pub kp: f64,
    
    /// Integral gain  
    pub ki: f64,
    
    /// Derivative gain
    pub kd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfortConfig {
    /// Maximum allowed oscillation amplitude
    pub max_amplitude: f64,
    
    /// Target frequency ranges for comfort
    pub comfort_frequency_range: (f64, f64),
    
    /// Comfort optimization weights
    pub comfort_weights: ComfortWeights,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComfortWeights {
    /// Weight for suspension optimization
    pub suspension_weight: f64,
    
    /// Weight for engine mount optimization
    pub engine_mount_weight: f64,
    
    /// Weight for HVAC optimization
    pub hvac_weight: f64,
    
    /// Weight for seat optimization
    pub seat_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    /// Number of endpoints to track
    pub max_endpoints: usize,
    
    /// Minimum energy threshold for endpoint detection
    pub energy_threshold: f64,
    
    /// Endpoint steering responsiveness
    pub steering_gain: f64,
}

/// Entropy controller implementing tangible entropy engineering
pub struct EntropyController {
    config: EntropyConfig,
    entropy_state: Arc<RwLock<EntropyState>>,
    endpoint_controller: EndpointController,
    comfort_optimizer: ComfortOptimizer,
}

impl EntropyController {
    pub fn new(config: EntropyConfig) -> Self {
        let endpoint_controller = EndpointController::new(config.control_gains.clone());
        let comfort_optimizer = ComfortOptimizer::new(config.comfort_config.clone());
        
        Self {
            config,
            entropy_state: Arc::new(RwLock::new(EntropyState::default())),
            endpoint_controller,
            comfort_optimizer,
        }
    }
    
    /// Calculate entropy from oscillation endpoints
    /// S_osc = -kB Σᵢ pᵢ ln pᵢ where pᵢ is probability of endpoint i
    pub async fn calculate_entropy_from_endpoints(
        &self,
        oscillations: &crate::verum_system::OscillationSpectrum
    ) -> Result<f64, VerumError> {
        // Extract endpoint probabilities from oscillation spectrum
        let mut endpoint_probabilities = Vec::new();
        let total_energy: f64 = oscillations.frequency_bins.iter()
            .map(|bin| bin.power_density)
            .sum();
        
        if total_energy <= 0.0 {
            return Ok(0.0);
        }
        
        // Calculate probability distribution from power density
        for bin in &oscillations.frequency_bins {
            let probability = bin.power_density / total_energy;
            if probability > 1e-10 { // Avoid log(0)
                endpoint_probabilities.push(probability);
            }
        }
        
        // Calculate Shannon entropy: S = -Σᵢ pᵢ ln pᵢ
        let entropy = endpoint_probabilities.iter()
            .map(|&p| -p * p.ln())
            .sum::<f64>();
        
        // Scale by effective Boltzmann constant for practical units
        let scaled_entropy = entropy * BOLTZMANN_CONSTANT * 1e20; // Scale for practical values
        
        // Update entropy state
        let mut state = self.entropy_state.write().await;
        state.current_entropy = scaled_entropy;
        state.entropy_error = state.target_entropy - scaled_entropy;
        state.last_update = std::time::Instant::now();
        
        Ok(scaled_entropy)
    }
    
    /// Apply entropy control to optimize system state
    pub async fn optimize_entropy(
        &self,
        oscillations: &crate::verum_system::OscillationSpectrum,
        target_entropy: Option<f64>
    ) -> Result<OptimizedState, VerumError> {
        let start_time = std::time::Instant::now();
        let original_entropy = self.calculate_entropy_from_endpoints(oscillations).await?;
        
        // Use provided target or default from config
        let target = target_entropy.unwrap_or(self.config.target_entropy);
        
        // Calculate control forces using PID controller
        let control_forces = self.endpoint_controller
            .calculate_control(original_entropy, target).await?;
        
        // Apply forces to modify oscillation spectrum
        let optimized_oscillations = self.apply_control_forces(oscillations, &control_forces).await?;
        
        // Get control precision
        let control_precision = self.endpoint_controller.get_precision().await?;
        
        let optimization_time = start_time.elapsed();
        
        Ok(OptimizedState {
            original_entropy,
            target_entropy: target,
            optimized_oscillations,
            control_precision,
            optimization_metadata: OptimizationMetadata {
                optimization_time,
                iterations: 1, // Single iteration for now
                converged: (original_entropy - target).abs() < 0.1,
                final_error: (original_entropy - target).abs(),
            },
        })
    }
    
    /// Apply control forces to oscillation spectrum
    async fn apply_control_forces(
        &self,
        spectrum: &crate::verum_system::OscillationSpectrum,
        forces: &ControlForces
    ) -> Result<crate::verum_system::OscillationSpectrum, VerumError> {
        let mut modified_bins = spectrum.frequency_bins.clone();
        
        // Apply frequency steering forces
        for (i, bin) in modified_bins.iter_mut().enumerate() {
            if let Some(&frequency_force) = forces.frequency_steering.get(i) {
                // Adjust center frequency based on steering force
                bin.center_frequency += frequency_force;
            }
            
            if let Some(&amplitude_adjustment) = forces.amplitude_adjustments.get(i) {
                // Adjust power density based on amplitude adjustment
                bin.power_density *= (1.0 + amplitude_adjustment).max(0.1);
            }
            
            if let Some(&phase_correction) = forces.phase_corrections.get(i) {
                // Apply phase correction (affects coherence)
                bin.phase_angle += phase_correction;
            }
        }
        
        // Normalize power densities to maintain energy conservation
        let total_power: f64 = modified_bins.iter().map(|bin| bin.power_density).sum();
        let original_power: f64 = spectrum.frequency_bins.iter().map(|bin| bin.power_density).sum();
        
        if total_power > 0.0 && original_power > 0.0 {
            let normalization_factor = original_power / total_power;
            for bin in &mut modified_bins {
                bin.power_density *= normalization_factor;
            }
        }
        
        Ok(crate::verum_system::OscillationSpectrum {
            frequency_bins: modified_bins,
            sample_rate: spectrum.sample_rate,
            analysis_window: spectrum.analysis_window,
            timestamp: std::time::Instant::now(),
        })
    }
    
    /// Optimize for comfort using entropy engineering
    pub async fn optimize_for_comfort(
        &self,
        profile: &crate::verum_system::OscillationProfile
    ) -> Result<ComfortOptimization, VerumError> {
        let comfort_target = self.comfort_optimizer.calculate_optimal_profile(profile).await?;
        
        Ok(ComfortOptimization {
            suspension_adjustments: comfort_target.suspension_damping_adjustments,
            engine_mount_controls: comfort_target.engine_mount_adjustments,
            hvac_oscillation_controls: comfort_target.hvac_adjustments,
            seat_adjustments: comfort_target.seat_optimization,
            expected_improvement: 1.0 - comfort_target.comfort_score,
            implementation_time: std::time::Duration::from_millis(50),
        })
    }
    
    pub async fn get_entropy_state(&self) -> EntropyState {
        self.entropy_state.read().await.clone()
    }
}

/// Current entropy state of the system
#[derive(Debug, Clone)]
pub struct EntropyState {
    /// Current entropy value
    pub current_entropy: f64,
    
    /// Target entropy value
    pub target_entropy: f64,
    
    /// Entropy error (target - current)
    pub entropy_error: f64,
    
    /// Entropy error history for integral term
    pub error_history: VecDeque<f64>,
    
    /// Previous entropy error for derivative term
    pub previous_error: f64,
    
    /// Control output
    pub control_output: f64,
    
    /// Timestamp of last update
    pub last_update: std::time::Instant,
}

impl Default for EntropyState {
    fn default() -> Self {
        Self {
            current_entropy: 0.0,
            target_entropy: 2.3, // Optimal entropy from research
            entropy_error: 0.0,
            error_history: VecDeque::with_capacity(100),
            previous_error: 0.0,
            control_output: 0.0,
            last_update: std::time::Instant::now(),
        }
    }
}

/// Optimized system state after entropy engineering
#[derive(Debug, Clone)]
pub struct OptimizedState {
    /// Original entropy before optimization
    pub original_entropy: f64,
    
    /// Target entropy for optimization
    pub target_entropy: f64,
    
    /// Optimized oscillation spectrum
    pub optimized_oscillations: crate::verum_system::OscillationSpectrum,
    
    /// Control precision achieved
    pub control_precision: f64,
    
    /// Optimization metadata
    pub optimization_metadata: OptimizationMetadata,
}

#[derive(Debug, Clone)]
pub struct OptimizationMetadata {
    /// Time taken for optimization
    pub optimization_time: std::time::Duration,
    
    /// Number of control iterations
    pub iterations: u32,
    
    /// Convergence achieved
    pub converged: bool,
    
    /// Final control error
    pub final_error: f64,
}

/// Control forces applied to oscillation endpoints
#[derive(Debug, Clone)]
pub struct ControlForces {
    /// Forces applied to steer frequency endpoints
    pub frequency_steering: Vec<f64>,
    
    /// Amplitude adjustments for oscillation control
    pub amplitude_adjustments: Vec<f64>,
    
    /// Phase corrections for coherence optimization
    pub phase_corrections: Vec<f64>,
    
    /// Total control energy applied
    pub total_energy: f64,
}

/// Endpoint controller implementing PID control theory
pub struct EndpointController {
    /// Control gains configuration
    gains: ControlGains,
    
    /// Control state history
    control_state: Arc<RwLock<ControlState>>,
    
    /// Endpoint steering configuration
    config: EndpointConfig,
}

#[derive(Debug, Clone)]
struct ControlState {
    /// Integral term accumulator
    integral_term: f64,
    
    /// Previous error for derivative calculation
    previous_error: f64,
    
    /// Control output history
    output_history: VecDeque<f64>,
    
    /// Last control update time
    last_update: std::time::Instant,
}

impl EndpointController {
    pub fn new(gains: ControlGains) -> Self {
        Self {
            gains,
            control_state: Arc::new(RwLock::new(ControlState {
                integral_term: 0.0,
                previous_error: 0.0,
                output_history: VecDeque::with_capacity(1000),
                last_update: std::time::Instant::now(),
            })),
            config: EndpointConfig::default(),
        }
    }
    
    /// Calculate PID control output
    /// F_control(t) = -K_p × e(t) - K_i × ∫e(t)dt - K_d × de/dt
    pub async fn calculate_control(&self, current_entropy: f64, target_entropy: f64) -> Result<ControlForces, VerumError> {
        let mut state = self.control_state.write().await;
        let now = std::time::Instant::now();
        let dt = now.duration_since(state.last_update).as_secs_f64();
        
        if dt < 1e-6 {
            return Err(VerumError::EntropyError("Time step too small for control calculation".to_string()));
        }
        
        // Calculate error
        let error = target_entropy - current_entropy;
        
        // Proportional term
        let proportional = self.gains.kp * error;
        
        // Integral term
        state.integral_term += error * dt;
        let integral = self.gains.ki * state.integral_term;
        
        // Derivative term
        let derivative = if dt > 0.0 {
            self.gains.kd * (error - state.previous_error) / dt
        } else {
            0.0
        };
        
        // Total control output
        let control_output = proportional + integral + derivative;
        
        // Update state
        state.previous_error = error;
        state.last_update = now;
        state.output_history.push_back(control_output);
        
        // Keep history bounded
        if state.output_history.len() > 1000 {
            state.output_history.pop_front();
        }
        
        // Convert control output to forces
        let forces = self.generate_control_forces(control_output, error).await?;
        
        Ok(forces)
    }
    
    pub async fn get_precision(&self) -> Result<f64, VerumError> {
        let state = self.control_state.read().await;
        
        if state.output_history.len() < 10 {
            return Ok(0.5); // Default precision when insufficient data
        }
        
        // Calculate precision as stability of recent control outputs
        let recent_outputs: Vec<f64> = state.output_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        
        let mean = recent_outputs.iter().sum::<f64>() / recent_outputs.len() as f64;
        let variance = recent_outputs.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / recent_outputs.len() as f64;
        
        let stability = 1.0 / (1.0 + variance);
        Ok(stability.min(1.0))
    }
    
    async fn generate_control_forces(&self, control_output: f64, error: f64) -> Result<ControlForces, VerumError> {
        // Generate control forces based on PID output and error magnitude
        let force_magnitude = control_output.abs() * self.config.steering_gain;
        let num_endpoints = self.config.max_endpoints;
        
        // Distribute forces across frequency bands
        let frequency_steering: Vec<f64> = (0..num_endpoints)
            .map(|i| {
                let frequency_factor = (i as f64 + 1.0) / num_endpoints as f64;
                force_magnitude * frequency_factor * error.signum()
            })
            .collect();
        
        // Calculate amplitude adjustments
        let amplitude_adjustments: Vec<f64> = frequency_steering.iter()
            .map(|&force| force * 0.1) // Scale amplitude adjustments
            .collect();
        
        // Calculate phase corrections for coherence
        let phase_corrections: Vec<f64> = (0..num_endpoints)
            .map(|i| {
                let phase_offset = (i as f64 * std::f64::consts::PI / num_endpoints as f64);
                control_output * 0.05 * phase_offset.sin()
            })
            .collect();
        
        let total_energy = frequency_steering.iter().map(|x| x.abs()).sum::<f64>();
        
        Ok(ControlForces {
            frequency_steering,
            amplitude_adjustments,
            phase_corrections,
            total_energy,
        })
    }
}

/// Comfort optimization system
pub struct ComfortOptimizer {
    config: ComfortConfig,
    optimization_history: Arc<RwLock<Vec<ComfortOptimization>>>,
}

impl ComfortOptimizer {
    pub fn new(config: ComfortConfig) -> Self {
        Self {
            config,
            optimization_history: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Calculate optimal oscillation profile for maximum comfort
    pub async fn calculate_optimal_profile(&self, current_profile: &crate::verum_system::OscillationProfile) -> Result<ComfortTarget, VerumError> {
        // Analyze current profile for comfort optimization opportunities
        let discomfort_sources = self.analyze_discomfort_sources(current_profile).await?;
        
        // Calculate target adjustments
        let suspension_adjustments = self.calculate_suspension_optimization(&discomfort_sources).await?;
        let engine_mount_adjustments = self.calculate_engine_mount_optimization(&discomfort_sources).await?;
        let hvac_adjustments = self.calculate_hvac_optimization(&discomfort_sources).await?;
        let seat_adjustments = self.calculate_seat_optimization(&discomfort_sources).await?;
        
        Ok(ComfortTarget {
            suspension_damping_adjustments: suspension_adjustments,
            engine_mount_adjustments,
            hvac_adjustments,
            seat_optimization: seat_adjustments,
            comfort_score: self.calculate_comfort_score(current_profile).await?,
            optimization_priority: self.calculate_optimization_priority(&discomfort_sources).await?,
        })
    }
    
    async fn analyze_discomfort_sources(&self, profile: &crate::verum_system::OscillationProfile) -> Result<DiscomfortAnalysis, VerumError> {
        let mut discomfort_sources = Vec::new();
        
        // Check for frequencies outside comfort range
        for (i, &freq) in profile.dominant_frequencies.iter().enumerate() {
            if freq < self.config.comfort_frequency_range.0 || freq > self.config.comfort_frequency_range.1 {
                let amplitude = profile.amplitude_distribution.get(i).unwrap_or(&0.0);
                if amplitude > &self.config.max_amplitude {
                    discomfort_sources.push(DiscomfortSource {
                        frequency: freq,
                        amplitude: *amplitude,
                        discomfort_level: amplitude / self.config.max_amplitude,
                        source_type: "frequency_out_of_range".to_string(),
                    });
                }
            }
        }
        
        // Check for excessive amplitudes
        for (i, &amplitude) in profile.amplitude_distribution.iter().enumerate() {
            if amplitude > self.config.max_amplitude {
                let frequency = profile.dominant_frequencies.get(i).unwrap_or(&0.0);
                discomfort_sources.push(DiscomfortSource {
                    frequency: *frequency,
                    amplitude,
                    discomfort_level: amplitude / self.config.max_amplitude,
                    source_type: "excessive_amplitude".to_string(),
                });
            }
        }
        
        // Check for poor phase coherence
        if profile.phase_coherence < 0.7 {
            discomfort_sources.push(DiscomfortSource {
                frequency: 0.0, // Global issue
                amplitude: 1.0 - profile.phase_coherence,
                discomfort_level: 1.0 - profile.phase_coherence,
                source_type: "poor_phase_coherence".to_string(),
            });
        }
        
        Ok(DiscomfortAnalysis {
            sources: discomfort_sources,
            overall_discomfort: self.calculate_overall_discomfort_score(profile).await?,
        })
    }
    
    async fn calculate_suspension_optimization(&self, analysis: &DiscomfortAnalysis) -> Result<SuspensionAdjustments, VerumError> {
        let mut front_damping_adjustment = 0.0;
        let mut rear_damping_adjustment = 0.0;
        let mut spring_stiffness_adjustment = 0.0;
        
        for source in &analysis.sources {
            // Suspension typically handles 1-15 Hz range
            if source.frequency >= 1.0 && source.frequency <= 15.0 {
                let adjustment_magnitude = source.discomfort_level * 0.2; // Max 20% adjustment
                
                if source.frequency < 3.0 {
                    // Low frequency - adjust spring stiffness
                    spring_stiffness_adjustment += adjustment_magnitude;
                } else {
                    // Higher frequency - adjust damping
                    front_damping_adjustment += adjustment_magnitude * 0.6;
                    rear_damping_adjustment += adjustment_magnitude * 0.4;
                }
            }
        }
        
        Ok(SuspensionAdjustments {
            front_damping_change: front_damping_adjustment.min(0.25), // Max 25% change
            rear_damping_change: rear_damping_adjustment.min(0.25),
            spring_stiffness_change: spring_stiffness_adjustment.min(0.15), // Max 15% change
        })
    }
    
    async fn calculate_engine_mount_optimization(&self, analysis: &DiscomfortAnalysis) -> Result<EngineMountAdjustments, VerumError> {
        let mut stiffness_adjustment = 0.0;
        let mut damping_adjustment = 0.0;
        
        for source in &analysis.sources {
            // Engine mounts typically handle 10-100 Hz range
            if source.frequency >= 10.0 && source.frequency <= 100.0 {
                let adjustment = source.discomfort_level * 0.1; // Max 10% adjustment
                stiffness_adjustment += adjustment;
                damping_adjustment += adjustment * 0.5;
            }
        }
        
        Ok(EngineMountAdjustments {
            stiffness_change: stiffness_adjustment.min(0.1),
            damping_change: damping_adjustment.min(0.05),
        })
    }
    
    async fn calculate_hvac_optimization(&self, analysis: &DiscomfortAnalysis) -> Result<HVACAdjustments, VerumError> {
        let mut blower_frequency_adjustment = 0.0;
        let mut airflow_pattern_adjustment = 0.0;
        
        for source in &analysis.sources {
            // HVAC typically operates 20-200 Hz range
            if source.frequency >= 20.0 && source.frequency <= 200.0 {
                blower_frequency_adjustment += source.discomfort_level * 2.0; // Hz adjustment
                airflow_pattern_adjustment += source.discomfort_level * 0.1;
            }
        }
        
        Ok(HVACAdjustments {
            blower_frequency_change: blower_frequency_adjustment.min(10.0), // Max 10 Hz change
            airflow_pattern_change: airflow_pattern_adjustment.min(0.2),
        })
    }
    
    async fn calculate_seat_optimization(&self, analysis: &DiscomfortAnalysis) -> Result<SeatAdjustments, VerumError> {
        let mut lumbar_support_adjustment = 0.0;
        let mut cushion_firmness_adjustment = 0.0;
        
        for source in &analysis.sources {
            // Seat comfort affected by low frequency vibrations 0.5-8 Hz
            if source.frequency >= 0.5 && source.frequency <= 8.0 {
                lumbar_support_adjustment += source.discomfort_level * 0.05;
                cushion_firmness_adjustment += source.discomfort_level * 0.03;
            }
        }
        
        Ok(SeatAdjustments {
            lumbar_support_change: lumbar_support_adjustment.min(0.1),
            cushion_firmness_change: cushion_firmness_adjustment.min(0.05),
        })
    }
    
    async fn calculate_comfort_score(&self, profile: &crate::verum_system::OscillationProfile) -> Result<f64, VerumError> {
        let mut comfort_score = 1.0;
        
        // Penalize frequencies outside comfort range
        for (i, &freq) in profile.dominant_frequencies.iter().enumerate() {
            let amplitude = profile.amplitude_distribution.get(i).unwrap_or(&0.0);
            
            if freq < self.config.comfort_frequency_range.0 || freq > self.config.comfort_frequency_range.1 {
                comfort_score -= amplitude * 0.2;
            }
            
            if amplitude > &self.config.max_amplitude {
                comfort_score -= (amplitude - self.config.max_amplitude) * 0.3;
            }
        }
        
        // Reward good phase coherence
        comfort_score += profile.phase_coherence * 0.1;
        
        // Reward low entropy (more organized oscillations)
        comfort_score += (3.0 - profile.entropy_signature) / 3.0 * 0.1;
        
        Ok(comfort_score.max(0.0).min(1.0))
    }
    
    async fn calculate_optimization_priority(&self, analysis: &DiscomfortAnalysis) -> Result<OptimizationPriority, VerumError> {
        if analysis.overall_discomfort > 0.8 {
            Ok(OptimizationPriority::Critical)
        } else if analysis.overall_discomfort > 0.5 {
            Ok(OptimizationPriority::High)
        } else if analysis.overall_discomfort > 0.2 {
            Ok(OptimizationPriority::Medium)
        } else {
            Ok(OptimizationPriority::Low)
        }
    }
    
    async fn calculate_overall_discomfort_score(&self, profile: &crate::verum_system::OscillationProfile) -> Result<f64, VerumError> {
        let comfort_score = self.calculate_comfort_score(profile).await?;
        Ok(1.0 - comfort_score)
    }
}

// Supporting data structures

#[derive(Debug, Clone)]
pub struct ComfortTarget {
    pub suspension_damping_adjustments: SuspensionAdjustments,
    pub engine_mount_adjustments: EngineMountAdjustments,
    pub hvac_adjustments: HVACAdjustments,
    pub seat_optimization: SeatAdjustments,
    pub comfort_score: f64,
    pub optimization_priority: OptimizationPriority,
}

#[derive(Debug, Clone)]
pub struct ComfortOptimization {
    pub suspension_adjustments: SuspensionAdjustments,
    pub engine_mount_controls: EngineMountAdjustments,
    pub hvac_oscillation_controls: HVACAdjustments,
    pub seat_adjustments: SeatAdjustments,
    pub expected_improvement: f64,
    pub implementation_time: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct DiscomfortAnalysis {
    pub sources: Vec<DiscomfortSource>,
    pub overall_discomfort: f64,
}

#[derive(Debug, Clone)]
pub struct DiscomfortSource {
    pub frequency: f64,
    pub amplitude: f64,
    pub discomfort_level: f64,
    pub source_type: String,
}

#[derive(Debug, Clone)]
pub struct SuspensionAdjustments {
    pub front_damping_change: f64,
    pub rear_damping_change: f64,
    pub spring_stiffness_change: f64,
}

#[derive(Debug, Clone)]
pub struct EngineMountAdjustments {
    pub stiffness_change: f64,
    pub damping_change: f64,
}

#[derive(Debug, Clone)]
pub struct HVACAdjustments {
    pub blower_frequency_change: f64,
    pub airflow_pattern_change: f64,
}

#[derive(Debug, Clone)]
pub struct SeatAdjustments {
    pub lumbar_support_change: f64,
    pub cushion_firmness_change: f64,
}

#[derive(Debug, Clone)]
pub enum OptimizationPriority {
    Critical,
    High,
    Medium,
    Low,
}

// Default implementations

impl Default for EntropyConfig {
    fn default() -> Self {
        Self {
            target_entropy: 2.3,
            control_gains: ControlGains::default(),
            comfort_config: ComfortConfig::default(),
            endpoint_config: EndpointConfig::default(),
        }
    }
}

impl Default for ControlGains {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki: 0.1,
            kd: 0.05,
        }
    }
}

impl Default for ComfortConfig {
    fn default() -> Self {
        Self {
            max_amplitude: 1.0,
            comfort_frequency_range: (0.5, 20.0), // Hz
            comfort_weights: ComfortWeights::default(),
        }
    }
}

impl Default for ComfortWeights {
    fn default() -> Self {
        Self {
            suspension_weight: 0.4,
            engine_mount_weight: 0.3,
            hvac_weight: 0.2,
            seat_weight: 0.1,
        }
    }
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            max_endpoints: 10,
            energy_threshold: 0.01,
            steering_gain: 0.1,
        }
    }
} 