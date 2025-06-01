//! Vehicle Control Module

use crate::utils::{Result, VerumError, Position, Velocity, Acceleration, config::VehicleConfig};
use crate::ai::decision_engine::DrivingDecision;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Current vehicle state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleState {
    pub position: Position,
    pub velocity: Velocity,
    pub acceleration: Acceleration,
    pub heading: f32,
    pub engine_rpm: f32,
    pub fuel_level: f32,
    pub temperature: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl VehicleState {
    pub fn environmental_factors(&self) -> HashMap<String, f32> {
        let mut factors = HashMap::new();
        factors.insert("speed".to_string(), self.velocity.magnitude());
        factors.insert("acceleration".to_string(), self.acceleration.magnitude());
        factors.insert("engine_load".to_string(), self.engine_rpm / 6000.0);
        factors
    }
    
    pub fn calculate_risk_level(&self) -> f32 {
        let speed_factor = (self.velocity.magnitude() / 30.0).min(1.0);
        let accel_factor = (self.acceleration.magnitude() / 5.0).min(1.0);
        (speed_factor + accel_factor) / 2.0
    }
    
    pub fn is_at_destination(&self, _destination: &str) -> bool {
        // Simplified check - in reality would check GPS coordinates
        false
    }
}

/// Vehicle controller
pub struct VehicleController {
    config: VehicleConfig,
    current_state: VehicleState,
}

impl VehicleController {
    pub async fn new(config: VehicleConfig) -> Result<Self> {
        let current_state = VehicleState {
            position: Position::new(0.0, 0.0),
            velocity: Velocity::new(0.0, 0.0, 0.0),
            acceleration: Acceleration::new(0.0, 0.0, 0.0),
            heading: 0.0,
            engine_rpm: 800.0,
            fuel_level: 1.0,
            temperature: 20.0,
            timestamp: chrono::Utc::now(),
        };
        
        Ok(Self {
            config,
            current_state,
        })
    }
    
    pub async fn start(&self) -> Result<()> {
        tracing::info!("Starting vehicle controller");
        Ok(())
    }
    
    pub async fn stop(&self) -> Result<()> {
        tracing::info!("Stopping vehicle controller");
        Ok(())
    }
    
    pub async fn get_state(&self) -> Result<VehicleState> {
        Ok(self.current_state.clone())
    }
    
    pub async fn execute_decision(&mut self, decision: DrivingDecision) -> Result<()> {
        tracing::debug!("Executing driving decision: {:?}", decision);
        
        // Update vehicle state based on decision
        match decision {
            DrivingDecision::Accelerate(amount) => {
                self.current_state.acceleration.x = amount * self.config.max_acceleration_ms2;
            }
            DrivingDecision::Brake(amount) => {
                self.current_state.acceleration.x = -amount * self.config.max_braking_ms2;
            }
            DrivingDecision::Stop => {
                self.current_state.velocity = Velocity::new(0.0, 0.0, 0.0);
                self.current_state.acceleration = Acceleration::new(0.0, 0.0, 0.0);
            }
            _ => {} // Handle other decisions
        }
        
        self.current_state.timestamp = chrono::Utc::now();
        Ok(())
    }
} 