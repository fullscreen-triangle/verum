//! # Vehicle Control System
//!
//! Executes driving decisions from the personal intelligence engine through
//! precise vehicle control commands with safety validation.

use crate::utils::{Result, VerumError};
use crate::intelligence::DrivingDecision;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use chrono::{DateTime, Utc};

/// Vehicle control system that executes intelligence decisions
pub struct VehicleController {
    actuator_interface: ActuatorInterface,
    control_validator: ControlValidator,
    execution_monitor: ExecutionMonitor,
    feedback_system: FeedbackSystem,
}

impl VehicleController {
    pub fn new() -> Self {
        Self {
            actuator_interface: ActuatorInterface::new(),
            control_validator: ControlValidator::new(),
            execution_monitor: ExecutionMonitor::new(),
            feedback_system: FeedbackSystem::new(),
        }
    }
    
    /// Execute a sequence of control commands
    pub async fn execute_commands(&mut self, commands: Vec<ControlCommand>) -> Result<ExecutionResult> {
        // Validate commands before execution
        for command in &commands {
            self.control_validator.validate_command(command).await?;
        }
        
        // Execute commands through actuator interface
        let mut execution_results = vec![];
        for command in commands {
            let result = self.actuator_interface.execute_command(command).await?;
            execution_results.push(result);
            
            // Monitor execution in real-time
            self.execution_monitor.monitor_execution(&result).await?;
        }
        
        // Aggregate results
        Ok(ExecutionResult {
            commands_executed: execution_results.len(),
            success_rate: execution_results.iter().map(|r| if r.success { 1.0 } else { 0.0 }).sum::<f32>() / execution_results.len() as f32,
            execution_time: execution_results.iter().map(|r| r.execution_time).sum(),
            feedback: self.feedback_system.generate_feedback(&execution_results).await?,
        })
    }
}

/// Control commands that can be executed by the vehicle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlCommand {
    // Throttle control
    Throttle(f32), // 0-1 throttle position
    
    // Braking control
    Brake(f32), // 0-1 brake pressure
    EmergencyBrake,
    
    // Steering control
    Steer(f32), // -1 to 1 steering angle
    EmergencySwerve(f32), // Emergency swerve direction
    
    // Lane changes
    SignalLeft,
    SignalRight,
    SignalOff,
    
    // Complex maneuvers
    ChangeLane(LaneChangeParams),
    PullOver,
    Merge(MergeParams),
    
    // Maintain current state
    Maintain,
    
    // Stop vehicle
    Stop,
    
    // Speed control
    SetSpeed(f32), // Target speed in m/s
    CruiseControl(f32), // Enable cruise at speed
    
    // Following behavior
    FollowVehicle(FollowParams),
    MaintainDistance(f32), // Distance in meters
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneChangeParams {
    pub direction: LaneDirection,
    pub urgency: f32, // 0-1 urgency level
    pub target_position: Option<f32>, // Target lane position
    pub duration: Option<Duration>, // Expected duration
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LaneDirection {
    Left,
    Right,
    Center,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeParams {
    pub merge_point: (f32, f32), // GPS coordinates
    pub target_speed: f32,
    pub gap_size: f32,
    pub time_to_merge: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FollowParams {
    pub target_vehicle_id: String,
    pub following_distance: f32, // Meters
    pub speed_matching: bool,
    pub adaptation_rate: f32, // How quickly to adapt to target
}

/// Actuator interface for vehicle control
pub struct ActuatorInterface {
    throttle_actuator: ThrottleActuator,
    brake_actuator: BrakeActuator,
    steering_actuator: SteeringActuator,
    signal_actuator: SignalActuator,
    transmission_actuator: TransmissionActuator,
}

impl ActuatorInterface {
    pub fn new() -> Self {
        Self {
            throttle_actuator: ThrottleActuator::new(),
            brake_actuator: BrakeActuator::new(),
            steering_actuator: SteeringActuator::new(),
            signal_actuator: SignalActuator::new(),
            transmission_actuator: TransmissionActuator::new(),
        }
    }
    
    pub async fn execute_command(&mut self, command: ControlCommand) -> Result<CommandExecutionResult> {
        let start_time = std::time::Instant::now();
        
        let success = match command {
            ControlCommand::Throttle(position) => {
                self.throttle_actuator.set_position(position).await?
            }
            ControlCommand::Brake(pressure) => {
                self.brake_actuator.set_pressure(pressure).await?
            }
            ControlCommand::EmergencyBrake => {
                self.brake_actuator.emergency_brake().await?
            }
            ControlCommand::Steer(angle) => {
                self.steering_actuator.set_angle(angle).await?
            }
            ControlCommand::EmergencySwerve(direction) => {
                self.steering_actuator.emergency_swerve(direction).await?
            }
            ControlCommand::SignalLeft => {
                self.signal_actuator.signal_left().await?
            }
            ControlCommand::SignalRight => {
                self.signal_actuator.signal_right().await?
            }
            ControlCommand::SignalOff => {
                self.signal_actuator.signal_off().await?
            }
            ControlCommand::ChangeLane(params) => {
                self.execute_lane_change(params).await?
            }
            ControlCommand::PullOver => {
                self.execute_pull_over().await?
            }
            ControlCommand::Merge(params) => {
                self.execute_merge(params).await?
            }
            ControlCommand::Maintain => {
                true // Always successful
            }
            ControlCommand::Stop => {
                self.execute_stop().await?
            }
            ControlCommand::SetSpeed(speed) => {
                self.execute_set_speed(speed).await?
            }
            ControlCommand::CruiseControl(speed) => {
                self.execute_cruise_control(speed).await?
            }
            ControlCommand::FollowVehicle(params) => {
                self.execute_follow_vehicle(params).await?
            }
            ControlCommand::MaintainDistance(distance) => {
                self.execute_maintain_distance(distance).await?
            }
        };
        
        Ok(CommandExecutionResult {
            command: command,
            success,
            execution_time: start_time.elapsed(),
            timestamp: Utc::now(),
            error_message: if success { None } else { Some("Execution failed".to_string()) },
        })
    }
    
    async fn execute_lane_change(&mut self, params: LaneChangeParams) -> Result<bool> {
        // Complex lane change execution
        match params.direction {
            LaneDirection::Left => {
                self.signal_actuator.signal_left().await?;
                // Gradual steering left
                self.steering_actuator.gradual_steer(-0.1, params.duration.unwrap_or(Duration::from_secs(3))).await?;
            }
            LaneDirection::Right => {
                self.signal_actuator.signal_right().await?;
                // Gradual steering right
                self.steering_actuator.gradual_steer(0.1, params.duration.unwrap_or(Duration::from_secs(3))).await?;
            }
            LaneDirection::Center => {
                // Return to center
                self.steering_actuator.return_to_center().await?;
            }
        }
        
        // Turn off signal after lane change
        tokio::time::sleep(params.duration.unwrap_or(Duration::from_secs(3))).await;
        self.signal_actuator.signal_off().await?;
        
        Ok(true)
    }
    
    async fn execute_pull_over(&mut self) -> Result<bool> {
        // Signal right
        self.signal_actuator.signal_right().await?;
        
        // Gradual deceleration
        self.throttle_actuator.set_position(0.0).await?;
        self.brake_actuator.gradual_brake(0.3, Duration::from_secs(5)).await?;
        
        // Steer to shoulder
        self.steering_actuator.gradual_steer(0.2, Duration::from_secs(4)).await?;
        
        // Complete stop
        self.brake_actuator.set_pressure(1.0).await?;
        self.transmission_actuator.set_park().await?;
        
        Ok(true)
    }
    
    async fn execute_merge(&mut self, params: MergeParams) -> Result<bool> {
        // Signal for merge
        self.signal_actuator.signal_left().await?; // Assuming left merge
        
        // Adjust speed to match target
        if params.target_speed > 0.0 {
            self.execute_set_speed(params.target_speed).await?;
        }
        
        // Wait for gap and merge
        tokio::time::sleep(params.time_to_merge).await;
        self.steering_actuator.gradual_steer(-0.15, Duration::from_secs(2)).await?;
        
        // Turn off signal
        self.signal_actuator.signal_off().await?;
        
        Ok(true)
    }
    
    async fn execute_stop(&mut self) -> Result<bool> {
        // Gradual deceleration to stop
        self.throttle_actuator.set_position(0.0).await?;
        self.brake_actuator.gradual_brake(0.8, Duration::from_secs(3)).await?;
        self.transmission_actuator.set_park().await?;
        Ok(true)
    }
    
    async fn execute_set_speed(&mut self, target_speed: f32) -> Result<bool> {
        // Simple speed control (in real implementation, this would be PID)
        let current_speed = 20.0; // Placeholder - would get from sensors
        
        if target_speed > current_speed {
            self.throttle_actuator.set_position(0.6).await?;
        } else if target_speed < current_speed {
            self.brake_actuator.set_pressure(0.3).await?;
        }
        
        Ok(true)
    }
    
    async fn execute_cruise_control(&mut self, speed: f32) -> Result<bool> {
        // Enable cruise control at target speed
        Ok(true) // Placeholder
    }
    
    async fn execute_follow_vehicle(&mut self, params: FollowParams) -> Result<bool> {
        // Vehicle following logic
        Ok(true) // Placeholder
    }
    
    async fn execute_maintain_distance(&mut self, distance: f32) -> Result<bool> {
        // Distance maintenance logic
        Ok(true) // Placeholder
    }
}

/// Individual actuator implementations
pub struct ThrottleActuator;
impl ThrottleActuator {
    pub fn new() -> Self { Self }
    pub async fn set_position(&mut self, position: f32) -> Result<bool> {
        // Throttle control implementation
        Ok(true)
    }
}

pub struct BrakeActuator;
impl BrakeActuator {
    pub fn new() -> Self { Self }
    pub async fn set_pressure(&mut self, pressure: f32) -> Result<bool> {
        // Brake control implementation
        Ok(true)
    }
    pub async fn emergency_brake(&mut self) -> Result<bool> {
        // Emergency braking
        self.set_pressure(1.0).await
    }
    pub async fn gradual_brake(&mut self, pressure: f32, duration: Duration) -> Result<bool> {
        // Gradual braking over time
        tokio::time::sleep(duration).await;
        self.set_pressure(pressure).await
    }
}

pub struct SteeringActuator;
impl SteeringActuator {
    pub fn new() -> Self { Self }
    pub async fn set_angle(&mut self, angle: f32) -> Result<bool> {
        // Steering control implementation
        Ok(true)
    }
    pub async fn emergency_swerve(&mut self, direction: f32) -> Result<bool> {
        // Emergency swerve maneuver
        self.set_angle(direction).await
    }
    pub async fn gradual_steer(&mut self, angle: f32, duration: Duration) -> Result<bool> {
        // Gradual steering over time
        tokio::time::sleep(duration).await;
        self.set_angle(angle).await
    }
    pub async fn return_to_center(&mut self) -> Result<bool> {
        // Return steering to center
        self.set_angle(0.0).await
    }
}

pub struct SignalActuator;
impl SignalActuator {
    pub fn new() -> Self { Self }
    pub async fn signal_left(&mut self) -> Result<bool> {
        // Left turn signal
        Ok(true)
    }
    pub async fn signal_right(&mut self) -> Result<bool> {
        // Right turn signal
        Ok(true)
    }
    pub async fn signal_off(&mut self) -> Result<bool> {
        // Turn off signals
        Ok(true)
    }
}

pub struct TransmissionActuator;
impl TransmissionActuator {
    pub fn new() -> Self { Self }
    pub async fn set_park(&mut self) -> Result<bool> {
        // Set transmission to park
        Ok(true)
    }
    pub async fn set_drive(&mut self) -> Result<bool> {
        // Set transmission to drive
        Ok(true)
    }
    pub async fn set_reverse(&mut self) -> Result<bool> {
        // Set transmission to reverse
        Ok(true)
    }
}

/// Control validation system
pub struct ControlValidator;
impl ControlValidator {
    pub fn new() -> Self { Self }
    
    pub async fn validate_command(&self, command: &ControlCommand) -> Result<()> {
        match command {
            ControlCommand::Throttle(position) => {
                if *position < 0.0 || *position > 1.0 {
                    return Err(VerumError::InvalidControlCommand("Throttle position out of range".to_string()));
                }
            }
            ControlCommand::Brake(pressure) => {
                if *pressure < 0.0 || *pressure > 1.0 {
                    return Err(VerumError::InvalidControlCommand("Brake pressure out of range".to_string()));
                }
            }
            ControlCommand::Steer(angle) => {
                if *angle < -1.0 || *angle > 1.0 {
                    return Err(VerumError::InvalidControlCommand("Steering angle out of range".to_string()));
                }
            }
            ControlCommand::SetSpeed(speed) => {
                if *speed < 0.0 || *speed > 40.0 { // Max 40 m/s (~90 mph)
                    return Err(VerumError::InvalidControlCommand("Speed out of safe range".to_string()));
                }
            }
            _ => {} // Other commands don't need validation
        }
        Ok(())
    }
}

/// Execution monitoring system
pub struct ExecutionMonitor;
impl ExecutionMonitor {
    pub fn new() -> Self { Self }
    
    pub async fn monitor_execution(&mut self, result: &CommandExecutionResult) -> Result<()> {
        // Monitor command execution for anomalies
        if !result.success {
            // Log execution failure
            eprintln!("Command execution failed: {:?}", result.command);
        }
        
        if result.execution_time > Duration::from_millis(100) {
            // Log slow execution
            eprintln!("Slow command execution: {:?} took {:?}", result.command, result.execution_time);
        }
        
        Ok(())
    }
}

/// Feedback system for control performance
pub struct FeedbackSystem;
impl FeedbackSystem {
    pub fn new() -> Self { Self }
    
    pub async fn generate_feedback(&self, results: &[CommandExecutionResult]) -> Result<ControlFeedback> {
        let total_commands = results.len();
        let successful_commands = results.iter().filter(|r| r.success).count();
        let average_execution_time = results.iter().map(|r| r.execution_time).sum::<Duration>() / total_commands as u32;
        
        Ok(ControlFeedback {
            success_rate: successful_commands as f32 / total_commands as f32,
            average_execution_time,
            total_commands,
            recommendations: self.generate_recommendations(results).await?,
        })
    }
    
    async fn generate_recommendations(&self, results: &[CommandExecutionResult]) -> Result<Vec<String>> {
        let mut recommendations = vec![];
        
        // Analyze patterns and generate recommendations
        let slow_commands = results.iter().filter(|r| r.execution_time > Duration::from_millis(50)).count();
        if slow_commands > results.len() / 4 {
            recommendations.push("Consider optimizing actuator response times".to_string());
        }
        
        let failed_commands = results.iter().filter(|r| !r.success).count();
        if failed_commands > 0 {
            recommendations.push("Investigate command execution failures".to_string());
        }
        
        Ok(recommendations)
    }
}

/// Results and feedback structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandExecutionResult {
    pub command: ControlCommand,
    pub success: bool,
    pub execution_time: Duration,
    pub timestamp: DateTime<Utc>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub commands_executed: usize,
    pub success_rate: f32,
    pub execution_time: Duration,
    pub feedback: ControlFeedback,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlFeedback {
    pub success_rate: f32,
    pub average_execution_time: Duration,
    pub total_commands: usize,
    pub recommendations: Vec<String>,
} 