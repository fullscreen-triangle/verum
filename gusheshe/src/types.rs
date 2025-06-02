//! Core types for the Gusheshe hybrid resolution engine

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Represents a confidence value between 0.0 and 1.0
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Confidence(f64);

impl Confidence {
    /// Creates a new confidence value, clamping to [0.0, 1.0]
    pub fn new(value: f64) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Returns the confidence value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Checks if confidence is above a threshold
    pub fn is_above(&self, threshold: f64) -> bool {
        self.0 > threshold
    }

    /// Combines two confidence values using multiplication (conservative)
    pub fn and(&self, other: Confidence) -> Confidence {
        Confidence::new(self.0 * other.0)
    }

    /// Combines two confidence values using probabilistic OR
    pub fn or(&self, other: Confidence) -> Confidence {
        Confidence::new(self.0 + other.0 - (self.0 * other.0))
    }
}

impl From<f64> for Confidence {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

/// Evidence that supports or challenges a point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub id: Uuid,
    pub content: String,
    pub evidence_type: EvidenceType,
    pub confidence: Confidence,
    pub source: String,
    pub timestamp: Instant,
    pub validity_window: Duration,
}

/// Types of evidence in the system
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Sensor data (radar, camera, lidar, etc.)
    Sensor,
    /// Logical rule or constraint
    Logical,
    /// Historical pattern or learned behavior
    Historical,
    /// External authority or regulation
    Regulatory,
    /// Statistical or probabilistic inference
    Statistical,
    /// Human input or override
    Human,
}

/// Evidence that supports a point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Affirmation {
    pub evidence: Evidence,
    pub strength: f64,
    pub relevance: f64,
}

/// Evidence that challenges a point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contention {
    pub evidence: Evidence,
    pub impact: f64,
    pub uncertainty: f64,
}

/// Possible actions that can be taken
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// Maintain current state/behavior
    Maintain,
    /// Execute a specific driving maneuver
    Execute(DrivingAction),
    /// Emergency safety action
    Emergency(EmergencyAction),
    /// Defer decision to human operator
    DeferToHuman,
}

/// Specific driving actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrivingAction {
    /// Change lanes (left/right)
    ChangeLane { direction: LaneDirection, urgency: Urgency },
    /// Adjust speed
    AdjustSpeed { delta_mph: i32, urgency: Urgency },
    /// Brake with specific intensity
    Brake { intensity: BrakeIntensity },
    /// Turn at intersection
    Turn { direction: TurnDirection },
    /// Merge into traffic
    Merge { target_lane: String },
    /// Pull over to side
    PullOver { reason: String },
}

/// Emergency actions for safety
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergencyAction {
    /// Emergency braking
    EmergencyBrake,
    /// Emergency stop (all systems)
    EmergencyStop,
    /// Evasive maneuver
    EvasiveManeuver { direction: LaneDirection },
    /// Activate hazard systems
    ActivateHazards,
}

/// Lane directions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaneDirection {
    Left,
    Right,
}

/// Turn directions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnDirection {
    Left,
    Right,
    UTurn,
}

/// Urgency levels for actions
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Urgency {
    Low,
    Medium,
    High,
    Critical,
}

/// Brake intensity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrakeIntensity {
    Light,      // < 0.3g
    Moderate,   // 0.3g - 0.6g
    Heavy,      // 0.6g - 0.8g
    Emergency,  // > 0.8g
}

/// Processing mode for the hybrid engine
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingMode {
    /// Deterministic rule-based processing
    Deterministic,
    /// Probabilistic fuzzy logic processing
    Probabilistic,
    /// Hybrid approach with dynamic switching
    Hybrid,
    /// Emergency mode with simplified logic
    Emergency,
}

/// Metadata attached to points and resolutions
pub type Metadata = HashMap<String, serde_json::Value>;

/// Time-bounded execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub timeout: Duration,
    pub start_time: Instant,
    pub confidence_threshold: Confidence,
    pub processing_mode: ProcessingMode,
    pub metadata: Metadata,
}

impl ExecutionContext {
    pub fn new(timeout: Duration, confidence_threshold: Confidence) -> Self {
        Self {
            timeout,
            start_time: Instant::now(),
            confidence_threshold,
            processing_mode: ProcessingMode::Hybrid,
            metadata: HashMap::new(),
        }
    }

    /// Check if execution has timed out
    pub fn is_timed_out(&self) -> bool {
        self.start_time.elapsed() > self.timeout
    }

    /// Get remaining time
    pub fn remaining_time(&self) -> Duration {
        self.timeout.saturating_sub(self.start_time.elapsed())
    }

    /// Check if we have enough time for a given operation
    pub fn has_time_for(&self, operation_duration: Duration) -> bool {
        self.remaining_time() > operation_duration
    }
} 