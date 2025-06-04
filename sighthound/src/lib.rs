//! Sighthound - Nanosecond-precision sensor fusion for autonomous driving
//!
//! Sighthound provides atomic-level timing precision for sensor data fusion,
//! enabling revolutionary behavioral timestamping and predictive maintenance
//! capabilities for autonomous driving systems.
//!
//! ## Key Features
//!
//! - **Nanosecond Precision**: Atomic clock-level timing for all sensor data
//! - **Behavioral Timestamping**: Precise correlation of driver/vehicle behaviors
//! - **Predictive Maintenance**: Early detection of component degradation
//! - **Multi-sensor Fusion**: Synchronized data from LiDAR, cameras, radar, IMU, GPS
//! - **Real-time Processing**: Sub-millisecond latency for critical decisions
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │   Sensor        │    │   Timing         │    │   Fusion        │
//! │   Interfaces    │───▶│   Synchronizer   │───▶│   Engine        │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//!          │                       │                       │
//!          ▼                       ▼                       ▼
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │   Hardware      │    │   Nanosecond     │    │   Behavioral    │
//! │   Abstraction   │    │   Clock          │    │   Analysis      │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//! ```

pub mod timing;
pub mod sensors;
pub mod fusion;
pub mod behavioral;
pub mod maintenance;
pub mod error;
pub mod types;

// Re-export main types
pub use timing::{NanoTimestamp, TimingSynchronizer, AtomicClock};
pub use sensors::{SensorInterface, SensorData, SensorType};
pub use fusion::{FusionEngine, FusionFrame, SynchronizedData};
pub use behavioral::{BehavioralAnalyzer, BehavioralEvent, BehavioralPattern};
pub use maintenance::{MaintenancePredictor, ComponentHealth, DegradationPattern};
pub use error::{SighthoundError, Result};
pub use types::*;

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Main Sighthound sensor fusion system
pub struct Sighthound {
    /// Timing synchronization system
    timing: Arc<TimingSynchronizer>,
    
    /// Sensor interface manager
    sensors: Arc<RwLock<Vec<Box<dyn SensorInterface>>>>,
    
    /// Fusion engine
    fusion: Arc<FusionEngine>,
    
    /// Behavioral analysis system
    behavioral: Arc<BehavioralAnalyzer>,
    
    /// Maintenance prediction system
    maintenance: Arc<MaintenancePredictor>,
    
    /// Configuration
    config: SighthoundConfig,
}

impl Sighthound {
    /// Create a new Sighthound system with the given configuration
    pub async fn new(config: SighthoundConfig) -> Result<Self> {
        info!("Initializing Sighthound sensor fusion system");
        
        let timing = Arc::new(TimingSynchronizer::new(config.timing_config.clone()).await?);
        let sensors = Arc::new(RwLock::new(Vec::new()));
        let fusion = Arc::new(FusionEngine::new(config.fusion_config.clone()).await?);
        let behavioral = Arc::new(BehavioralAnalyzer::new(config.behavioral_config.clone()).await?);
        let maintenance = Arc::new(MaintenancePredictor::new(config.maintenance_config.clone()).await?);
        
        Ok(Self {
            timing,
            sensors,
            fusion,
            behavioral,
            maintenance,
            config,
        })
    }
    
    /// Start the sensor fusion system
    pub async fn start(&self) -> Result<()> {
        info!("Starting Sighthound sensor fusion system");
        
        // Start timing synchronization
        self.timing.start().await?;
        
        // Initialize sensors
        self.initialize_sensors().await?;
        
        // Start fusion engine
        self.fusion.start().await?;
        
        // Start behavioral analysis
        self.behavioral.start().await?;
        
        // Start maintenance prediction
        self.maintenance.start().await?;
        
        info!("Sighthound system started successfully");
        Ok(())
    }
    
    /// Stop the sensor fusion system
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Sighthound sensor fusion system");
        
        // Stop components in reverse order
        self.maintenance.stop().await?;
        self.behavioral.stop().await?;
        self.fusion.stop().await?;
        self.timing.stop().await?;
        
        info!("Sighthound system stopped successfully");
        Ok(())
    }
    
    /// Get the current synchronized sensor data
    pub async fn get_current_frame(&self) -> Result<FusionFrame> {
        self.fusion.get_current_frame().await
    }
    
    /// Get behavioral analysis for a specific time window
    pub async fn get_behavioral_analysis(&self, start: NanoTimestamp, end: NanoTimestamp) -> Result<Vec<BehavioralEvent>> {
        self.behavioral.analyze_time_window(start, end).await
    }
    
    /// Get maintenance predictions
    pub async fn get_maintenance_predictions(&self) -> Result<Vec<ComponentHealth>> {
        self.maintenance.get_current_health().await
    }
    
    /// Register a new sensor
    pub async fn register_sensor(&self, sensor: Box<dyn SensorInterface>) -> Result<()> {
        let mut sensors = self.sensors.write().await;
        sensors.push(sensor);
        Ok(())
    }
    
    /// Get system metrics
    pub async fn get_metrics(&self) -> Result<SighthoundMetrics> {
        Ok(SighthoundMetrics {
            timing_accuracy: self.timing.get_accuracy().await?,
            fusion_latency: self.fusion.get_latency().await?,
            sensor_health: self.get_sensor_health().await?,
            behavioral_events_detected: self.behavioral.get_event_count().await?,
            maintenance_alerts: self.maintenance.get_alert_count().await?,
        })
    }
    
    async fn initialize_sensors(&self) -> Result<()> {
        // Initialize sensors based on configuration
        // This would be implemented based on specific hardware requirements
        Ok(())
    }
    
    async fn get_sensor_health(&self) -> Result<Vec<SensorHealth>> {
        let sensors = self.sensors.read().await;
        let mut health = Vec::new();
        
        for sensor in sensors.iter() {
            health.push(sensor.get_health().await?);
        }
        
        Ok(health)
    }
}

/// Configuration for the Sighthound system
#[derive(Debug, Clone)]
pub struct SighthoundConfig {
    pub timing_config: timing::TimingConfig,
    pub fusion_config: fusion::FusionConfig,
    pub behavioral_config: behavioral::BehavioralConfig,
    pub maintenance_config: maintenance::MaintenanceConfig,
}

impl Default for SighthoundConfig {
    fn default() -> Self {
        Self {
            timing_config: timing::TimingConfig::default(),
            fusion_config: fusion::FusionConfig::default(),
            behavioral_config: behavioral::BehavioralConfig::default(),
            maintenance_config: maintenance::MaintenanceConfig::default(),
        }
    }
}

/// System metrics for monitoring Sighthound performance
#[derive(Debug, Clone)]
pub struct SighthoundMetrics {
    pub timing_accuracy: f64,
    pub fusion_latency: std::time::Duration,
    pub sensor_health: Vec<SensorHealth>,
    pub behavioral_events_detected: u64,
    pub maintenance_alerts: u64,
}

/// Health status of a sensor
#[derive(Debug, Clone)]
pub struct SensorHealth {
    pub sensor_type: SensorType,
    pub status: HealthStatus,
    pub last_update: NanoTimestamp,
    pub error_rate: f64,
}

/// Health status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Failed,
    Unknown,
} 