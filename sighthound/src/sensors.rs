//! Sensor interface abstractions

use crate::error::Result;
use crate::timing::NanoTimestamp;
use crate::types::{SensorData, SensorType};
use crate::{SensorHealth, HealthStatus};
use async_trait::async_trait;

/// Trait for sensor interfaces
#[async_trait]
pub trait SensorInterface: Send + Sync {
    /// Get sensor type
    fn sensor_type(&self) -> SensorType;
    
    /// Start sensor data collection
    async fn start(&self) -> Result<()>;
    
    /// Stop sensor data collection
    async fn stop(&self) -> Result<()>;
    
    /// Get latest sensor data
    async fn get_data(&self) -> Result<SensorData>;
    
    /// Get sensor health status
    async fn get_health(&self) -> Result<SensorHealth>;
    
    /// Configure sensor parameters
    async fn configure(&self, config: SensorConfig) -> Result<()>;
}

/// Sensor configuration
#[derive(Debug, Clone)]
pub struct SensorConfig {
    pub sample_rate: f64,
    pub resolution: Option<SensorResolution>,
    pub range: Option<SensorRange>,
    pub filters: Vec<SensorFilter>,
}

/// Sensor resolution settings
#[derive(Debug, Clone)]
pub struct SensorResolution {
    pub horizontal: u32,
    pub vertical: u32,
    pub depth: Option<u32>,
}

/// Sensor range settings
#[derive(Debug, Clone)]
pub struct SensorRange {
    pub min: f64,
    pub max: f64,
    pub unit: String,
}

/// Sensor filter types
#[derive(Debug, Clone)]
pub enum SensorFilter {
    LowPass { cutoff: f64 },
    HighPass { cutoff: f64 },
    BandPass { low: f64, high: f64 },
    Kalman { process_noise: f64, measurement_noise: f64 },
}

/// Stub implementation for testing
pub struct StubSensor {
    sensor_type: SensorType,
    is_running: std::sync::atomic::AtomicBool,
}

impl StubSensor {
    pub fn new(sensor_type: SensorType) -> Self {
        Self {
            sensor_type,
            is_running: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl SensorInterface for StubSensor {
    fn sensor_type(&self) -> SensorType {
        self.sensor_type.clone()
    }
    
    async fn start(&self) -> Result<()> {
        self.is_running.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        self.is_running.store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
    
    async fn get_data(&self) -> Result<SensorData> {
        // Return stub data
        Ok(SensorData {
            sensor_type: self.sensor_type.clone(),
            timestamp: NanoTimestamp::now(),
            data: crate::types::SensorDataType::Distance(1.0),
            quality: crate::types::DataQuality {
                confidence: 0.9,
                noise_level: 0.1,
                completeness: 1.0,
            },
        })
    }
    
    async fn get_health(&self) -> Result<SensorHealth> {
        Ok(SensorHealth {
            sensor_type: self.sensor_type.clone(),
            status: if self.is_running.load(std::sync::atomic::Ordering::SeqCst) {
                HealthStatus::Healthy
            } else {
                HealthStatus::Unknown
            },
            last_update: NanoTimestamp::now(),
            error_rate: 0.0,
        })
    }
    
    async fn configure(&self, _config: SensorConfig) -> Result<()> {
        Ok(())
    }
} 