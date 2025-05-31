//! # Configuration Module
//!
//! Configuration management for all Verum components

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration structure for Verum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub ai: AIConfig,
    pub biometrics: BiometricsConfig,
    pub vehicle: VehicleConfig,
    pub network: NetworkConfig,
    pub logging: LoggingConfig,
    pub data: DataConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ai: AIConfig::default(),
            biometrics: BiometricsConfig::default(),
            vehicle: VehicleConfig::default(),
            network: NetworkConfig::default(),
            logging: LoggingConfig::default(),
            data: DataConfig::default(),
        }
    }
}

/// AI system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIConfig {
    pub model_path: String,
    pub pattern_similarity_threshold: f32,
    pub fear_response_sensitivity: f32,
    pub decision_timeout_ms: u64,
    pub max_stress_threshold: f32,
    pub learning_rate: f32,
    pub enable_cross_domain_learning: bool,
    pub supported_domains: Vec<String>,
    pub model_update_interval_mins: u32,
}

impl Default for AIConfig {
    fn default() -> Self {
        Self {
            model_path: "/var/lib/verum/models".to_string(),
            pattern_similarity_threshold: 0.7,
            fear_response_sensitivity: 0.8,
            decision_timeout_ms: 100,
            max_stress_threshold: 0.85,
            learning_rate: 0.001,
            enable_cross_domain_learning: true,
            supported_domains: vec![
                "driving".to_string(),
                "walking".to_string(),
                "tennis".to_string(),
                "cycling".to_string(),
            ],
            model_update_interval_mins: 60,
        }
    }
}

/// Biometric processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricsConfig {
    pub sample_rate_hz: u32,
    pub buffer_size: usize,
    pub enabled_sensors: Vec<String>,
    pub heart_rate_range: (f32, f32),
    pub skin_conductance_threshold: f32,
    pub muscle_tension_sensitivity: f32,
    pub validation_window_ms: u64,
    pub baseline_calibration_duration_mins: u32,
}

impl Default for BiometricsConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 100,
            buffer_size: 1000,
            enabled_sensors: vec![
                "heart_rate".to_string(),
                "skin_conductance".to_string(),
                "accelerometer".to_string(),
                "gyroscope".to_string(),
            ],
            heart_rate_range: (50.0, 200.0),
            skin_conductance_threshold: 0.5,
            muscle_tension_sensitivity: 0.8,
            validation_window_ms: 1000,
            baseline_calibration_duration_mins: 10,
        }
    }
}

/// Vehicle control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleConfig {
    pub max_speed_kmh: f32,
    pub max_acceleration_ms2: f32,
    pub max_braking_ms2: f32,
    pub steering_sensitivity: f32,
    pub safety_distance_m: f32,
    pub emergency_brake_threshold: f32,
    pub control_loop_frequency_hz: u32,
    pub enable_safety_override: bool,
    pub sensor_fusion_enabled: bool,
}

impl Default for VehicleConfig {
    fn default() -> Self {
        Self {
            max_speed_kmh: 130.0,
            max_acceleration_ms2: 3.0,
            max_braking_ms2: 8.0,
            steering_sensitivity: 1.0,
            safety_distance_m: 2.0,
            emergency_brake_threshold: 0.9,
            control_loop_frequency_hz: 50,
            enable_safety_override: true,
            sensor_fusion_enabled: true,
        }
    }
}

/// Network communication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub port: u16,
    pub coordinator_url: String,
    pub discovery_interval_ms: u64,
    pub max_connections: u32,
    pub timeout_ms: u64,
    pub enable_tls: bool,
    pub certificate_path: Option<String>,
    pub enable_coordination: bool,
    pub mesh_network: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            coordinator_url: "http://localhost:8081".to_string(),
            discovery_interval_ms: 30000,
            max_connections: 1000,
            timeout_ms: 5000,
            enable_tls: false,
            certificate_path: None,
            enable_coordination: true,
            mesh_network: false,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file_path: Option<String>,
    pub console_output: bool,
    pub structured_logging: bool,
    pub max_file_size_mb: u64,
    pub max_files: u32,
    pub enable_metrics: bool,
    pub metrics_port: u16,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file_path: Some("/var/log/verum/verum.log".to_string()),
            console_output: true,
            structured_logging: true,
            max_file_size_mb: 100,
            max_files: 10,
            enable_metrics: true,
            metrics_port: 9090,
        }
    }
}

/// Data storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub data_dir: String,
    pub database_url: Option<String>,
    pub cache_size_mb: u64,
    pub backup_interval_hours: u32,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
    pub privacy_mode: PrivacyMode,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            data_dir: "/var/lib/verum/data".to_string(),
            database_url: None,
            cache_size_mb: 512,
            backup_interval_hours: 24,
            compression_enabled: true,
            encryption_enabled: true,
            privacy_mode: PrivacyMode::Personal,
        }
    }
}

/// Privacy modes for data handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyMode {
    /// No privacy protection (research/testing only)
    None,
    /// Basic anonymization
    Basic,
    /// Personal data protection (default)
    Personal,
    /// Maximum privacy with differential privacy
    Maximum,
}

impl Config {
    /// Load configuration from file
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> crate::utils::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> crate::utils::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Load configuration from environment variables
    pub fn from_env() -> crate::utils::Result<Self> {
        let mut config = Config::default();
        
        // AI configuration
        if let Ok(model_path) = std::env::var("VERUM_AI_MODEL_PATH") {
            config.ai.model_path = model_path;
        }
        if let Ok(threshold) = std::env::var("VERUM_AI_PATTERN_THRESHOLD") {
            config.ai.pattern_similarity_threshold = threshold.parse()
                .map_err(|_| crate::utils::VerumError::Configuration("Invalid pattern threshold".to_string()))?;
        }
        
        // Network configuration
        if let Ok(port) = std::env::var("VERUM_NETWORK_PORT") {
            config.network.port = port.parse()
                .map_err(|_| crate::utils::VerumError::Configuration("Invalid network port".to_string()))?;
        }
        if let Ok(coordinator_url) = std::env::var("VERUM_COORDINATOR_URL") {
            config.network.coordinator_url = coordinator_url;
        }
        
        // Logging configuration
        if let Ok(log_level) = std::env::var("VERUM_LOG_LEVEL") {
            config.logging.level = log_level;
        }
        
        // Data configuration
        if let Ok(data_dir) = std::env::var("VERUM_DATA_DIR") {
            config.data.data_dir = data_dir;
        }
        if let Ok(database_url) = std::env::var("VERUM_DATABASE_URL") {
            config.data.database_url = Some(database_url);
        }
        
        config.validate()?;
        Ok(config)
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> crate::utils::Result<()> {
        // Validate AI configuration
        if self.ai.pattern_similarity_threshold < 0.0 || self.ai.pattern_similarity_threshold > 1.0 {
            return Err(crate::utils::VerumError::Configuration(
                "Pattern similarity threshold must be between 0.0 and 1.0".to_string()
            ));
        }
        
        if self.ai.fear_response_sensitivity < 0.0 || self.ai.fear_response_sensitivity > 1.0 {
            return Err(crate::utils::VerumError::Configuration(
                "Fear response sensitivity must be between 0.0 and 1.0".to_string()
            ));
        }
        
        // Validate biometric configuration
        if self.biometrics.sample_rate_hz == 0 {
            return Err(crate::utils::VerumError::Configuration(
                "Biometric sample rate must be greater than 0".to_string()
            ));
        }
        
        // Validate vehicle configuration
        if self.vehicle.max_speed_kmh <= 0.0 {
            return Err(crate::utils::VerumError::Configuration(
                "Maximum speed must be greater than 0".to_string()
            ));
        }
        
        // Validate network configuration
        if self.network.port == 0 {
            return Err(crate::utils::VerumError::Configuration(
                "Network port must be greater than 0".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Create directories specified in configuration
    pub fn create_directories(&self) -> crate::utils::Result<()> {
        // Create data directory
        std::fs::create_dir_all(&self.data.data_dir)?;
        
        // Create AI model directory
        let model_dir = PathBuf::from(&self.ai.model_path);
        if let Some(parent) = model_dir.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Create log directory if specified
        if let Some(log_path) = &self.logging.file_path {
            let log_dir = PathBuf::from(log_path);
            if let Some(parent) = log_dir.parent() {
                std::fs::create_dir_all(parent)?;
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        
        // Test invalid pattern threshold
        config.ai.pattern_similarity_threshold = 1.5;
        assert!(config.validate().is_err());
        
        // Reset and test invalid port
        config = Config::default();
        config.network.port = 0;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_privacy_mode_serialization() {
        let mode = PrivacyMode::Personal;
        let serialized = serde_json::to_string(&mode).unwrap();
        let deserialized: PrivacyMode = serde_json::from_str(&serialized).unwrap();
        assert!(matches!(deserialized, PrivacyMode::Personal));
    }
} 