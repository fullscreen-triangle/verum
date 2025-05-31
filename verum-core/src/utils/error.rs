//! # Error Handling
//!
//! Comprehensive error types for the Verum system

use thiserror::Error;

/// Result type alias for Verum operations
pub type Result<T> = std::result::Result<T, VerumError>;

/// Main error type for Verum operations
#[derive(Error, Debug)]
pub enum VerumError {
    // AI and ML errors
    #[error("AI model error: {0}")]
    AIModel(String),
    
    #[error("Pattern matching failed: {0}")]
    PatternMatching(String),
    
    #[error("Fear response learning error: {0}")]
    FearResponse(String),
    
    #[error("Decision engine error: {0}")]
    DecisionEngine(String),
    
    // Biometric errors
    #[error("Biometric sensor error: {0}")]
    BiometricSensor(String),
    
    #[error("Biometric processing error: {0}")]
    BiometricProcessing(String),
    
    #[error("Biometric validation failed: {0}")]
    BiometricValidation(String),
    
    // Vehicle control errors
    #[error("Vehicle control error: {0}")]
    VehicleControl(String),
    
    #[error("Vehicle sensor error: {0}")]
    VehicleSensor(String),
    
    #[error("Safety override triggered: {0}")]
    SafetyOverride(String),
    
    // Network errors
    #[error("Network connection error: {0}")]
    NetworkConnection(String),
    
    #[error("Protocol error: {0}")]
    Protocol(String),
    
    #[error("Coordination error: {0}")]
    Coordination(String),
    
    // Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    // Data errors
    #[error("Data serialization error: {0}")]
    Serialization(String),
    
    #[error("Data validation error: {0}")]
    DataValidation(String),
    
    #[error("Database error: {0}")]
    Database(String),
    
    // IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    // Timeout errors
    #[error("Operation timed out: {0}")]
    Timeout(String),
    
    // Critical safety errors
    #[error("Critical safety violation: {0}")]
    CriticalSafety(String),
    
    #[error("Emergency stop triggered: {0}")]
    EmergencyStop(String),
    
    // External library errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),
    
    #[error("UUID error: {0}")]
    Uuid(#[from] uuid::Error),
    
    #[error("Time error: {0}")]
    Time(String),
    
    // Generic errors
    #[error("Internal error: {0}")]
    Internal(String),
    
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

impl VerumError {
    /// Check if this is a critical safety error that requires immediate attention
    pub fn is_critical_safety(&self) -> bool {
        matches!(
            self,
            VerumError::CriticalSafety(_) | VerumError::EmergencyStop(_) | VerumError::SafetyOverride(_)
        )
    }
    
    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        !matches!(
            self,
            VerumError::CriticalSafety(_) 
            | VerumError::EmergencyStop(_) 
            | VerumError::Configuration(_)
            | VerumError::FileNotFound(_)
        )
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            VerumError::CriticalSafety(_) | VerumError::EmergencyStop(_) => ErrorSeverity::Critical,
            VerumError::SafetyOverride(_) | VerumError::VehicleControl(_) => ErrorSeverity::High,
            VerumError::BiometricValidation(_) | VerumError::DecisionEngine(_) => ErrorSeverity::Medium,
            _ => ErrorSeverity::Low,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "LOW"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Helper macro for creating errors with context
#[macro_export]
macro_rules! verum_error {
    ($variant:ident, $msg:expr) => {
        VerumError::$variant($msg.to_string())
    };
    ($variant:ident, $fmt:expr, $($arg:tt)*) => {
        VerumError::$variant(format!($fmt, $($arg)*))
    };
}

/// Helper macro for returning errors
#[macro_export]
macro_rules! bail {
    ($variant:ident, $msg:expr) => {
        return Err(verum_error!($variant, $msg))
    };
    ($variant:ident, $fmt:expr, $($arg:tt)*) => {
        return Err(verum_error!($variant, $fmt, $($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_severity() {
        let critical = VerumError::CriticalSafety("test".to_string());
        let low = VerumError::Internal("test".to_string());
        
        assert_eq!(critical.severity(), ErrorSeverity::Critical);
        assert_eq!(low.severity(), ErrorSeverity::Low);
        assert!(critical.is_critical_safety());
        assert!(!low.is_critical_safety());
    }
    
    #[test]
    fn test_error_recoverability() {
        let critical = VerumError::EmergencyStop("test".to_string());
        let recoverable = VerumError::NetworkConnection("test".to_string());
        
        assert!(!critical.is_recoverable());
        assert!(recoverable.is_recoverable());
    }
} 