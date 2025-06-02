//! Error types for the Gusheshe hybrid resolution engine

use thiserror::Error;
use std::time::Duration;

/// Result type alias for Gusheshe operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in the Gusheshe resolution engine
#[derive(Error, Debug, Clone)]
pub enum Error {
    /// Timeout occurred during resolution
    #[error("Resolution timed out after {timeout:?}")]
    Timeout { timeout: Duration },

    /// Confidence level too low for decision
    #[error("Confidence {confidence} below threshold {threshold}")]
    InsufficientConfidence { confidence: f64, threshold: f64 },

    /// No valid resolution found
    #[error("No valid resolution found for point: {point_id}")]
    NoResolution { point_id: String },

    /// Conflicting evidence cannot be resolved
    #[error("Conflicting evidence cannot be resolved: {details}")]
    ConflictingEvidence { details: String },

    /// Invalid certificate format
    #[error("Invalid certificate: {reason}")]
    InvalidCertificate { reason: String },

    /// Certificate has expired
    #[error("Certificate expired at {expiry:?}")]
    ExpiredCertificate { expiry: std::time::Instant },

    /// Logical inconsistency in rules
    #[error("Logical inconsistency detected: {rule}")]
    LogicalInconsistency { rule: String },

    /// Fuzzy logic processing error
    #[error("Fuzzy logic error: {message}")]
    FuzzyLogicError { message: String },

    /// Bayesian network error
    #[error("Bayesian inference error: {message}")]
    BayesianError { message: String },

    /// Resource exhaustion (memory, CPU, etc.)
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    /// Emergency fallback triggered
    #[error("Emergency fallback triggered: {reason}")]
    EmergencyFallback { reason: String },

    /// Invalid input data
    #[error("Invalid input: {field} - {reason}")]
    InvalidInput { field: String, reason: String },

    /// System configuration error
    #[error("Configuration error: {parameter} - {reason}")]
    ConfigurationError { parameter: String, reason: String },

    /// Thread safety or concurrency error
    #[error("Concurrency error: {message}")]
    ConcurrencyError { message: String },

    /// Serialization/deserialization error
    #[error("Serialization error: {source}")]
    SerializationError { 
        #[from]
        source: serde_json::Error 
    },

    /// I/O operation failed
    #[error("I/O error: {source}")]
    IoError { 
        #[from]
        source: std::io::Error 
    },

    /// Generic error with context
    #[error("Resolution engine error: {message}")]
    Generic { message: String },
}

impl Error {
    /// Create a timeout error
    pub fn timeout(duration: Duration) -> Self {
        Self::Timeout { timeout: duration }
    }

    /// Create an insufficient confidence error
    pub fn insufficient_confidence(confidence: f64, threshold: f64) -> Self {
        Self::InsufficientConfidence { confidence, threshold }
    }

    /// Create a no resolution error
    pub fn no_resolution(point_id: impl Into<String>) -> Self {
        Self::NoResolution { point_id: point_id.into() }
    }

    /// Create a conflicting evidence error
    pub fn conflicting_evidence(details: impl Into<String>) -> Self {
        Self::ConflictingEvidence { details: details.into() }
    }

    /// Create an invalid certificate error
    pub fn invalid_certificate(reason: impl Into<String>) -> Self {
        Self::InvalidCertificate { reason: reason.into() }
    }

    /// Create an expired certificate error
    pub fn expired_certificate(expiry: std::time::Instant) -> Self {
        Self::ExpiredCertificate { expiry }
    }

    /// Create a logical inconsistency error
    pub fn logical_inconsistency(rule: impl Into<String>) -> Self {
        Self::LogicalInconsistency { rule: rule.into() }
    }

    /// Create a fuzzy logic error
    pub fn fuzzy_logic_error(message: impl Into<String>) -> Self {
        Self::FuzzyLogicError { message: message.into() }
    }

    /// Create a Bayesian error
    pub fn bayesian_error(message: impl Into<String>) -> Self {
        Self::BayesianError { message: message.into() }
    }

    /// Create a resource exhausted error
    pub fn resource_exhausted(resource: impl Into<String>) -> Self {
        Self::ResourceExhausted { resource: resource.into() }
    }

    /// Create an emergency fallback error
    pub fn emergency_fallback(reason: impl Into<String>) -> Self {
        Self::EmergencyFallback { reason: reason.into() }
    }

    /// Create an invalid input error
    pub fn invalid_input(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidInput { 
            field: field.into(), 
            reason: reason.into() 
        }
    }

    /// Create a configuration error
    pub fn configuration_error(parameter: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ConfigurationError { 
            parameter: parameter.into(), 
            reason: reason.into() 
        }
    }

    /// Create a concurrency error
    pub fn concurrency_error(message: impl Into<String>) -> Self {
        Self::ConcurrencyError { message: message.into() }
    }

    /// Create a generic error
    pub fn generic(message: impl Into<String>) -> Self {
        Self::Generic { message: message.into() }
    }

    /// Check if error is recoverable (non-fatal)
    pub fn is_recoverable(&self) -> bool {
        match self {
            Error::Timeout { .. } => true,
            Error::InsufficientConfidence { .. } => true,
            Error::NoResolution { .. } => true,
            Error::ConflictingEvidence { .. } => true,
            Error::FuzzyLogicError { .. } => true,
            Error::ResourceExhausted { .. } => false,
            Error::EmergencyFallback { .. } => false,
            Error::InvalidCertificate { .. } => false,
            Error::ExpiredCertificate { .. } => true,
            Error::LogicalInconsistency { .. } => false,
            Error::BayesianError { .. } => true,
            Error::InvalidInput { .. } => false,
            Error::ConfigurationError { .. } => false,
            Error::ConcurrencyError { .. } => true,
            Error::SerializationError { .. } => false,
            Error::IoError { .. } => true,
            Error::Generic { .. } => true,
        }
    }

    /// Check if error requires emergency fallback
    pub fn requires_emergency_fallback(&self) -> bool {
        match self {
            Error::ResourceExhausted { .. } => true,
            Error::EmergencyFallback { .. } => true,
            Error::LogicalInconsistency { .. } => true,
            Error::Timeout { .. } => true,
            _ => false,
        }
    }
} 