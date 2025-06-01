//! Utility types and functions for the Verum system

use std::fmt;

pub type Result<T> = std::result::Result<T, VerumError>;

#[derive(Debug)]
pub enum VerumError {
    DataCollectionError(String),
    PatternExtractionError(String),
    LearningError(String),
    IntelligenceError(String),
    ConfigurationError(String),
    IoError(std::io::Error),
    SerializationError(String),
    NetworkError(String),
    
    // New error types for specialized intelligence system
    AgentNotFound(String),
    ProcessingError(String),
    TemporalPrecisionError(String),
    QuantumPatternError(String),
    OrchestrationError(String),
    MetacognitiveError(String),
}

impl fmt::Display for VerumError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VerumError::DataCollectionError(msg) => write!(f, "Data collection error: {}", msg),
            VerumError::PatternExtractionError(msg) => write!(f, "Pattern extraction error: {}", msg),
            VerumError::LearningError(msg) => write!(f, "Learning error: {}", msg),
            VerumError::IntelligenceError(msg) => write!(f, "Intelligence error: {}", msg),
            VerumError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            VerumError::IoError(err) => write!(f, "IO error: {}", err),
            VerumError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            VerumError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            VerumError::AgentNotFound(msg) => write!(f, "Agent not found: {}", msg),
            VerumError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
            VerumError::TemporalPrecisionError(msg) => write!(f, "Temporal precision error: {}", msg),
            VerumError::QuantumPatternError(msg) => write!(f, "Quantum pattern error: {}", msg),
            VerumError::OrchestrationError(msg) => write!(f, "Orchestration error: {}", msg),
            VerumError::MetacognitiveError(msg) => write!(f, "Metacognitive error: {}", msg),
        }
    }
}

impl std::error::Error for VerumError {}

impl From<std::io::Error> for VerumError {
    fn from(error: std::io::Error) -> Self {
        VerumError::IoError(error)
    }
} 