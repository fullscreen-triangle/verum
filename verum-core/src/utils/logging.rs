//! Logging utilities

use crate::utils::Result;

/// Setup logging for the application
pub fn setup_logging() -> Result<()> {
    tracing_subscriber::fmt::init();
    Ok(())
} 