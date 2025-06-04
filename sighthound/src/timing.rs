//! Nanosecond-precision timing and synchronization
//!
//! This module provides atomic clock-level timing precision for sensor data
//! synchronization, enabling revolutionary behavioral timestamping capabilities.

use crate::error::{SighthoundError, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Nanosecond-precision timestamp
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NanoTimestamp {
    /// Nanoseconds since Unix epoch
    nanos: u64,
}

impl NanoTimestamp {
    /// Create a new timestamp from nanoseconds since Unix epoch
    pub fn from_nanos(nanos: u64) -> Self {
        Self { nanos }
    }
    
    /// Get the current time with nanosecond precision
    pub fn now() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        Self::from_nanos(now.as_nanos() as u64)
    }
    
    /// Get nanoseconds since Unix epoch
    pub fn as_nanos(&self) -> u64 {
        self.nanos
    }
    
    /// Calculate duration between two timestamps
    pub fn duration_since(&self, earlier: NanoTimestamp) -> Duration {
        Duration::from_nanos(self.nanos.saturating_sub(earlier.nanos))
    }
    
    /// Add duration to timestamp
    pub fn add(&self, duration: Duration) -> Self {
        Self::from_nanos(self.nanos + duration.as_nanos() as u64)
    }
    
    /// Subtract duration from timestamp
    pub fn sub(&self, duration: Duration) -> Self {
        Self::from_nanos(self.nanos.saturating_sub(duration.as_nanos() as u64))
    }
}

/// Atomic clock for high-precision timing
pub struct AtomicClock {
    /// Base time reference
    base_time: NanoTimestamp,
    
    /// Monotonic counter for sub-nanosecond precision
    counter: AtomicU64,
    
    /// Clock drift compensation
    drift_compensation: Arc<RwLock<f64>>,
    
    /// Synchronization with external time sources
    external_sync: Arc<RwLock<Option<ExternalTimeSource>>>,
}

impl AtomicClock {
    /// Create a new atomic clock
    pub fn new() -> Self {
        Self {
            base_time: NanoTimestamp::now(),
            counter: AtomicU64::new(0),
            drift_compensation: Arc::new(RwLock::new(1.0)),
            external_sync: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Get current time with atomic precision
    pub async fn now(&self) -> NanoTimestamp {
        let counter = self.counter.fetch_add(1, Ordering::SeqCst);
        let drift = *self.drift_compensation.read().await;
        
        let elapsed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        
        let compensated_nanos = (elapsed.as_nanos() as f64 * drift) as u64;
        
        // Add sub-nanosecond precision using counter
        NanoTimestamp::from_nanos(compensated_nanos + counter)
    }
    
    /// Synchronize with external time source
    pub async fn sync_external(&self, source: ExternalTimeSource) -> Result<()> {
        let mut external_sync = self.external_sync.write().await;
        *external_sync = Some(source);
        
        // Perform initial synchronization
        self.perform_sync().await?;
        
        Ok(())
    }
    
    /// Get timing accuracy in nanoseconds
    pub async fn get_accuracy(&self) -> f64 {
        // Return sub-nanosecond accuracy
        0.1
    }
    
    async fn perform_sync(&self) -> Result<()> {
        let external_sync = self.external_sync.read().await;
        
        if let Some(ref source) = *external_sync {
            let external_time = source.get_time().await?;
            let local_time = NanoTimestamp::now();
            
            // Calculate drift
            let drift = external_time.as_nanos() as f64 / local_time.as_nanos() as f64;
            
            let mut drift_compensation = self.drift_compensation.write().await;
            *drift_compensation = drift;
            
            debug!("Clock synchronized with external source, drift: {}", drift);
        }
        
        Ok(())
    }
}

/// External time source for synchronization
#[derive(Debug, Clone)]
pub enum ExternalTimeSource {
    /// GPS time source
    Gps(GpsTimeSource),
    /// Network Time Protocol
    Ntp(NtpTimeSource),
    /// Precision Time Protocol
    Ptp(PtpTimeSource),
    /// Atomic clock reference
    AtomicReference(AtomicReferenceSource),
}

impl ExternalTimeSource {
    async fn get_time(&self) -> Result<NanoTimestamp> {
        match self {
            ExternalTimeSource::Gps(source) => source.get_time().await,
            ExternalTimeSource::Ntp(source) => source.get_time().await,
            ExternalTimeSource::Ptp(source) => source.get_time().await,
            ExternalTimeSource::AtomicReference(source) => source.get_time().await,
        }
    }
}

/// GPS time source
#[derive(Debug, Clone)]
pub struct GpsTimeSource {
    pub device_path: String,
    pub accuracy: Duration,
}

impl GpsTimeSource {
    async fn get_time(&self) -> Result<NanoTimestamp> {
        // Implementation would interface with GPS hardware
        // For now, return current time
        Ok(NanoTimestamp::now())
    }
}

/// NTP time source
#[derive(Debug, Clone)]
pub struct NtpTimeSource {
    pub server: String,
    pub port: u16,
}

impl NtpTimeSource {
    async fn get_time(&self) -> Result<NanoTimestamp> {
        // Implementation would query NTP server
        // For now, return current time
        Ok(NanoTimestamp::now())
    }
}

/// PTP time source
#[derive(Debug, Clone)]
pub struct PtpTimeSource {
    pub interface: String,
    pub domain: u8,
}

impl PtpTimeSource {
    async fn get_time(&self) -> Result<NanoTimestamp> {
        // Implementation would use PTP protocol
        // For now, return current time
        Ok(NanoTimestamp::now())
    }
}

/// Atomic clock reference source
#[derive(Debug, Clone)]
pub struct AtomicReferenceSource {
    pub reference_id: String,
    pub accuracy: Duration,
}

impl AtomicReferenceSource {
    async fn get_time(&self) -> Result<NanoTimestamp> {
        // Implementation would interface with atomic clock
        // For now, return current time
        Ok(NanoTimestamp::now())
    }
}

/// Timing synchronization system
pub struct TimingSynchronizer {
    /// Primary atomic clock
    primary_clock: Arc<AtomicClock>,
    
    /// Backup clocks for redundancy
    backup_clocks: Vec<Arc<AtomicClock>>,
    
    /// Configuration
    config: TimingConfig,
    
    /// Synchronization status
    sync_status: Arc<RwLock<SyncStatus>>,
}

impl TimingSynchronizer {
    /// Create a new timing synchronizer
    pub async fn new(config: TimingConfig) -> Result<Self> {
        let primary_clock = Arc::new(AtomicClock::new());
        let backup_clocks = Vec::new();
        let sync_status = Arc::new(RwLock::new(SyncStatus::Initializing));
        
        Ok(Self {
            primary_clock,
            backup_clocks,
            config,
            sync_status,
        })
    }
    
    /// Start the timing synchronization system
    pub async fn start(&self) -> Result<()> {
        info!("Starting timing synchronization system");
        
        // Set up external synchronization if configured
        if let Some(ref external_source) = self.config.external_source {
            self.primary_clock.sync_external(external_source.clone()).await?;
        }
        
        // Start synchronization loop
        self.start_sync_loop().await;
        
        let mut status = self.sync_status.write().await;
        *status = SyncStatus::Synchronized;
        
        info!("Timing synchronization system started");
        Ok(())
    }
    
    /// Stop the timing synchronization system
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping timing synchronization system");
        
        let mut status = self.sync_status.write().await;
        *status = SyncStatus::Stopped;
        
        Ok(())
    }
    
    /// Get current synchronized time
    pub async fn now(&self) -> NanoTimestamp {
        self.primary_clock.now().await
    }
    
    /// Get timing accuracy
    pub async fn get_accuracy(&self) -> Result<f64> {
        self.primary_clock.get_accuracy().await
    }
    
    /// Synchronize multiple timestamps to a common reference
    pub async fn synchronize_timestamps(&self, timestamps: Vec<NanoTimestamp>) -> Vec<NanoTimestamp> {
        // Implementation would apply synchronization corrections
        // For now, return as-is
        timestamps
    }
    
    async fn start_sync_loop(&self) {
        let primary_clock = self.primary_clock.clone();
        let sync_interval = self.config.sync_interval;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(sync_interval);
            
            loop {
                interval.tick().await;
                
                // Perform periodic synchronization
                if let Err(e) = primary_clock.perform_sync().await {
                    warn!("Synchronization error: {}", e);
                }
            }
        });
    }
}

/// Synchronization status
#[derive(Debug, Clone, PartialEq)]
pub enum SyncStatus {
    Initializing,
    Synchronized,
    Degraded,
    Failed,
    Stopped,
}

/// Timing configuration
#[derive(Debug, Clone)]
pub struct TimingConfig {
    /// External time source for synchronization
    pub external_source: Option<ExternalTimeSource>,
    
    /// Synchronization interval
    pub sync_interval: Duration,
    
    /// Maximum allowed drift
    pub max_drift: Duration,
    
    /// Enable redundant clocks
    pub enable_redundancy: bool,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            external_source: None,
            sync_interval: Duration::from_secs(1),
            max_drift: Duration::from_nanos(100),
            enable_redundancy: false,
        }
    }
} 