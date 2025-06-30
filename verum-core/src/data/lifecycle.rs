//! Data lifecycle management and retention policies

use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Configuration for data lifecycle management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleConfig {
    pub retention_policies: Vec<RetentionPolicy>,
    pub cleanup_interval_ms: u64,
    pub archive_threshold_mb: u64,
    pub compression_enabled: bool,
}

/// Data retention policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub data_type: String,
    pub max_age_hours: u64,
    pub max_size_mb: u64,
    pub priority: u8,
    pub archive_after_hours: Option<u64>,
}

/// Data lifecycle manager
pub struct LifecycleManager {
    config: LifecycleConfig,
    tracked_data: HashMap<String, DataEntry>,
    last_cleanup: Instant,
}

impl LifecycleManager {
    pub fn new(config: LifecycleConfig) -> Self {
        Self {
            config,
            tracked_data: HashMap::new(),
            last_cleanup: Instant::now(),
        }
    }
    
    /// Register data for lifecycle tracking
    pub fn register_data(&mut self, key: String, data_type: String, size_bytes: u64) {
        let entry = DataEntry {
            data_type,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            size_bytes,
            access_count: 1,
            archived: false,
        };
        
        self.tracked_data.insert(key, entry);
    }
    
    /// Update access time for data
    pub fn mark_accessed(&mut self, key: &str) {
        if let Some(entry) = self.tracked_data.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
        }
    }
    
    /// Cleanup expired data based on retention policies
    pub fn cleanup_expired_data(&mut self) -> Vec<String> {
        let now = Instant::now();
        let cleanup_interval = Duration::from_millis(self.config.cleanup_interval_ms);
        
        if now.duration_since(self.last_cleanup) < cleanup_interval {
            return Vec::new();
        }
        
        let mut expired_keys = Vec::new();
        
        for (key, entry) in &self.tracked_data {
            if let Some(policy) = self.get_policy_for_type(&entry.data_type) {
                let max_age = Duration::from_secs(policy.max_age_hours * 3600);
                
                if now.duration_since(entry.created_at) > max_age {
                    expired_keys.push(key.clone());
                }
            }
        }
        
        // Remove expired entries
        for key in &expired_keys {
            self.tracked_data.remove(key);
        }
        
        self.last_cleanup = now;
        expired_keys
    }
    
    /// Get data eligible for archiving
    pub fn get_archival_candidates(&self) -> Vec<String> {
        let now = Instant::now();
        let mut candidates = Vec::new();
        
        for (key, entry) in &self.tracked_data {
            if entry.archived {
                continue;
            }
            
            if let Some(policy) = self.get_policy_for_type(&entry.data_type) {
                if let Some(archive_hours) = policy.archive_after_hours {
                    let archive_threshold = Duration::from_secs(archive_hours * 3600);
                    
                    if now.duration_since(entry.created_at) > archive_threshold {
                        candidates.push(key.clone());
                    }
                }
            }
        }
        
        candidates
    }
    
    /// Mark data as archived
    pub fn mark_archived(&mut self, key: &str) {
        if let Some(entry) = self.tracked_data.get_mut(key) {
            entry.archived = true;
        }
    }
    
    /// Get total size of tracked data
    pub fn get_total_size(&self) -> u64 {
        self.tracked_data.values().map(|e| e.size_bytes).sum()
    }
    
    /// Get data statistics
    pub fn get_statistics(&self) -> LifecycleStatistics {
        let total_entries = self.tracked_data.len();
        let total_size = self.get_total_size();
        let archived_count = self.tracked_data.values().filter(|e| e.archived).count();
        
        let mut type_counts = HashMap::new();
        for entry in self.tracked_data.values() {
            *type_counts.entry(entry.data_type.clone()).or_insert(0) += 1;
        }
        
        LifecycleStatistics {
            total_entries,
            total_size_bytes: total_size,
            archived_entries: archived_count,
            type_distribution: type_counts,
        }
    }
    
    fn get_policy_for_type(&self, data_type: &str) -> Option<&RetentionPolicy> {
        self.config.retention_policies
            .iter()
            .find(|p| p.data_type == data_type)
    }
}

/// Tracked data entry
#[derive(Debug, Clone)]
struct DataEntry {
    data_type: String,
    created_at: Instant,
    last_accessed: Instant,
    size_bytes: u64,
    access_count: u64,
    archived: bool,
}

/// Lifecycle statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleStatistics {
    pub total_entries: usize,
    pub total_size_bytes: u64,
    pub archived_entries: usize,
    pub type_distribution: HashMap<String, usize>,
}

/// Data archiver for long-term storage
pub struct DataArchiver {
    config: ArchiveConfig,
}

impl DataArchiver {
    pub fn new(config: ArchiveConfig) -> Self {
        Self { config }
    }
    
    /// Archive data to long-term storage
    pub async fn archive_data(&self, key: &str, _data: &[u8]) -> Result<(), ArchiveError> {
        // Implementation would depend on specific archive backend
        // For now, just simulate archiving
        println!("Archiving data: {}", key);
        Ok(())
    }
    
    /// Retrieve archived data
    pub async fn retrieve_archived_data(&self, key: &str) -> Result<Vec<u8>, ArchiveError> {
        // Implementation would depend on specific archive backend
        Err(ArchiveError::NotFound(key.to_string()))
    }
}

/// Archive configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveConfig {
    pub storage_path: String,
    pub compression_level: u8,
    pub encryption_enabled: bool,
}

/// Archive error types
#[derive(Debug, Clone)]
pub enum ArchiveError {
    NotFound(String),
    StorageError(String),
    CompressionError(String),
    EncryptionError(String),
}

impl std::fmt::Display for ArchiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchiveError::NotFound(key) => write!(f, "Archived data not found: {}", key),
            ArchiveError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            ArchiveError::CompressionError(msg) => write!(f, "Compression error: {}", msg),
            ArchiveError::EncryptionError(msg) => write!(f, "Encryption error: {}", msg),
        }
    }
}

impl std::error::Error for ArchiveError {}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            retention_policies: vec![
                RetentionPolicy {
                    data_type: "oscillation_data".to_string(),
                    max_age_hours: 24,
                    max_size_mb: 100,
                    priority: 1,
                    archive_after_hours: Some(6),
                },
                RetentionPolicy {
                    data_type: "entropy_states".to_string(),
                    max_age_hours: 168, // 1 week
                    max_size_mb: 50,
                    priority: 2,
                    archive_after_hours: Some(24),
                },
            ],
            cleanup_interval_ms: 300000, // 5 minutes
            archive_threshold_mb: 500,
            compression_enabled: true,
        }
    }
}

impl Default for ArchiveConfig {
    fn default() -> Self {
        Self {
            storage_path: "./archive".to_string(),
            compression_level: 6,
            encryption_enabled: false,
        }
    }
} 