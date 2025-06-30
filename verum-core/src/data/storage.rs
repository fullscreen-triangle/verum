//! Data storage and persistence systems

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Configuration for data storage systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub max_buffer_size: usize,
    pub persistence_interval_ms: u64,
    pub compression_enabled: bool,
}

/// In-memory data buffer for high-performance access
#[derive(Debug, Clone)]
pub struct DataBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    write_index: usize,
}

impl<T: Clone> DataBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
            write_index: 0,
        }
    }
    
    pub fn push(&mut self, item: T) {
        if self.data.len() < self.capacity {
            self.data.push(item);
        } else {
            self.data[self.write_index] = item;
            self.write_index = (self.write_index + 1) % self.capacity;
        }
    }
    
    pub fn get_recent(&self, count: usize) -> Vec<&T> {
        let take_count = count.min(self.data.len());
        if self.data.len() < self.capacity {
            self.data.iter().rev().take(take_count).collect()
        } else {
            let mut result = Vec::new();
            let mut index = if self.write_index == 0 { self.capacity - 1 } else { self.write_index - 1 };
            for _ in 0..take_count {
                result.push(&self.data[index]);
                index = if index == 0 { self.capacity - 1 } else { index - 1 };
            }
            result
        }
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

/// Persistent storage interface
pub trait PersistentStorage {
    type Item;
    
    fn store(&mut self, key: &str, item: &Self::Item) -> Result<(), StorageError>;
    fn retrieve(&self, key: &str) -> Result<Option<Self::Item>, StorageError>;
    fn delete(&mut self, key: &str) -> Result<(), StorageError>;
    fn list_keys(&self) -> Result<Vec<String>, StorageError>;
}

/// In-memory storage implementation
#[derive(Debug, Clone)]
pub struct MemoryStorage<T> {
    data: HashMap<String, T>,
}

impl<T: Clone> MemoryStorage<T> {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl<T: Clone> PersistentStorage for MemoryStorage<T> {
    type Item = T;
    
    fn store(&mut self, key: &str, item: &Self::Item) -> Result<(), StorageError> {
        self.data.insert(key.to_string(), item.clone());
        Ok(())
    }
    
    fn retrieve(&self, key: &str) -> Result<Option<Self::Item>, StorageError> {
        Ok(self.data.get(key).cloned())
    }
    
    fn delete(&mut self, key: &str) -> Result<(), StorageError> {
        self.data.remove(key);
        Ok(())
    }
    
    fn list_keys(&self) -> Result<Vec<String>, StorageError> {
        Ok(self.data.keys().cloned().collect())
    }
}

#[derive(Debug, Clone)]
pub enum StorageError {
    KeyNotFound(String),
    SerializationError(String),
    IoError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::KeyNotFound(key) => write!(f, "Key not found: {}", key),
            StorageError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            StorageError::IoError(msg) => write!(f, "IO error: {}", msg),
            StorageError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for StorageError {}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10000,
            persistence_interval_ms: 1000,
            compression_enabled: true,
        }
    }
} 