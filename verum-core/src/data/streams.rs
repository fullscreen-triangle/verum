//! Data streaming and real-time processing systems

use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use serde::{Deserialize, Serialize};

/// Configuration for data streaming systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    pub buffer_size: usize,
    pub batch_size: usize,
    pub processing_interval_ms: u64,
    pub max_backpressure: usize,
}

/// Generic data stream processor
pub struct DataStream<T> {
    config: StreamConfig,
    sender: mpsc::Sender<T>,
    receiver: Arc<RwLock<mpsc::Receiver<T>>>,
    processors: Vec<Box<dyn StreamProcessor<T> + Send + Sync>>,
}

impl<T: Send + 'static> DataStream<T> {
    pub fn new(config: StreamConfig) -> Self {
        let (sender, receiver) = mpsc::channel(config.buffer_size);
        
        Self {
            config,
            sender,
            receiver: Arc::new(RwLock::new(receiver)),
            processors: Vec::new(),
        }
    }
    
    pub fn get_sender(&self) -> mpsc::Sender<T> {
        self.sender.clone()
    }
    
    pub fn add_processor(&mut self, processor: Box<dyn StreamProcessor<T> + Send + Sync>) {
        self.processors.push(processor);
    }
    
    pub async fn start_processing(&self) -> Result<(), StreamError> {
        let receiver = self.receiver.clone();
        let batch_size = self.config.batch_size;
        
        tokio::spawn(async move {
            let mut batch = Vec::with_capacity(batch_size);
            
            loop {
                let mut rx = receiver.write().await;
                
                // Collect batch
                for _ in 0..batch_size {
                    match rx.try_recv() {
                        Ok(item) => batch.push(item),
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => return,
                    }
                }
                
                if !batch.is_empty() {
                    // Process batch would go here
                    batch.clear();
                }
                
                // Small delay to prevent busy waiting
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        });
        
        Ok(())
    }
}

/// Stream processor trait for data transformation
pub trait StreamProcessor<T> {
    fn process(&self, data: &T) -> Result<Option<T>, StreamError>;
    fn process_batch(&self, batch: &[T]) -> Result<Vec<T>, StreamError>;
}

/// Real-time data aggregator
pub struct DataAggregator<T> {
    window_size: usize,
    current_window: Vec<T>,
    aggregation_fn: Box<dyn Fn(&[T]) -> T + Send + Sync>,
}

impl<T: Clone> DataAggregator<T> {
    pub fn new(window_size: usize, aggregation_fn: Box<dyn Fn(&[T]) -> T + Send + Sync>) -> Self {
        Self {
            window_size,
            current_window: Vec::with_capacity(window_size),
            aggregation_fn,
        }
    }
    
    pub fn add_sample(&mut self, sample: T) -> Option<T> {
        self.current_window.push(sample);
        
        if self.current_window.len() >= self.window_size {
            let result = (self.aggregation_fn)(&self.current_window);
            self.current_window.clear();
            Some(result)
        } else {
            None
        }
    }
}

/// Stream multiplexer for handling multiple data sources
pub struct StreamMultiplexer<T> {
    streams: Vec<mpsc::Receiver<T>>,
    output: mpsc::Sender<T>,
}

impl<T: Send + 'static> StreamMultiplexer<T> {
    pub fn new(output: mpsc::Sender<T>) -> Self {
        Self {
            streams: Vec::new(),
            output,
        }
    }
    
    pub fn add_stream(&mut self, stream: mpsc::Receiver<T>) {
        self.streams.push(stream);
    }
    
    pub async fn start_multiplexing(&mut self) -> Result<(), StreamError> {
        // This would normally use select! to multiplex streams
        // Simplified implementation for now
        Ok(())
    }
}

/// Stream error types
#[derive(Debug, Clone)]
pub enum StreamError {
    ChannelClosed,
    BufferFull,
    ProcessingError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::ChannelClosed => write!(f, "Stream channel closed"),
            StreamError::BufferFull => write!(f, "Stream buffer full"),
            StreamError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
            StreamError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for StreamError {}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1000,
            batch_size: 100,
            processing_interval_ms: 100,
            max_backpressure: 5000,
        }
    }
} 