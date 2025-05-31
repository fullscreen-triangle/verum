//! # Utilities Module
//!
//! Common utilities, error handling, and configuration management for Verum

pub mod config;
pub mod error;
pub mod logging;

// Re-exports
pub use config::Config;
pub use error::{Result, VerumError};

use serde::{Deserialize, Serialize};

/// Common position type used throughout the system
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f32>,
}

impl Position {
    pub fn new(lat: f64, lon: f64) -> Self {
        Self {
            latitude: lat,
            longitude: lon,
            altitude: None,
        }
    }
    
    pub fn with_altitude(lat: f64, lon: f64, alt: f32) -> Self {
        Self {
            latitude: lat,
            longitude: lon,
            altitude: Some(alt),
        }
    }
    
    /// Calculate distance to another position in meters
    pub fn distance_to(&self, other: &Position) -> f64 {
        let earth_radius = 6371000.0; // Earth radius in meters
        
        let lat1_rad = self.latitude.to_radians();
        let lat2_rad = other.latitude.to_radians();
        let delta_lat = (other.latitude - self.latitude).to_radians();
        let delta_lon = (other.longitude - self.longitude).to_radians();
        
        let a = (delta_lat / 2.0).sin().powi(2) +
                lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
        
        earth_radius * c
    }
}

/// Velocity vector
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Velocity {
    pub x: f32, // m/s in forward direction
    pub y: f32, // m/s in lateral direction
    pub z: f32, // m/s in vertical direction
}

impl Velocity {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    
    pub fn magnitude(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
    
    pub fn speed_2d(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

/// Acceleration vector
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Acceleration {
    pub x: f32, // m/s² in forward direction
    pub y: f32, // m/s² in lateral direction
    pub z: f32, // m/s² in vertical direction
}

impl Acceleration {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    
    pub fn magnitude(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
}

/// Time utilities
pub mod time {
    use chrono::{DateTime, Utc};
    
    pub fn now() -> DateTime<Utc> {
        Utc::now()
    }
    
    pub fn timestamp_ms() -> u64 {
        now().timestamp_millis() as u64
    }
    
    pub fn timestamp_us() -> u64 {
        now().timestamp_micros() as u64
    }
}

/// Mathematical utilities
pub mod math {
    /// Clamp a value between min and max
    pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }
    
    /// Linear interpolation
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }
    
    /// Normalize angle to [-π, π]
    pub fn normalize_angle(angle: f32) -> f32 {
        let mut normalized = angle;
        while normalized > std::f32::consts::PI {
            normalized -= 2.0 * std::f32::consts::PI;
        }
        while normalized < -std::f32::consts::PI {
            normalized += 2.0 * std::f32::consts::PI;
        }
        normalized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_position_distance() {
        let pos1 = Position::new(52.520008, 13.404954); // Berlin
        let pos2 = Position::new(48.856614, 2.352222);   // Paris
        
        let distance = pos1.distance_to(&pos2);
        // Approximate distance Berlin to Paris
        assert!((distance - 878000.0).abs() < 10000.0);
    }
    
    #[test]
    fn test_velocity_magnitude() {
        let vel = Velocity::new(3.0, 4.0, 0.0);
        assert_eq!(vel.magnitude(), 5.0);
    }
    
    #[test]
    fn test_math_clamp() {
        assert_eq!(math::clamp(5, 0, 10), 5);
        assert_eq!(math::clamp(-5, 0, 10), 0);
        assert_eq!(math::clamp(15, 0, 10), 10);
    }
} 