//! Common types for Sighthound sensor fusion system

use crate::timing::NanoTimestamp;
use serde::{Deserialize, Serialize};

/// Sensor type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensorType {
    Lidar,
    Camera,
    Radar,
    Imu,
    Gps,
    Ultrasonic,
    Temperature,
    Pressure,
}

/// Generic sensor data wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorData {
    pub sensor_type: SensorType,
    pub timestamp: NanoTimestamp,
    pub data: SensorDataType,
    pub quality: DataQuality,
}

/// Sensor data type variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorDataType {
    PointCloud(Vec<Point3D>),
    Image(ImageData),
    RadarReflection(RadarData),
    Inertial(InertialData),
    Position(PositionData),
    Distance(f64),
    Temperature(f64),
    Pressure(f64),
}

/// 3D point data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub intensity: Option<f64>,
}

/// Image data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub data: Vec<u8>,
}

/// Image format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageFormat {
    Rgb8,
    Rgba8,
    Gray8,
    Gray16,
}

/// Radar data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadarData {
    pub range: f64,
    pub velocity: f64,
    pub angle: f64,
    pub intensity: f64,
}

/// Inertial measurement data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InertialData {
    pub acceleration: Vector3D,
    pub angular_velocity: Vector3D,
    pub magnetic_field: Option<Vector3D>,
}

/// Position data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionData {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
    pub accuracy: f64,
}

/// 3D vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Data quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuality {
    pub confidence: f64,
    pub noise_level: f64,
    pub completeness: f64,
}

/// Environmental context data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalData {
    pub weather_conditions: WeatherConditions,
    pub lighting_conditions: LightingConditions,
    pub road_conditions: RoadConditions,
}

/// Weather conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeatherConditions {
    pub temperature: f64,
    pub humidity: f64,
    pub precipitation: PrecipitationType,
    pub wind_speed: f64,
    pub visibility: f64,
}

/// Precipitation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecipitationType {
    None,
    Rain,
    Snow,
    Sleet,
    Hail,
}

/// Lighting conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightingConditions {
    pub ambient_light: f64,
    pub sun_angle: Option<f64>,
    pub artificial_lighting: bool,
}

/// Road conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadConditions {
    pub surface_type: RoadSurfaceType,
    pub condition: RoadCondition,
    pub grip_coefficient: f64,
}

/// Road surface type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoadSurfaceType {
    Asphalt,
    Concrete,
    Gravel,
    Dirt,
    Snow,
    Ice,
}

/// Road condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoadCondition {
    Dry,
    Wet,
    Icy,
    Snowy,
    Muddy,
} 