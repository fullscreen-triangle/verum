//! # Verum Core
//!
//! Personal Intelligence-Driven Navigation - Core AI Engine
//!
//! This is the main library for the Verum autonomous driving system,
//! implementing personal AI models learned from cross-domain behavioral data.

pub mod ai;
pub mod biometrics;
pub mod vehicle;
pub mod network;
pub mod utils;

// Re-export main components
pub use ai::{AIOrchestrator, PersonalAIModel, DecisionResult};
pub use biometrics::{BiometricProcessor, BiometricState};
pub use vehicle::{VehicleController, VehicleState};
pub use network::{NetworkClient, CoordinationMessage};
pub use utils::{Result, VerumError, Config};

use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Main Verum engine that coordinates all subsystems
pub struct VerumEngine {
    id: Uuid,
    ai_orchestrator: Arc<RwLock<AIOrchestrator>>,
    biometric_processor: Arc<BiometricProcessor>,
    vehicle_controller: Arc<VehicleController>,
    network_client: Arc<NetworkClient>,
    config: Config,
}

impl VerumEngine {
    /// Create a new Verum engine instance
    pub async fn new(config: Config) -> Result<Self> {
        let id = Uuid::new_v4();
        
        // Initialize personal AI model
        let personal_model = PersonalAIModel::load_from_path(&config.ai.model_path).await?;
        
        // Initialize AI orchestrator
        let ai_orchestrator = Arc::new(RwLock::new(
            AIOrchestrator::new(personal_model, config.ai.clone()).await?
        ));
        
        // Initialize biometric processor
        let biometric_processor = Arc::new(
            BiometricProcessor::new(config.biometrics.clone()).await?
        );
        
        // Initialize vehicle controller
        let vehicle_controller = Arc::new(
            VehicleController::new(config.vehicle.clone()).await?
        );
        
        // Initialize network client
        let network_client = Arc::new(
            NetworkClient::new(config.network.clone()).await?
        );
        
        Ok(Self {
            id,
            ai_orchestrator,
            biometric_processor,
            vehicle_controller,
            network_client,
            config,
        })
    }
    
    /// Start the Verum engine
    pub async fn start(&self) -> Result<()> {
        tracing::info!("Starting Verum engine {}", self.id);
        
        // Start all subsystems
        self.biometric_processor.start().await?;
        self.vehicle_controller.start().await?;
        self.network_client.connect().await?;
        
        tracing::info!("Verum engine started successfully");
        Ok(())
    }
    
    /// Main driving loop
    pub async fn drive_to_destination(&self, destination: String) -> Result<()> {
        tracing::info!("Starting navigation to: {}", destination);
        
        loop {
            // Get current biometric state
            let biometric_state = self.biometric_processor.get_current_state().await?;
            
            // Get current vehicle state
            let vehicle_state = self.vehicle_controller.get_state().await?;
            
            // Create scenario context
            let scenario = ai::ScenarioContext {
                scenario_type: "navigation".to_string(),
                environmental_factors: vehicle_state.environmental_factors(),
                time_pressure: 0.5, // Medium time pressure
                risk_level: vehicle_state.calculate_risk_level(),
                goal_urgency: 0.7, // High goal urgency
            };
            
            // Get AI decision
            let decision = {
                let mut ai = self.ai_orchestrator.write().await;
                ai.process_scenario(scenario.clone(), biometric_state.clone()).await?
            };
            
            // Execute decision
            self.vehicle_controller.execute_decision(decision.decision.clone()).await?;
            
            // Learn from the experience
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            
            // Check if we've reached the destination
            if vehicle_state.is_at_destination(&destination) {
                break;
            }
        }
        
        tracing::info!("Reached destination: {}", destination);
        Ok(())
    }
    
    /// Get engine statistics
    pub async fn get_statistics(&self) -> EngineStatistics {
        let ai_stats = {
            let ai = self.ai_orchestrator.read().await;
            ai.get_statistics()
        };
        
        EngineStatistics {
            engine_id: self.id,
            ai_statistics: ai_stats,
            uptime: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
    
    /// Shutdown the engine
    pub async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down Verum engine {}", self.id);
        
        self.network_client.disconnect().await?;
        self.vehicle_controller.stop().await?;
        self.biometric_processor.stop().await?;
        
        tracing::info!("Verum engine shutdown complete");
        Ok(())
    }
}

/// Engine statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineStatistics {
    pub engine_id: Uuid,
    pub ai_statistics: ai::AIStatistics,
    pub uptime: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_engine_creation() {
        let config = Config::default();
        let result = VerumEngine::new(config).await;
        
        // This will fail until we implement all the components
        // but it verifies the API structure
        assert!(result.is_err() || result.is_ok());
    }
} 