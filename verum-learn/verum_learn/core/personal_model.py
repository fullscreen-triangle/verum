"""
Personal Model Training System

Trains personalized AI models from 5+ years of cross-domain behavioral data.
This creates AI that drives exactly like the individual person would, based on
their patterns learned from driving, walking, tennis, cycling, and daily activities.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class PersonalDataPoint:
    """Single data point from personal behavioral history."""
    timestamp: datetime
    domain: str
    scenario: str
    context_vector: np.ndarray
    action_vector: np.ndarray
    biometric_state: np.ndarray
    outcome: float  # Success/performance metric
    stress_level: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainProfile:
    """Profile for a specific domain of activity."""
    domain_name: str
    total_hours: float
    data_points: int
    skill_level: float  # 0.0 to 1.0
    consistency: float  # How consistent the person is
    stress_tolerance: float
    preferred_patterns: List[str]
    risk_preference: float
    adaptation_speed: float
    dominant_strategies: Dict[str, float]


@dataclass
class PersonalityProfile:
    """Overall personality profile derived from cross-domain analysis."""
    risk_tolerance: float
    stress_sensitivity: float
    reaction_speed: float
    spatial_awareness: float
    pattern_consistency: float
    learning_adaptability: float
    social_coordination: float  # How well they work with others
    planning_horizon: float  # Short-term vs long-term thinking
    decision_confidence: float
    emotional_regulation: float


class PersonalDataset(Dataset):
    """PyTorch dataset for personal behavioral data."""
    
    def __init__(self, data_points: List[PersonalDataPoint], sequence_length: int = 32):
        self.data_points = data_points
        self.sequence_length = sequence_length
        
        # Group data points by domain and scenario for sequence creation
        self.sequences = self._create_sequences()
        
        # Normalize data
        self.scaler_context = StandardScaler()
        self.scaler_action = StandardScaler()
        self.scaler_biometric = StandardScaler()
        
        self._fit_scalers()
    
    def _create_sequences(self) -> List[Tuple[List[PersonalDataPoint], PersonalDataPoint]]:
        """Create sequences of data points for temporal learning."""
        sequences = []
        
        # Sort by timestamp
        sorted_points = sorted(self.data_points, key=lambda x: x.timestamp)
        
        # Create sliding windows
        for i in range(len(sorted_points) - self.sequence_length):
            sequence = sorted_points[i:i + self.sequence_length]
            target = sorted_points[i + self.sequence_length]
            
            # Only create sequence if within reasonable time window (e.g., same day)
            if (target.timestamp - sequence[0].timestamp).total_seconds() < 86400:  # 24 hours
                sequences.append((sequence, target))
        
        return sequences
    
    def _fit_scalers(self):
        """Fit scalers to the data."""
        contexts = np.array([dp.context_vector for dp in self.data_points])
        actions = np.array([dp.action_vector for dp in self.data_points])
        biometrics = np.array([dp.biometric_state for dp in self.data_points])
        
        self.scaler_context.fit(contexts)
        self.scaler_action.fit(actions)
        self.scaler_biometric.fit(biometrics)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence, target = self.sequences[idx]
        
        # Convert sequence to tensors
        seq_contexts = np.array([dp.context_vector for dp in sequence])
        seq_actions = np.array([dp.action_vector for dp in sequence])
        seq_biometrics = np.array([dp.biometric_state for dp in sequence])
        
        # Normalize
        seq_contexts = self.scaler_context.transform(seq_contexts)
        seq_actions = self.scaler_action.transform(seq_actions)
        seq_biometrics = self.scaler_biometric.transform(seq_biometrics)
        
        # Target
        target_context = self.scaler_context.transform(target.context_vector.reshape(1, -1))
        target_action = self.scaler_action.transform(target.action_vector.reshape(1, -1))
        
        # Combine sequence features
        sequence_tensor = torch.FloatTensor(
            np.concatenate([seq_contexts, seq_biometrics], axis=1)
        )
        
        # Input: current context + sequence history
        input_tensor = torch.cat([
            torch.FloatTensor(target_context),
            sequence_tensor.mean(dim=0, keepdim=True)  # Summarize sequence
        ], dim=1)
        
        # Target: action to take
        target_tensor = torch.FloatTensor(target_action)
        
        # Additional info
        info_tensor = torch.FloatTensor([
            target.stress_level,
            target.confidence,
            target.outcome
        ])
        
        return input_tensor.squeeze(0), target_tensor.squeeze(0), info_tensor


class PersonalAINetwork(nn.Module):
    """Neural network that learns personal behavioral patterns."""
    
    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        biometric_dim: int,
        hidden_dim: int = 512,
        num_domains: int = 7,
        sequence_length: int = 32
    ):
        super().__init__()
        
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.biometric_dim = biometric_dim
        self.hidden_dim = hidden_dim
        
        # Input processing
        input_dim = context_dim + biometric_dim + context_dim  # context + biometric summary + current context
        
        # Domain embedding
        self.domain_embedding = nn.Embedding(num_domains, 64)
        
        # Main processing network
        self.input_processor = nn.Sequential(
            nn.Linear(input_dim + 64, hidden_dim),  # +64 for domain embedding
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Sequence processing (LSTM for temporal patterns)
        self.sequence_lstm = nn.LSTM(
            context_dim + biometric_dim,
            hidden_dim // 2,
            batch_first=True,
            num_layers=2,
            dropout=0.2
        )
        
        # Attention mechanism for focusing on relevant past experiences
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Decision networks
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Actions are normalized
        )
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Stress prediction
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        context: torch.Tensor,
        sequence: torch.Tensor,
        domain_id: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size = context.size(0)
        
        # Domain embedding
        domain_emb = self.domain_embedding(domain_id)
        
        # Process current context
        current_input = torch.cat([context, domain_emb], dim=1)
        current_features = self.input_processor(current_input)
        
        # Process sequence if provided
        if sequence.size(1) > 0:
            sequence_output, _ = self.sequence_lstm(sequence)
            sequence_features = sequence_output[:, -1, :]  # Take last output
            
            # Attention between current and sequence
            combined_features = torch.cat([
                current_features.unsqueeze(1),
                sequence_features.unsqueeze(1)
            ], dim=1)
            
            attended_features, _ = self.attention(
                combined_features, combined_features, combined_features
            )
            
            final_features = attended_features.mean(dim=1)
        else:
            final_features = current_features
        
        # Generate outputs
        action = self.action_head(final_features)
        confidence = self.confidence_head(final_features)
        stress = self.stress_head(final_features)
        
        return {
            'action': action,
            'confidence': confidence,
            'stress': stress,
            'features': final_features
        }


class PersonalModelTrainer:
    """Trains personalized AI models from multi-domain behavioral data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Data storage
        self.training_data: List[PersonalDataPoint] = []
        self.validation_data: List[PersonalDataPoint] = []
        
        # Domain profiles
        self.domain_profiles: Dict[str, DomainProfile] = {}
        
        # Overall personality profile
        self.personality_profile: Optional[PersonalityProfile] = None
        
        # Model components
        self.model: Optional[PersonalAINetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'confidence': [],
            'domain_performance': {}
        }
        
        # Domain mapping
        self.domain_to_id = {
            'driving': 0,
            'walking': 1,
            'cycling': 2,
            'tennis': 3,
            'basketball': 4,
            'gaming': 5,
            'daily_navigation': 6
        }
        
        logger.info("PersonalModelTrainer initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for training."""
        return {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'hidden_dim': 512,
            'sequence_length': 32,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'save_best_model': True,
            'model_save_path': 'personal_model.pth'
        }
    
    def add_training_data(
        self,
        domain: str,
        scenario: str,
        context: np.ndarray,
        action: np.ndarray,
        biometrics: np.ndarray,
        outcome: float,
        stress_level: float,
        confidence: float,
        timestamp: datetime = None,
        metadata: Dict[str, Any] = None
    ):
        """Add a training data point."""
        if timestamp is None:
            timestamp = datetime.now()
        
        data_point = PersonalDataPoint(
            timestamp=timestamp,
            domain=domain,
            scenario=scenario,
            context_vector=context,
            action_vector=action,
            biometric_state=biometrics,
            outcome=outcome,
            stress_level=stress_level,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.training_data.append(data_point)
        logger.debug(f"Added training data point for {domain}/{scenario}")
    
    def load_historical_data(self, data_path: Union[str, Path]):
        """Load historical data from file."""
        data_path = Path(data_path)
        
        if data_path.suffix == '.json':
            with open(data_path, 'r') as f:
                data = json.load(f)
                self._process_json_data(data)
        elif data_path.suffix == '.pkl':
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.training_data.extend(data)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        logger.info(f"Loaded {len(self.training_data)} data points from {data_path}")
    
    def _process_json_data(self, data: List[Dict[str, Any]]):
        """Process JSON data into PersonalDataPoint objects."""
        for item in data:
            self.add_training_data(
                domain=item['domain'],
                scenario=item['scenario'],
                context=np.array(item['context']),
                action=np.array(item['action']),
                biometrics=np.array(item['biometrics']),
                outcome=item['outcome'],
                stress_level=item['stress_level'],
                confidence=item['confidence'],
                timestamp=datetime.fromisoformat(item['timestamp']),
                metadata=item.get('metadata', {})
            )
    
    def analyze_domains(self) -> Dict[str, DomainProfile]:
        """Analyze data to create domain profiles."""
        domain_data = {}
        
        # Group data by domain
        for dp in self.training_data:
            if dp.domain not in domain_data:
                domain_data[dp.domain] = []
            domain_data[dp.domain].append(dp)
        
        # Create profiles for each domain
        for domain, data_points in domain_data.items():
            profile = self._create_domain_profile(domain, data_points)
            self.domain_profiles[domain] = profile
            logger.info(f"Created profile for {domain}: {profile.skill_level:.2f} skill level")
        
        return self.domain_profiles
    
    def _create_domain_profile(self, domain: str, data_points: List[PersonalDataPoint]) -> DomainProfile:
        """Create a profile for a specific domain."""
        outcomes = [dp.outcome for dp in data_points]
        stress_levels = [dp.stress_level for dp in data_points]
        confidences = [dp.confidence for dp in data_points]
        
        # Calculate metrics
        skill_level = np.mean(outcomes)
        consistency = 1.0 - np.std(outcomes)
        stress_tolerance = 1.0 - np.mean(stress_levels)
        
        # Estimate total hours (rough calculation based on data density)
        time_span = max(dp.timestamp for dp in data_points) - min(dp.timestamp for dp in data_points)
        estimated_hours = min(time_span.total_seconds() / 3600, len(data_points) * 0.1)
        
        # Analyze preferred patterns (simplified)
        scenarios = [dp.scenario for dp in data_points]
        scenario_counts = {}
        for scenario in scenarios:
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        preferred_patterns = sorted(scenario_counts.keys(), 
                                  key=lambda x: scenario_counts[x], 
                                  reverse=True)[:5]
        
        # Risk preference based on action magnitudes
        action_magnitudes = [np.linalg.norm(dp.action_vector) for dp in data_points]
        risk_preference = np.mean(action_magnitudes)
        
        return DomainProfile(
            domain_name=domain,
            total_hours=estimated_hours,
            data_points=len(data_points),
            skill_level=skill_level,
            consistency=consistency,
            stress_tolerance=stress_tolerance,
            preferred_patterns=preferred_patterns,
            risk_preference=risk_preference,
            adaptation_speed=np.mean(confidences),
            dominant_strategies=scenario_counts
        )
    
    def create_personality_profile(self) -> PersonalityProfile:
        """Create overall personality profile from cross-domain analysis."""
        if not self.domain_profiles:
            self.analyze_domains()
        
        # Aggregate metrics across domains
        skill_levels = [p.skill_level for p in self.domain_profiles.values()]
        stress_tolerances = [p.stress_tolerance for p in self.domain_profiles.values()]
        consistencies = [p.consistency for p in self.domain_profiles.values()]
        risk_preferences = [p.risk_preference for p in self.domain_profiles.values()]
        
        # Calculate reaction speed from temporal patterns
        reaction_times = []
        for dp in self.training_data:
            if 'reaction_time' in dp.metadata:
                reaction_times.append(dp.metadata['reaction_time'])
        
        avg_reaction_speed = 1.0 - np.mean(reaction_times) if reaction_times else 0.5
        
        # Spatial awareness from spatial domains
        spatial_domains = ['driving', 'cycling', 'tennis', 'basketball']
        spatial_skills = [self.domain_profiles[d].skill_level 
                         for d in spatial_domains 
                         if d in self.domain_profiles]
        spatial_awareness = np.mean(spatial_skills) if spatial_skills else 0.5
        
        self.personality_profile = PersonalityProfile(
            risk_tolerance=np.mean(risk_preferences),
            stress_sensitivity=1.0 - np.mean(stress_tolerances),
            reaction_speed=avg_reaction_speed,
            spatial_awareness=spatial_awareness,
            pattern_consistency=np.mean(consistencies),
            learning_adaptability=np.std(skill_levels),  # High variance = good adaptation
            social_coordination=0.5,  # Would need team-based data to calculate
            planning_horizon=0.5,  # Would need long-term decision data
            decision_confidence=np.mean([dp.confidence for dp in self.training_data]),
            emotional_regulation=np.mean(stress_tolerances)
        )
        
        logger.info(f"Created personality profile: risk_tolerance={self.personality_profile.risk_tolerance:.2f}")
        return self.personality_profile
    
    def prepare_training_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for training."""
        if len(self.training_data) < 100:
            logger.warning(f"Only {len(self.training_data)} data points available. Recommend 1000+")
        
        # Split data
        split_idx = int(len(self.training_data) * (1 - self.config['validation_split']))
        train_data = self.training_data[:split_idx]
        val_data = self.training_data[split_idx:]
        
        # Create datasets
        train_dataset = PersonalDataset(train_data, self.config['sequence_length'])
        val_dataset = PersonalDataset(val_data, self.config['sequence_length'])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            drop_last=False
        )
        
        return train_loader, val_loader
    
    def initialize_model(self, context_dim: int, action_dim: int, biometric_dim: int):
        """Initialize the neural network model."""
        self.model = PersonalAINetwork(
            context_dim=context_dim,
            action_dim=action_dim,
            biometric_dim=biometric_dim,
            hidden_dim=self.config['hidden_dim'],
            num_domains=len(self.domain_to_id),
            sequence_length=self.config['sequence_length']
        )
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-5
        )
        
        logger.info(f"Initialized model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_model(self) -> Dict[str, Any]:
        """Train the personal AI model."""
        if not self.training_data:
            raise ValueError("No training data available")
        
        # Prepare data
        train_loader, val_loader = self.prepare_training_data()
        
        # Initialize model if not done
        if self.model is None:
            sample_dp = self.training_data[0]
            self.initialize_model(
                context_dim=len(sample_dp.context_vector),
                action_dim=len(sample_dp.action_vector),
                biometric_dim=len(sample_dp.biometric_state)
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Record history
            self.training_history['loss'].append(train_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if self.config['save_best_model']:
                    self.save_model(self.config['model_save_path'])
            else:
                patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Final evaluation
        final_metrics = self.evaluate_model()
        
        logger.info("Training completed")
        return final_metrics
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_input, batch_target, batch_info in train_loader:
            # Extract domain information (simplified - would need proper domain encoding)
            domain_ids = torch.zeros(batch_input.size(0), dtype=torch.long)
            
            # Forward pass
            outputs = self.model(
                context=batch_input,
                sequence=torch.zeros(batch_input.size(0), 0, batch_input.size(1)),  # Empty sequence for now
                domain_id=domain_ids
            )
            
            # Calculate losses
            action_loss = self.criterion(outputs['action'], batch_target)
            confidence_loss = self.criterion(outputs['confidence'], batch_info[:, 1:2])
            stress_loss = self.criterion(outputs['stress'], batch_info[:, 0:1])
            
            total_loss_batch = action_loss + 0.1 * confidence_loss + 0.1 * stress_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_input, batch_target, batch_info in val_loader:
                domain_ids = torch.zeros(batch_input.size(0), dtype=torch.long)
                
                outputs = self.model(
                    context=batch_input,
                    sequence=torch.zeros(batch_input.size(0), 0, batch_input.size(1)),
                    domain_id=domain_ids
                )
                
                action_loss = self.criterion(outputs['action'], batch_target)
                total_loss += action_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        metrics = {'validation_loss': avg_loss}
        
        return avg_loss, metrics
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate the trained model comprehensively."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        metrics = {
            'domain_profiles': self.domain_profiles,
            'personality_profile': self.personality_profile,
            'training_data_points': len(self.training_data),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'training_history': self.training_history
        }
        
        return metrics
    
    def save_model(self, path: Union[str, Path]):
        """Save the trained model and associated data."""
        save_data = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'config': self.config,
            'domain_profiles': self.domain_profiles,
            'personality_profile': self.personality_profile,
            'training_history': self.training_history,
            'domain_to_id': self.domain_to_id
        }
        
        torch.save(save_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]):
        """Load a trained model."""
        save_data = torch.load(path)
        
        self.config = save_data['config']
        self.domain_profiles = save_data['domain_profiles']
        self.personality_profile = save_data['personality_profile']
        self.training_history = save_data['training_history']
        self.domain_to_id = save_data['domain_to_id']
        
        if save_data['model_state_dict']:
            # Would need to reconstruct model architecture
            logger.info(f"Model loaded from {path}")
    
    def predict(
        self,
        context: np.ndarray,
        domain: str,
        biometric_state: np.ndarray = None
    ) -> Dict[str, float]:
        """Make a prediction using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        
        with torch.no_grad():
            # Prepare input
            if biometric_state is None:
                biometric_state = np.zeros(10)  # Default biometric state
            
            input_tensor = torch.cat([
                torch.FloatTensor(context),
                torch.FloatTensor(biometric_state)
            ]).unsqueeze(0)
            
            domain_id = torch.LongTensor([self.domain_to_id.get(domain, 0)])
            
            # Forward pass
            outputs = self.model(
                context=input_tensor,
                sequence=torch.zeros(1, 0, input_tensor.size(1)),
                domain_id=domain_id
            )
            
            return {
                'action': outputs['action'].squeeze().numpy(),
                'confidence': outputs['confidence'].item(),
                'stress': outputs['stress'].item()
            }
    
    def visualize_training_progress(self, save_path: str = None):
        """Visualize training progress and model characteristics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.training_history['loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Domain skill levels
        if self.domain_profiles:
            domains = list(self.domain_profiles.keys())
            skills = [self.domain_profiles[d].skill_level for d in domains]
            
            axes[0, 1].bar(domains, skills)
            axes[0, 1].set_title('Skill Levels by Domain')
            axes[0, 1].set_ylabel('Skill Level')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Data distribution
        domain_counts = {}
        for dp in self.training_data:
            domain_counts[dp.domain] = domain_counts.get(dp.domain, 0) + 1
        
        if domain_counts:
            axes[1, 0].pie(domain_counts.values(), labels=domain_counts.keys(), autopct='%1.1f%%')
            axes[1, 0].set_title('Data Distribution by Domain')
        
        # Personality radar chart
        if self.personality_profile:
            categories = ['Risk Tolerance', 'Reaction Speed', 'Spatial Awareness', 
                         'Pattern Consistency', 'Decision Confidence']
            values = [
                self.personality_profile.risk_tolerance,
                self.personality_profile.reaction_speed,
                self.personality_profile.spatial_awareness,
                self.personality_profile.pattern_consistency,
                self.personality_profile.decision_confidence
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            values += values[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))
            
            axes[1, 1].plot(angles, values, 'o-', linewidth=2)
            axes[1, 1].fill(angles, values, alpha=0.25)
            axes[1, 1].set_xticks(angles[:-1])
            axes[1, 1].set_xticklabels(categories)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title('Personality Profile')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training visualization saved to {save_path}")
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    # Create trainer
    trainer = PersonalModelTrainer()
    
    # Simulate adding historical data
    for i in range(1000):
        domain = np.random.choice(['driving', 'walking', 'tennis', 'cycling'])
        context = np.random.rand(32)
        action = np.random.rand(16) 
        biometrics = np.random.rand(10)
        outcome = np.random.rand()
        stress = np.random.rand()
        confidence = np.random.rand()
        
        trainer.add_training_data(
            domain=domain,
            scenario=f"{domain}_scenario_{i%10}",
            context=context,
            action=action,
            biometrics=biometrics,
            outcome=outcome,
            stress_level=stress,
            confidence=confidence
        )
    
    # Analyze and train
    trainer.analyze_domains()
    trainer.create_personality_profile()
    
    print("Domain Profiles:")
    for domain, profile in trainer.domain_profiles.items():
        print(f"  {domain}: skill={profile.skill_level:.2f}, consistency={profile.consistency:.2f}")
    
    print(f"\nPersonality Profile:")
    print(f"  Risk Tolerance: {trainer.personality_profile.risk_tolerance:.2f}")
    print(f"  Spatial Awareness: {trainer.personality_profile.spatial_awareness:.2f}")
    
    # Train model
    metrics = trainer.train_model()
    print(f"\nTraining completed with {metrics['training_data_points']} data points") 