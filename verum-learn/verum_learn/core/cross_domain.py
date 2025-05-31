"""
Cross-Domain Learning System

Implements the core innovation of Verum: learning patterns from one domain
(e.g., tennis defensive patterns) and applying them to another domain (e.g., driving avoidance).

The key insight is that all human locomotion shares fundamental patterns of
goal-directed obstacle avoidance, and fear responses transfer across activities.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class LearningDomain(Enum):
    """Supported learning domains."""
    DRIVING = "driving"
    WALKING = "walking"
    CYCLING = "cycling"
    TENNIS = "tennis"
    BASKETBALL = "basketball"
    GAMING = "gaming"
    DAILY_NAVIGATION = "daily_navigation"


@dataclass
class DomainPattern:
    """Represents a learned pattern from a specific domain."""
    domain: LearningDomain
    scenario_type: str
    stimulus_vector: np.ndarray
    response_vector: np.ndarray
    biometric_state: np.ndarray
    success_rate: float
    stress_level: float
    confidence: float
    timestamp: float
    context: Dict[str, Any]


@dataclass
class TransferResult:
    """Result of pattern transfer between domains."""
    source_domain: LearningDomain
    target_domain: LearningDomain
    similarity_score: float
    transfer_confidence: float
    adapted_response: np.ndarray
    reasoning: str


class DomainSimilarityCalculator:
    """Calculates similarity between different domains for pattern transfer."""
    
    def __init__(self):
        # Domain similarity matrix (learned from data or expert knowledge)
        self.domain_similarities = {
            (LearningDomain.TENNIS, LearningDomain.DRIVING): 0.7,  # Defensive reactions
            (LearningDomain.WALKING, LearningDomain.DRIVING): 0.8,  # Navigation patterns
            (LearningDomain.CYCLING, LearningDomain.DRIVING): 0.9,  # Vehicle control
            (LearningDomain.BASKETBALL, LearningDomain.DRIVING): 0.6,  # Spatial awareness
            (LearningDomain.GAMING, LearningDomain.DRIVING): 0.5,   # Strategic thinking
        }
        
        # Feature weights for different aspects of patterns
        self.feature_weights = {
            "reaction_time": 0.9,
            "spatial_awareness": 0.8,
            "fear_response": 0.95,
            "decision_speed": 0.7,
            "biometric_pattern": 0.85,
        }
    
    def calculate_similarity(
        self, 
        pattern1: DomainPattern, 
        pattern2: DomainPattern
    ) -> float:
        """Calculate similarity between two domain patterns."""
        
        # Base domain similarity
        domain_sim = self.domain_similarities.get(
            (pattern1.domain, pattern2.domain), 0.3
        )
        
        # Stimulus similarity (scenario context)
        stimulus_sim = cosine_similarity(
            pattern1.stimulus_vector.reshape(1, -1),
            pattern2.stimulus_vector.reshape(1, -1)
        )[0, 0]
        
        # Biometric similarity
        biometric_sim = cosine_similarity(
            pattern1.biometric_state.reshape(1, -1),
            pattern2.biometric_state.reshape(1, -1)
        )[0, 0]
        
        # Stress level similarity
        stress_sim = 1.0 - abs(pattern1.stress_level - pattern2.stress_level)
        
        # Combine similarities with weights
        total_similarity = (
            domain_sim * 0.3 +
            stimulus_sim * 0.3 +
            biometric_sim * 0.25 +
            stress_sim * 0.15
        )
        
        return np.clip(total_similarity, 0.0, 1.0)


class PatternTransferNetwork(nn.Module):
    """Neural network for adapting patterns between domains."""
    
    def __init__(
        self, 
        stimulus_dim: int = 128,
        response_dim: int = 64,
        biometric_dim: int = 32,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.stimulus_dim = stimulus_dim
        self.response_dim = response_dim
        self.biometric_dim = biometric_dim
        
        # Encoder: Encode domain-specific patterns into universal space
        self.encoder = nn.Sequential(
            nn.Linear(stimulus_dim + biometric_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 128)  # Universal pattern space
        )
        
        # Domain adapters: Adapt universal patterns to specific domains
        self.domain_adapters = nn.ModuleDict({
            domain.value: nn.Sequential(
                nn.Linear(128, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, response_dim)
            )
            for domain in LearningDomain
        })
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(128 + stimulus_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        stimulus: torch.Tensor, 
        biometric: torch.Tensor,
        target_domain: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for pattern transfer.
        
        Args:
            stimulus: Input stimulus vector
            biometric: Biometric state vector
            target_domain: Target domain for adaptation
            
        Returns:
            Tuple of (adapted_response, confidence)
        """
        # Encode to universal pattern space
        combined_input = torch.cat([stimulus, biometric], dim=-1)
        universal_pattern = self.encoder(combined_input)
        
        # Adapt to target domain
        if target_domain not in self.domain_adapters:
            raise ValueError(f"Unknown domain: {target_domain}")
        
        adapted_response = self.domain_adapters[target_domain](universal_pattern)
        
        # Predict confidence
        confidence_input = torch.cat([universal_pattern, stimulus], dim=-1)
        confidence = self.confidence_predictor(confidence_input)
        
        return adapted_response, confidence


class CrossDomainLearner:
    """
    Main cross-domain learning system.
    
    This system learns patterns from multiple domains and enables transfer
    of successful patterns between domains. For example, defensive patterns
    learned from tennis can be applied to driving scenarios.
    """
    
    def __init__(
        self,
        stimulus_dim: int = 128,
        response_dim: int = 64,
        biometric_dim: int = 32,
        learning_rate: float = 0.001,
        device: str = "cpu"
    ):
        self.device = device
        self.similarity_calculator = DomainSimilarityCalculator()
        self.pattern_database: Dict[LearningDomain, List[DomainPattern]] = {
            domain: [] for domain in LearningDomain
        }
        
        # Neural network for pattern transfer
        self.transfer_network = PatternTransferNetwork(
            stimulus_dim, response_dim, biometric_dim
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.transfer_network.parameters(), 
            lr=learning_rate
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        logger.info("CrossDomainLearner initialized")
    
    def add_pattern(self, pattern: DomainPattern) -> None:
        """Add a new pattern to the database."""
        self.pattern_database[pattern.domain].append(pattern)
        logger.debug(f"Added pattern for {pattern.domain.value}: {pattern.scenario_type}")
    
    def find_similar_patterns(
        self, 
        query_pattern: DomainPattern,
        target_domain: LearningDomain,
        min_similarity: float = 0.5,
        max_results: int = 5
    ) -> List[Tuple[DomainPattern, float]]:
        """
        Find patterns similar to the query pattern from other domains.
        
        Args:
            query_pattern: Pattern to find similarities for
            target_domain: Domain to search for patterns
            min_similarity: Minimum similarity threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of (pattern, similarity_score) tuples
        """
        similar_patterns = []
        
        # Search through all domains except the query domain
        for domain, patterns in self.pattern_database.items():
            if domain == query_pattern.domain:
                continue
                
            for pattern in patterns:
                similarity = self.similarity_calculator.calculate_similarity(
                    query_pattern, pattern
                )
                
                if similarity >= min_similarity:
                    similar_patterns.append((pattern, similarity))
        
        # Sort by similarity and return top results
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        return similar_patterns[:max_results]
    
    def transfer_pattern(
        self,
        source_pattern: DomainPattern,
        target_domain: LearningDomain,
        target_stimulus: np.ndarray,
        target_biometric: np.ndarray
    ) -> TransferResult:
        """
        Transfer a pattern from one domain to another.
        
        This is the core innovation: taking a successful avoidance pattern
        from tennis and applying it to a driving scenario.
        """
        if not self.is_trained:
            logger.warning("Transfer network not trained, using similarity-based transfer")
            return self._similarity_based_transfer(
                source_pattern, target_domain, target_stimulus, target_biometric
            )
        
        # Convert inputs to tensors
        stimulus_tensor = torch.FloatTensor(target_stimulus).unsqueeze(0).to(self.device)
        biometric_tensor = torch.FloatTensor(target_biometric).unsqueeze(0).to(self.device)
        
        # Get adapted response from neural network
        with torch.no_grad():
            adapted_response, confidence = self.transfer_network(
                stimulus_tensor, biometric_tensor, target_domain.value
            )
        
        transfer_confidence = confidence.item()
        adapted_response_np = adapted_response.squeeze(0).cpu().numpy()
        
        # Calculate similarity for reasoning
        similarity = self.similarity_calculator.calculate_similarity(
            source_pattern, 
            DomainPattern(
                domain=target_domain,
                scenario_type="target",
                stimulus_vector=target_stimulus,
                response_vector=adapted_response_np,
                biometric_state=target_biometric,
                success_rate=0.0,
                stress_level=0.0,
                confidence=0.0,
                timestamp=0.0,
                context={}
            )
        )
        
        reasoning = self._generate_transfer_reasoning(
            source_pattern, target_domain, similarity, transfer_confidence
        )
        
        return TransferResult(
            source_domain=source_pattern.domain,
            target_domain=target_domain,
            similarity_score=similarity,
            transfer_confidence=transfer_confidence,
            adapted_response=adapted_response_np,
            reasoning=reasoning
        )
    
    def _similarity_based_transfer(
        self,
        source_pattern: DomainPattern,
        target_domain: LearningDomain,
        target_stimulus: np.ndarray,
        target_biometric: np.ndarray
    ) -> TransferResult:
        """Fallback transfer method based on similarity matching."""
        
        # Find the most similar pattern in the target domain
        target_patterns = self.pattern_database.get(target_domain, [])
        
        if not target_patterns:
            # Use direct adaptation with scaling
            adapted_response = source_pattern.response_vector * 0.8  # Conservative scaling
            transfer_confidence = 0.3
        else:
            # Find most similar target pattern
            best_pattern = None
            best_similarity = 0.0
            
            for pattern in target_patterns:
                similarity = cosine_similarity(
                    target_stimulus.reshape(1, -1),
                    pattern.stimulus_vector.reshape(1, -1)
                )[0, 0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pattern = pattern
            
            if best_pattern:
                # Blend source and target responses
                blend_factor = best_similarity
                adapted_response = (
                    blend_factor * best_pattern.response_vector +
                    (1 - blend_factor) * source_pattern.response_vector
                )
                transfer_confidence = best_similarity * 0.8
            else:
                adapted_response = source_pattern.response_vector * 0.6
                transfer_confidence = 0.2
        
        similarity = self.similarity_calculator.calculate_similarity(
            source_pattern,
            DomainPattern(
                domain=target_domain,
                scenario_type="target",
                stimulus_vector=target_stimulus,
                response_vector=adapted_response,
                biometric_state=target_biometric,
                success_rate=0.0,
                stress_level=0.0,
                confidence=0.0,
                timestamp=0.0,
                context={}
            )
        )
        
        reasoning = f"Similarity-based transfer from {source_pattern.domain.value} to {target_domain.value}"
        
        return TransferResult(
            source_domain=source_pattern.domain,
            target_domain=target_domain,
            similarity_score=similarity,
            transfer_confidence=transfer_confidence,
            adapted_response=adapted_response,
            reasoning=reasoning
        )
    
    def _generate_transfer_reasoning(
        self,
        source_pattern: DomainPattern,
        target_domain: LearningDomain,
        similarity: float,
        confidence: float
    ) -> str:
        """Generate human-readable reasoning for pattern transfer."""
        
        domain_map = {
            LearningDomain.TENNIS: "tennis defensive patterns",
            LearningDomain.WALKING: "pedestrian navigation skills",
            LearningDomain.CYCLING: "bicycle control experience",
            LearningDomain.BASKETBALL: "spatial awareness from basketball",
            LearningDomain.GAMING: "strategic decision-making from gaming",
            LearningDomain.DRIVING: "driving experience",
        }
        
        source_desc = domain_map.get(source_pattern.domain, source_pattern.domain.value)
        target_desc = domain_map.get(target_domain, target_domain.value)
        
        if confidence > 0.8:
            confidence_desc = "High confidence"
        elif confidence > 0.6:
            confidence_desc = "Medium confidence"
        else:
            confidence_desc = "Low confidence"
        
        return (
            f"{confidence_desc} transfer from {source_desc} to {target_desc}. "
            f"Pattern similarity: {similarity:.2f}. "
            f"Adapting {source_pattern.scenario_type} response pattern."
        )
    
    def train_transfer_network(
        self, 
        training_patterns: List[DomainPattern],
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Train the neural network for pattern transfer.
        
        This trains the network to learn how to adapt patterns between domains
        based on successful transfer examples.
        """
        logger.info("Training pattern transfer network...")
        
        # Prepare training data
        X_stimulus, X_biometric, y_response, domains = self._prepare_training_data(
            training_patterns
        )
        
        # Fit scaler
        combined_X = np.concatenate([X_stimulus, X_biometric], axis=1)
        self.scaler.fit(combined_X)
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_stimulus),
            torch.FloatTensor(X_biometric),
            torch.FloatTensor(y_response),
            torch.LongTensor(domains)
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Training loop
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_stimulus, batch_biometric, batch_response, batch_domains in dataloader:
                batch_stimulus = batch_stimulus.to(self.device)
                batch_biometric = batch_biometric.to(self.device)
                batch_response = batch_response.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass for each domain in batch
                total_loss = 0.0
                for i in range(len(batch_stimulus)):
                    domain_name = list(LearningDomain)[batch_domains[i]].value
                    
                    pred_response, pred_confidence = self.transfer_network(
                        batch_stimulus[i:i+1],
                        batch_biometric[i:i+1], 
                        domain_name
                    )
                    
                    # Response loss
                    response_loss = nn.MSELoss()(pred_response, batch_response[i:i+1])
                    
                    # Confidence loss (encourage high confidence for good matches)
                    target_confidence = torch.FloatTensor([0.8]).to(self.device)
                    confidence_loss = nn.MSELoss()(pred_confidence, target_confidence)
                    
                    total_loss += response_loss + 0.1 * confidence_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            training_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info("Transfer network training completed")
        
        return {
            "final_loss": training_losses[-1],
            "initial_loss": training_losses[0],
            "epochs_trained": epochs,
        }
    
    def _prepare_training_data(
        self, patterns: List[DomainPattern]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from domain patterns."""
        
        X_stimulus = []
        X_biometric = []
        y_response = []
        domains = []
        
        for pattern in patterns:
            X_stimulus.append(pattern.stimulus_vector)
            X_biometric.append(pattern.biometric_state)
            y_response.append(pattern.response_vector)
            domains.append(list(LearningDomain).index(pattern.domain))
        
        return (
            np.array(X_stimulus),
            np.array(X_biometric),
            np.array(y_response),
            np.array(domains)
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        total_patterns = sum(len(patterns) for patterns in self.pattern_database.values())
        
        domain_counts = {
            domain.value: len(patterns) 
            for domain, patterns in self.pattern_database.items()
        }
        
        return {
            "total_patterns": total_patterns,
            "domain_counts": domain_counts,
            "is_trained": self.is_trained,
            "supported_domains": [domain.value for domain in LearningDomain],
        }


# Example usage and testing
if __name__ == "__main__":
    # Create a cross-domain learner
    learner = CrossDomainLearner()
    
    # Example: Tennis defensive pattern
    tennis_pattern = DomainPattern(
        domain=LearningDomain.TENNIS,
        scenario_type="defensive_backhand",
        stimulus_vector=np.random.rand(128),  # Ball trajectory, speed, etc.
        response_vector=np.random.rand(64),   # Body movement, timing
        biometric_state=np.random.rand(32),   # Heart rate, stress level
        success_rate=0.85,
        stress_level=0.7,
        confidence=0.9,
        timestamp=1234567890.0,
        context={"ball_speed": 120, "court_position": "baseline"}
    )
    
    learner.add_pattern(tennis_pattern)
    
    # Transfer to driving scenario
    driving_stimulus = np.random.rand(128)  # Obstacle approach vector
    driving_biometric = np.random.rand(32)  # Current stress state
    
    transfer_result = learner.transfer_pattern(
        tennis_pattern,
        LearningDomain.DRIVING,
        driving_stimulus,
        driving_biometric
    )
    
    print(f"Transfer confidence: {transfer_result.transfer_confidence:.2f}")
    print(f"Reasoning: {transfer_result.reasoning}") 