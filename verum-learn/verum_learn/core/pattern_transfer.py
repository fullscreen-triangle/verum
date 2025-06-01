"""
Pattern Transfer System

Advanced algorithms for transferring learned patterns between domains.
This implements the core innovation of transferring tennis defensive patterns
to driving scenarios, walking navigation to vehicle coordination, etc.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


@dataclass
class TransferPattern:
    """A pattern that can be transferred between domains."""
    id: str
    source_domain: str
    target_domain: str
    pattern_vector: np.ndarray
    context_vector: np.ndarray
    success_probability: float
    transfer_confidence: float
    biometric_signature: np.ndarray
    temporal_sequence: List[np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class TransferResult:
    """Result of a pattern transfer operation."""
    success: bool
    adapted_pattern: Optional[TransferPattern]
    confidence: float
    reasoning: str
    alternative_patterns: List[TransferPattern]
    risk_assessment: float
    expected_performance: float


class DomainBridge:
    """Creates bridges between different learning domains."""
    
    def __init__(self, source_domain: str, target_domain: str):
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.bridge_strength = 0.0
        self.verified_transfers = []
        self.failed_transfers = []
        
        # Domain-specific transformation matrices
        self.transformation_matrix = None
        self.inverse_transformation_matrix = None
        
        # Learned mappings between domains
        self.feature_mappings = {}
        self.temporal_mappings = {}
        
    def calculate_bridge_strength(self) -> float:
        """Calculate how strong the bridge is between domains."""
        if not self.verified_transfers:
            return 0.0
            
        success_rate = len(self.verified_transfers) / (
            len(self.verified_transfers) + len(self.failed_transfers)
        )
        
        avg_confidence = np.mean([t.transfer_confidence for t in self.verified_transfers])
        
        # Factor in domain similarity
        domain_similarity = self._calculate_domain_similarity()
        
        self.bridge_strength = (success_rate * 0.4 + avg_confidence * 0.4 + domain_similarity * 0.2)
        return self.bridge_strength
    
    def _calculate_domain_similarity(self) -> float:
        """Calculate inherent similarity between domains."""
        similarity_matrix = {
            ("tennis", "driving"): 0.75,  # Defensive reactions, spatial awareness
            ("walking", "driving"): 0.85,  # Navigation, path planning
            ("cycling", "driving"): 0.90,  # Vehicle control, balance
            ("basketball", "driving"): 0.65,  # Spatial awareness, team coordination
            ("gaming", "driving"): 0.55,   # Strategic thinking, reaction time
            ("daily_navigation", "driving"): 0.80,  # Route planning, efficiency
        }
        
        key1 = (self.source_domain, self.target_domain)
        key2 = (self.target_domain, self.source_domain)
        
        return similarity_matrix.get(key1, similarity_matrix.get(key2, 0.3))


class TemporalPatternExtractor:
    """Extracts temporal patterns from behavioral sequences."""
    
    def __init__(self, sequence_length: int = 50):
        self.sequence_length = sequence_length
        self.pattern_templates = {}
        
    def extract_pattern(self, sequence: List[np.ndarray], domain: str) -> Dict[str, np.ndarray]:
        """Extract key temporal patterns from a sequence."""
        if len(sequence) < self.sequence_length:
            # Pad sequence if too short
            padding = [sequence[-1]] * (self.sequence_length - len(sequence))
            sequence = sequence + padding
        elif len(sequence) > self.sequence_length:
            # Sample key points if too long
            indices = np.linspace(0, len(sequence) - 1, self.sequence_length, dtype=int)
            sequence = [sequence[i] for i in indices]
        
        sequence_array = np.array(sequence)
        
        patterns = {
            'onset_pattern': self._extract_onset_pattern(sequence_array),
            'peak_pattern': self._extract_peak_pattern(sequence_array),
            'recovery_pattern': self._extract_recovery_pattern(sequence_array),
            'rhythm_pattern': self._extract_rhythm_pattern(sequence_array),
            'variability_pattern': self._extract_variability_pattern(sequence_array),
        }
        
        return patterns
    
    def _extract_onset_pattern(self, sequence: np.ndarray) -> np.ndarray:
        """Extract how the pattern begins."""
        onset_length = min(10, len(sequence) // 3)
        onset = sequence[:onset_length]
        
        # Calculate acceleration and jerk
        if len(onset) > 2:
            velocity = np.diff(onset, axis=0)
            acceleration = np.diff(velocity, axis=0)
            return np.concatenate([onset.flatten(), acceleration.flatten()])
        return onset.flatten()
    
    def _extract_peak_pattern(self, sequence: np.ndarray) -> np.ndarray:
        """Extract peak response characteristics."""
        # Find peaks in the magnitude of the sequence
        magnitudes = np.linalg.norm(sequence, axis=1) if sequence.ndim > 1 else np.abs(sequence)
        peak_idx = np.argmax(magnitudes)
        
        # Extract pattern around peak
        window = 5
        start = max(0, peak_idx - window)
        end = min(len(sequence), peak_idx + window + 1)
        
        peak_pattern = sequence[start:end]
        return peak_pattern.flatten()
    
    def _extract_recovery_pattern(self, sequence: np.ndarray) -> np.ndarray:
        """Extract how the pattern returns to baseline."""
        recovery_length = min(10, len(sequence) // 3)
        recovery = sequence[-recovery_length:]
        
        # Calculate decay characteristics
        if len(recovery) > 1:
            decay_rate = np.diff(recovery, axis=0)
            return np.concatenate([recovery.flatten(), decay_rate.flatten()])
        return recovery.flatten()
    
    def _extract_rhythm_pattern(self, sequence: np.ndarray) -> np.ndarray:
        """Extract rhythmic/periodic components."""
        magnitudes = np.linalg.norm(sequence, axis=1) if sequence.ndim > 1 else np.abs(sequence)
        
        # Simple FFT to find dominant frequencies
        fft = np.fft.fft(magnitudes)
        power_spectrum = np.abs(fft[:len(fft)//2])
        
        # Return top frequency components
        return power_spectrum[:min(20, len(power_spectrum))]
    
    def _extract_variability_pattern(self, sequence: np.ndarray) -> np.ndarray:
        """Extract variability and consistency patterns."""
        if sequence.ndim > 1:
            # Multi-dimensional variability
            means = np.mean(sequence, axis=0)
            stds = np.std(sequence, axis=0)
            ranges = np.ptp(sequence, axis=0)
            return np.concatenate([means, stds, ranges])
        else:
            # Single-dimensional variability
            return np.array([
                np.mean(sequence),
                np.std(sequence),
                np.ptp(sequence),
                np.median(sequence)
            ])


class BiometricPatternMatcher:
    """Matches biometric patterns across domains."""
    
    def __init__(self):
        self.stress_patterns = {}
        self.arousal_patterns = {}
        self.fear_patterns = {}
        
    def extract_biometric_signature(self, biometric_sequence: List[Dict[str, float]]) -> np.ndarray:
        """Extract a signature biometric pattern."""
        features = []
        
        for metric in ['heart_rate', 'skin_conductance', 'breathing_rate']:
            values = [b.get(metric, 0.0) for b in biometric_sequence]
            
            if values:
                features.extend([
                    np.mean(values),
                    np.std(values),
                    np.max(values),
                    np.min(values),
                    np.median(values),
                ])
        
        return np.array(features)
    
    def match_stress_patterns(
        self, 
        source_pattern: np.ndarray, 
        target_candidates: List[np.ndarray],
        threshold: float = 0.8
    ) -> List[Tuple[int, float]]:
        """Match stress response patterns between domains."""
        matches = []
        
        for i, candidate in enumerate(target_candidates):
            similarity = 1 - cosine(source_pattern, candidate)
            if similarity >= threshold:
                matches.append((i, similarity))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)


class PatternTransfer:
    """Main pattern transfer engine implementing cross-domain learning."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.domain_bridges = {}
        self.temporal_extractor = TemporalPatternExtractor()
        self.biometric_matcher = BiometricPatternMatcher()
        
        # Neural networks for pattern adaptation
        self.adaptation_networks = {}
        self.confidence_networks = {}
        
        # Pattern databases
        self.verified_patterns = {}
        self.failed_patterns = {}
        
        # Transfer statistics
        self.transfer_stats = {
            'total_attempts': 0,
            'successful_transfers': 0,
            'failed_transfers': 0,
            'average_confidence': 0.0,
        }
        
        logger.info("PatternTransfer system initialized")
    
    def create_domain_bridge(self, source_domain: str, target_domain: str) -> DomainBridge:
        """Create or retrieve a bridge between two domains."""
        bridge_key = f"{source_domain}->{target_domain}"
        
        if bridge_key not in self.domain_bridges:
            self.domain_bridges[bridge_key] = DomainBridge(source_domain, target_domain)
            logger.info(f"Created domain bridge: {bridge_key}")
        
        return self.domain_bridges[bridge_key]
    
    def transfer_pattern(
        self,
        source_pattern: TransferPattern,
        target_domain: str,
        target_context: np.ndarray,
        target_biometrics: np.ndarray
    ) -> TransferResult:
        """Transfer a pattern from source domain to target domain."""
        self.transfer_stats['total_attempts'] += 1
        
        # Get or create domain bridge
        bridge = self.create_domain_bridge(source_pattern.source_domain, target_domain)
        
        # Check if we have enough bridge strength
        bridge_strength = bridge.calculate_bridge_strength()
        
        if bridge_strength < 0.3:
            logger.warning(f"Weak bridge strength ({bridge_strength:.2f}) for {source_pattern.source_domain} -> {target_domain}")
        
        # Extract temporal patterns
        source_temporal = self.temporal_extractor.extract_pattern(
            source_pattern.temporal_sequence, 
            source_pattern.source_domain
        )
        
        # Match biometric patterns
        biometric_matches = self.biometric_matcher.match_stress_patterns(
            source_pattern.biometric_signature,
            [target_biometrics]
        )
        
        biometric_confidence = biometric_matches[0][1] if biometric_matches else 0.0
        
        # Adapt the pattern to target domain
        adapted_pattern = self._adapt_pattern_to_domain(
            source_pattern,
            target_domain,
            target_context,
            bridge
        )
        
        # Calculate overall confidence
        confidence = self._calculate_transfer_confidence(
            source_pattern,
            adapted_pattern,
            bridge_strength,
            biometric_confidence
        )
        
        # Generate reasoning
        reasoning = self._generate_transfer_reasoning(
            source_pattern,
            target_domain,
            confidence,
            bridge_strength,
            biometric_confidence
        )
        
        # Assess risk
        risk = self._assess_transfer_risk(source_pattern, target_domain, confidence)
        
        # Create result
        result = TransferResult(
            success=confidence >= 0.6,
            adapted_pattern=adapted_pattern if confidence >= 0.6 else None,
            confidence=confidence,
            reasoning=reasoning,
            alternative_patterns=[],  # Could implement alternative suggestions
            risk_assessment=risk,
            expected_performance=confidence * source_pattern.success_probability
        )
        
        # Update statistics
        if result.success:
            self.transfer_stats['successful_transfers'] += 1
            bridge.verified_transfers.append(adapted_pattern)
        else:
            self.transfer_stats['failed_transfers'] += 1
            bridge.failed_transfers.append(source_pattern)
        
        self._update_transfer_statistics()
        
        return result
    
    def _adapt_pattern_to_domain(
        self,
        source_pattern: TransferPattern,
        target_domain: str,
        target_context: np.ndarray,
        bridge: DomainBridge
    ) -> TransferPattern:
        """Adapt a pattern to work in the target domain."""
        
        # Use neural network if available
        network_key = f"{source_pattern.source_domain}->{target_domain}"
        if network_key in self.adaptation_networks:
            adapted_vector = self._neural_adaptation(
                source_pattern.pattern_vector,
                target_context,
                self.adaptation_networks[network_key]
            )
        else:
            # Use heuristic adaptation
            adapted_vector = self._heuristic_adaptation(
                source_pattern.pattern_vector,
                source_pattern.source_domain,
                target_domain
            )
        
        # Create adapted pattern
        adapted_pattern = TransferPattern(
            id=f"{source_pattern.id}_adapted_{target_domain}",
            source_domain=source_pattern.source_domain,
            target_domain=target_domain,
            pattern_vector=adapted_vector,
            context_vector=target_context,
            success_probability=source_pattern.success_probability * bridge.bridge_strength,
            transfer_confidence=0.0,  # Will be calculated
            biometric_signature=source_pattern.biometric_signature,
            temporal_sequence=source_pattern.temporal_sequence,
            metadata={
                **source_pattern.metadata,
                'adapted_from': source_pattern.id,
                'adaptation_method': 'neural' if network_key in self.adaptation_networks else 'heuristic'
            }
        )
        
        return adapted_pattern
    
    def _neural_adaptation(
        self,
        source_vector: np.ndarray,
        target_context: np.ndarray,
        network: nn.Module
    ) -> np.ndarray:
        """Use neural network to adapt pattern vector."""
        with torch.no_grad():
            source_tensor = torch.FloatTensor(source_vector).unsqueeze(0)
            context_tensor = torch.FloatTensor(target_context).unsqueeze(0)
            
            adapted_tensor = network(source_tensor, context_tensor)
            return adapted_tensor.squeeze(0).numpy()
    
    def _heuristic_adaptation(
        self,
        source_vector: np.ndarray,
        source_domain: str,
        target_domain: str
    ) -> np.ndarray:
        """Use heuristic rules to adapt pattern vector."""
        adaptation_rules = {
            ('tennis', 'driving'): {
                'defensive_amplification': 1.2,
                'reaction_time_adjustment': 0.8,
                'spatial_scaling': 1.1,
            },
            ('walking', 'driving'): {
                'speed_scaling': 10.0,  # Walking to driving speed
                'obstacle_sensitivity': 1.5,
                'path_smoothing': 0.7,
            },
            ('cycling', 'driving'): {
                'balance_transfer': 0.3,  # Less balance needed in car
                'momentum_conservation': 1.1,
                'steering_sensitivity': 0.8,
            },
        }
        
        rule_key = (source_domain, target_domain)
        if rule_key in adaptation_rules:
            rules = adaptation_rules[rule_key]
            adapted_vector = source_vector.copy()
            
            # Apply domain-specific adaptations
            for rule_name, factor in rules.items():
                # Simple scaling - in reality would be more sophisticated
                adapted_vector *= factor
                
            return adapted_vector
        else:
            # Default conservative adaptation
            return source_vector * 0.8
    
    def _calculate_transfer_confidence(
        self,
        source_pattern: TransferPattern,
        adapted_pattern: TransferPattern,
        bridge_strength: float,
        biometric_confidence: float
    ) -> float:
        """Calculate confidence in the pattern transfer."""
        
        # Base confidence from source pattern
        base_confidence = source_pattern.success_probability
        
        # Domain bridge strength factor
        bridge_factor = bridge_strength
        
        # Biometric similarity factor
        biometric_factor = biometric_confidence
        
        # Pattern complexity factor (simpler patterns transfer better)
        complexity = np.std(source_pattern.pattern_vector)
        complexity_factor = 1.0 / (1.0 + complexity)
        
        # Combine factors
        confidence = (
            base_confidence * 0.3 +
            bridge_factor * 0.3 +
            biometric_factor * 0.2 +
            complexity_factor * 0.2
        )
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _generate_transfer_reasoning(
        self,
        source_pattern: TransferPattern,
        target_domain: str,
        confidence: float,
        bridge_strength: float,
        biometric_confidence: float
    ) -> str:
        """Generate human-readable reasoning for the transfer."""
        
        domain_descriptions = {
            'tennis': 'tennis defensive patterns',
            'walking': 'pedestrian navigation skills',
            'cycling': 'bicycle control experience',
            'basketball': 'court awareness and teamwork',
            'gaming': 'strategic decision-making',
            'driving': 'automotive control patterns'
        }
        
        source_desc = domain_descriptions.get(source_pattern.source_domain, source_pattern.source_domain)
        target_desc = domain_descriptions.get(target_domain, target_domain)
        
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        
        reasoning = f"Transferring {source_desc} to {target_desc} with {confidence_level} confidence ({confidence:.2f}). "
        
        if bridge_strength > 0.7:
            reasoning += f"Strong domain bridge (strength: {bridge_strength:.2f}) supports this transfer. "
        elif bridge_strength > 0.4:
            reasoning += f"Moderate domain bridge (strength: {bridge_strength:.2f}) allows cautious transfer. "
        else:
            reasoning += f"Weak domain bridge (strength: {bridge_strength:.2f}) requires careful validation. "
        
        if biometric_confidence > 0.7:
            reasoning += f"Biometric patterns show good compatibility ({biometric_confidence:.2f}). "
        
        reasoning += f"Expected performance: {confidence * source_pattern.success_probability:.2f}"
        
        return reasoning
    
    def _assess_transfer_risk(
        self,
        source_pattern: TransferPattern,
        target_domain: str,
        confidence: float
    ) -> float:
        """Assess risk level of the pattern transfer."""
        
        # Higher risk for low confidence transfers
        confidence_risk = 1.0 - confidence
        
        # Domain-specific risk factors
        domain_risks = {
            'driving': 0.8,  # High risk domain
            'cycling': 0.6,  # Medium risk
            'walking': 0.3,  # Lower risk
            'gaming': 0.1,   # Very low risk
        }
        
        domain_risk = domain_risks.get(target_domain, 0.5)
        
        # Pattern-specific risk (high-intensity patterns are riskier)
        pattern_intensity = np.linalg.norm(source_pattern.pattern_vector)
        intensity_risk = min(pattern_intensity / 10.0, 1.0)
        
        # Combine risk factors
        total_risk = (
            confidence_risk * 0.4 +
            domain_risk * 0.4 +
            intensity_risk * 0.2
        )
        
        return np.clip(total_risk, 0.0, 1.0)
    
    def _update_transfer_statistics(self):
        """Update running statistics for transfers."""
        total = self.transfer_stats['total_attempts']
        if total > 0:
            success_rate = self.transfer_stats['successful_transfers'] / total
            self.transfer_stats['success_rate'] = success_rate
            
            # Update average confidence (simplified)
            # In reality would maintain running average
            self.transfer_stats['average_confidence'] = success_rate * 0.8
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """Get comprehensive transfer statistics."""
        stats = self.transfer_stats.copy()
        
        # Add bridge statistics
        bridge_stats = {}
        for bridge_key, bridge in self.domain_bridges.items():
            bridge_stats[bridge_key] = {
                'strength': bridge.bridge_strength,
                'verified_transfers': len(bridge.verified_transfers),
                'failed_transfers': len(bridge.failed_transfers),
            }
        
        stats['bridges'] = bridge_stats
        return stats
    
    def visualize_pattern_space(self, patterns: List[TransferPattern], output_path: str = None):
        """Visualize patterns in 2D space using t-SNE."""
        if len(patterns) < 2:
            logger.warning("Need at least 2 patterns for visualization")
            return
        
        # Extract pattern vectors
        vectors = np.array([p.pattern_vector for p in patterns])
        domains = [p.source_domain for p in patterns]
        
        # Reduce dimensionality
        if vectors.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            vectors_2d = tsne.fit_transform(vectors)
        else:
            vectors_2d = vectors
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Color by domain
        unique_domains = list(set(domains))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_domains)))
        
        for i, domain in enumerate(unique_domains):
            mask = np.array(domains) == domain
            plt.scatter(
                vectors_2d[mask, 0], 
                vectors_2d[mask, 1],
                c=[colors[i]], 
                label=domain,
                alpha=0.7,
                s=100
            )
        
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('Pattern Transfer Space Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pattern space visualization saved to {output_path}")
        else:
            plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Create pattern transfer system
    transfer_system = PatternTransfer()
    
    # Example: Create a tennis defensive pattern
    tennis_pattern = TransferPattern(
        id="tennis_defensive_001",
        source_domain="tennis",
        target_domain="driving",
        pattern_vector=np.random.rand(64),
        context_vector=np.random.rand(32),
        success_probability=0.85,
        transfer_confidence=0.0,
        biometric_signature=np.random.rand(15),
        temporal_sequence=[np.random.rand(8) for _ in range(20)],
        metadata={
            'scenario': 'defensive_backhand',
            'stress_level': 0.7,
            'opponent_pressure': 0.8
        }
    )
    
    # Transfer to driving domain
    target_context = np.random.rand(32)  # Current driving context
    target_biometrics = np.random.rand(15)  # Current biometric state
    
    result = transfer_system.transfer_pattern(
        tennis_pattern,
        "driving",
        target_context,
        target_biometrics
    )
    
    print(f"Transfer successful: {result.success}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Risk assessment: {result.risk_assessment:.2f}")
    
    # Show statistics
    stats = transfer_system.get_transfer_statistics()
    print(f"\nTransfer Statistics: {stats}") 