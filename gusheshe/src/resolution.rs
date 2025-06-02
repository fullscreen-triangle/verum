//! Resolution implementation - debate platforms for evidence-based decision making

use crate::types::{Confidence, Action, Affirmation, Contention, ExecutionContext, ProcessingMode};
use crate::point::Point;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// A Resolution represents a debate platform that processes affirmations and contentions
/// to reach a probabilistic decision about a Point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resolution {
    /// Unique identifier for this resolution
    pub id: Uuid,
    
    /// The point being resolved
    pub point_id: Uuid,
    
    /// Affirmations (supporting evidence)
    pub affirmations: Vec<Affirmation>,
    
    /// Contentions (challenging evidence)
    pub contentions: Vec<Contention>,
    
    /// The outcome of this resolution
    pub outcome: Option<ResolutionOutcome>,
    
    /// Strategy used for resolution
    pub strategy: ResolutionStrategy,
    
    /// When this resolution was created
    pub timestamp: Instant,
    
    /// Maximum time allowed for resolution
    pub timeout: Duration,
    
    /// Processing history for this resolution
    pub processing_history: Vec<ProcessingStep>,
}

/// The outcome of a resolution process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionOutcome {
    /// The recommended action
    pub action: Action,
    
    /// Confidence in this resolution
    pub confidence: Confidence,
    
    /// Detailed reasoning for the decision
    pub reasoning: String,
    
    /// Supporting evidence summary
    pub evidence_summary: EvidenceSummary,
    
    /// When this outcome was reached
    pub timestamp: Instant,
    
    /// How long it took to reach this resolution
    pub processing_time: Duration,
    
    /// Whether this resolution required fallback to emergency mode
    pub used_fallback: bool,
}

/// Strategies for resolving debates between affirmations and contentions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// Bayesian inference with formal probability updating
    Bayesian,
    
    /// Choose the most likely interpretation given evidence
    MaximumLikelihood,
    
    /// Conservative approach - err on side of caution
    Conservative,
    
    /// Exploratory - maintain multiple hypotheses
    Exploratory,
    
    /// Emergency mode - simplified, fast decision making
    Emergency,
    
    /// Adaptive - choose strategy based on context
    Adaptive,
}

/// Summary of evidence used in resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSummary {
    /// Total strength of affirmations
    pub affirmation_strength: f64,
    
    /// Total strength of contentions
    pub contention_strength: f64,
    
    /// Number of affirmations
    pub affirmation_count: usize,
    
    /// Number of contentions  
    pub contention_count: usize,
    
    /// Strongest piece of supporting evidence
    pub strongest_affirmation: Option<String>,
    
    /// Strongest piece of challenging evidence
    pub strongest_contention: Option<String>,
    
    /// Evidence quality score (0.0-1.0)
    pub evidence_quality: f64,
}

/// A step in the resolution processing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStep {
    /// Type of processing step
    pub step_type: ProcessingStepType,
    
    /// When this step occurred
    pub timestamp: Instant,
    
    /// How long this step took
    pub duration: Duration,
    
    /// Result of this step
    pub result: String,
    
    /// Processing mode used
    pub mode: ProcessingMode,
}

/// Types of processing steps in resolution
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingStepType {
    /// Initial evidence gathering
    EvidenceGathering,
    
    /// Affirmation analysis
    AffirmationAnalysis,
    
    /// Contention analysis
    ContentionAnalysis,
    
    /// Conflict detection
    ConflictDetection,
    
    /// Bayesian inference
    BayesianInference,
    
    /// Fuzzy logic processing
    FuzzyProcessing,
    
    /// Decision synthesis
    DecisionSynthesis,
    
    /// Fallback triggered
    FallbackTriggered,
}

impl Resolution {
    /// Create a new resolution for a given point
    pub fn new(point: &Point, strategy: ResolutionStrategy, timeout: Duration) -> Self {
        Self {
            id: Uuid::new_v4(),
            point_id: point.id,
            affirmations: Vec::new(),
            contentions: Vec::new(),
            outcome: None,
            strategy,
            timestamp: Instant::now(),
            timeout,
            processing_history: Vec::new(),
        }
    }

    /// Add an affirmation to this resolution
    pub fn add_affirmation(&mut self, affirmation: Affirmation) {
        self.affirmations.push(affirmation);
    }

    /// Add a contention to this resolution
    pub fn add_contention(&mut self, contention: Contention) {
        self.contentions.push(contention);
    }

    /// Process the resolution using the specified strategy
    pub async fn resolve(&mut self, context: &ExecutionContext) -> Result<ResolutionOutcome> {
        let start_time = Instant::now();
        
        // Check if we have timed out before starting
        if context.is_timed_out() {
            return Err(Error::timeout(context.timeout));
        }

        // Record evidence gathering step
        self.record_step(ProcessingStepType::EvidenceGathering, start_time, context.processing_mode.clone());

        // Analyze affirmations and contentions
        let affirmation_strength = self.analyze_affirmations(context).await?;
        let contention_strength = self.analyze_contentions(context).await?;

        // Check for timeouts during analysis
        if context.is_timed_out() {
            return self.emergency_fallback("Timeout during evidence analysis");
        }

        // Detect and handle conflicts
        let conflicts = self.detect_conflicts().await?;
        if !conflicts.is_empty() && context.processing_mode != ProcessingMode::Emergency {
            self.record_step(ProcessingStepType::ConflictDetection, start_time, context.processing_mode.clone());
            
            // Try to resolve conflicts if we have time
            if context.has_time_for(Duration::from_millis(20)) {
                self.resolve_conflicts(&conflicts, context).await?;
            }
        }

        // Apply resolution strategy
        let outcome = match self.strategy {
            ResolutionStrategy::Bayesian => {
                self.bayesian_resolution(affirmation_strength, contention_strength, context).await
            },
            ResolutionStrategy::MaximumLikelihood => {
                self.maximum_likelihood_resolution(affirmation_strength, contention_strength, context).await
            },
            ResolutionStrategy::Conservative => {
                self.conservative_resolution(affirmation_strength, contention_strength, context).await
            },
            ResolutionStrategy::Exploratory => {
                self.exploratory_resolution(affirmation_strength, contention_strength, context).await
            },
            ResolutionStrategy::Emergency => {
                self.emergency_resolution(affirmation_strength, contention_strength, context).await
            },
            ResolutionStrategy::Adaptive => {
                self.adaptive_resolution(affirmation_strength, contention_strength, context).await
            },
        };

        let final_outcome = outcome?;
        
        // Check if confidence meets threshold
        if final_outcome.confidence.value() < context.confidence_threshold.value() {
            // If confidence is too low, try fallback
            if context.processing_mode != ProcessingMode::Emergency {
                return self.emergency_fallback("Confidence below threshold");
            }
        }

        self.outcome = Some(final_outcome.clone());
        Ok(final_outcome)
    }

    /// Analyze the strength of affirmations
    async fn analyze_affirmations(&mut self, context: &ExecutionContext) -> Result<f64> {
        self.record_step(ProcessingStepType::AffirmationAnalysis, Instant::now(), context.processing_mode.clone());
        
        if self.affirmations.is_empty() {
            return Ok(0.0);
        }

        // Weight affirmations by strength and relevance
        let weighted_sum: f64 = self.affirmations
            .iter()
            .map(|a| a.strength * a.relevance * a.evidence.confidence.value())
            .sum();
        
        let count = self.affirmations.len() as f64;
        Ok(weighted_sum / count)
    }

    /// Analyze the strength of contentions
    async fn analyze_contentions(&mut self, context: &ExecutionContext) -> Result<f64> {
        self.record_step(ProcessingStepType::ContentionAnalysis, Instant::now(), context.processing_mode.clone());
        
        if self.contentions.is_empty() {
            return Ok(0.0);
        }

        // Weight contentions by impact and uncertainty
        let weighted_sum: f64 = self.contentions
            .iter()
            .map(|c| c.impact * c.uncertainty * c.evidence.confidence.value())
            .sum();
        
        let count = self.contentions.len() as f64;
        Ok(weighted_sum / count)
    }

    /// Detect conflicts between evidence pieces
    async fn detect_conflicts(&self) -> Result<Vec<String>> {
        let mut conflicts = Vec::new();
        
        // Simple conflict detection - look for contradictory evidence
        for affirmation in &self.affirmations {
            for contention in &self.contentions {
                // If both have high confidence but opposite implications
                if affirmation.evidence.confidence.value() > 0.7 && 
                   contention.evidence.confidence.value() > 0.7 {
                    let conflict_desc = format!(
                        "High-confidence affirmation '{}' conflicts with high-confidence contention '{}'",
                        affirmation.evidence.content,
                        contention.evidence.content
                    );
                    conflicts.push(conflict_desc);
                }
            }
        }
        
        Ok(conflicts)
    }

    /// Attempt to resolve conflicts between evidence
    async fn resolve_conflicts(&mut self, conflicts: &[String], context: &ExecutionContext) -> Result<()> {
        // For now, just log conflicts - in practice would implement sophisticated conflict resolution
        for conflict in conflicts {
            tracing::warn!("Conflict detected: {}", conflict);
        }
        
        // Could implement evidence re-weighting, additional evidence gathering, etc.
        Ok(())
    }

    /// Bayesian resolution strategy
    async fn bayesian_resolution(&mut self, affirmation_strength: f64, contention_strength: f64, context: &ExecutionContext) -> Result<ResolutionOutcome> {
        self.record_step(ProcessingStepType::BayesianInference, Instant::now(), context.processing_mode.clone());
        
        // Prior probability (could be learned from historical data)
        let prior = 0.5;
        
        // Likelihood of evidence given hypothesis is true/false
        let likelihood_true = affirmation_strength;
        let likelihood_false = contention_strength;
        
        // Bayesian update
        let posterior = (likelihood_true * prior) / 
                       (likelihood_true * prior + likelihood_false * (1.0 - prior));
        
        let confidence = Confidence::new(posterior);
        let action = self.determine_action(confidence, affirmation_strength, contention_strength);
        
        Ok(ResolutionOutcome {
            action,
            confidence,
            reasoning: format!("Bayesian inference: prior={:.3}, posterior={:.3}", prior, posterior),
            evidence_summary: self.create_evidence_summary(affirmation_strength, contention_strength),
            timestamp: Instant::now(),
            processing_time: self.timestamp.elapsed(),
            used_fallback: false,
        })
    }

    /// Maximum likelihood resolution strategy
    async fn maximum_likelihood_resolution(&mut self, affirmation_strength: f64, contention_strength: f64, context: &ExecutionContext) -> Result<ResolutionOutcome> {
        // Choose the hypothesis with maximum likelihood
        let confidence = if affirmation_strength > contention_strength {
            Confidence::new(affirmation_strength)
        } else {
            Confidence::new(1.0 - contention_strength)
        };
        
        let action = self.determine_action(confidence, affirmation_strength, contention_strength);
        
        Ok(ResolutionOutcome {
            action,
            confidence,
            reasoning: format!("Maximum likelihood: affirmations={:.3}, contentions={:.3}", affirmation_strength, contention_strength),
            evidence_summary: self.create_evidence_summary(affirmation_strength, contention_strength),
            timestamp: Instant::now(),
            processing_time: self.timestamp.elapsed(),
            used_fallback: false,
        })
    }

    /// Conservative resolution strategy (err on side of caution)
    async fn conservative_resolution(&mut self, affirmation_strength: f64, contention_strength: f64, context: &ExecutionContext) -> Result<ResolutionOutcome> {
        // Apply conservative bias - require higher confidence for risky actions
        let conservative_bias = 0.8; // Require 80% confidence for action
        
        let raw_confidence = if affirmation_strength > contention_strength {
            affirmation_strength - contention_strength
        } else {
            0.0
        };
        
        let confidence = Confidence::new(raw_confidence * conservative_bias);
        let action = if confidence.value() > context.confidence_threshold.value() {
            self.determine_action(confidence, affirmation_strength, contention_strength)
        } else {
            Action::Maintain // Conservative fallback
        };
        
        Ok(ResolutionOutcome {
            action,
            confidence,
            reasoning: format!("Conservative resolution with bias={:.3}", conservative_bias),
            evidence_summary: self.create_evidence_summary(affirmation_strength, contention_strength),
            timestamp: Instant::now(),
            processing_time: self.timestamp.elapsed(),
            used_fallback: false,
        })
    }

    /// Exploratory resolution strategy (maintain multiple hypotheses)
    async fn exploratory_resolution(&mut self, affirmation_strength: f64, contention_strength: f64, context: &ExecutionContext) -> Result<ResolutionOutcome> {
        // In exploratory mode, we're more willing to accept uncertainty
        let confidence_balance = (affirmation_strength + (1.0 - contention_strength)) / 2.0;
        let confidence = Confidence::new(confidence_balance);
        
        let action = Action::Maintain; // Exploratory mode tends to maintain state while gathering more info
        
        Ok(ResolutionOutcome {
            action,
            confidence,
            reasoning: "Exploratory resolution - maintaining state while evaluating hypotheses".to_string(),
            evidence_summary: self.create_evidence_summary(affirmation_strength, contention_strength),
            timestamp: Instant::now(),
            processing_time: self.timestamp.elapsed(),
            used_fallback: false,
        })
    }

    /// Emergency resolution strategy (fast, simplified decision making)
    async fn emergency_resolution(&mut self, affirmation_strength: f64, contention_strength: f64, context: &ExecutionContext) -> Result<ResolutionOutcome> {
        // Simple, fast heuristic
        let confidence = if affirmation_strength > 0.5 {
            Confidence::new(0.8) // High confidence in emergency mode if any decent affirmation
        } else {
            Confidence::new(0.2) // Low confidence otherwise
        };
        
        // Default to safe action in emergency mode
        let action = Action::Emergency(crate::types::EmergencyAction::EmergencyBrake);
        
        Ok(ResolutionOutcome {
            action,
            confidence,
            reasoning: "Emergency resolution - prioritizing safety".to_string(),
            evidence_summary: self.create_evidence_summary(affirmation_strength, contention_strength),
            timestamp: Instant::now(),
            processing_time: self.timestamp.elapsed(),
            used_fallback: true,
        })
    }

    /// Adaptive resolution strategy (choose strategy based on context)
    async fn adaptive_resolution(&mut self, affirmation_strength: f64, contention_strength: f64, context: &ExecutionContext) -> Result<ResolutionOutcome> {
        // Choose strategy based on available time and evidence quality
        let evidence_quality = (affirmation_strength + contention_strength) / 2.0;
        
        let chosen_strategy = if context.remaining_time() < Duration::from_millis(10) {
            ResolutionStrategy::Emergency
        } else if evidence_quality > 0.8 {
            ResolutionStrategy::Bayesian
        } else if evidence_quality > 0.5 {
            ResolutionStrategy::MaximumLikelihood
        } else {
            ResolutionStrategy::Conservative
        };
        
        // Temporarily change strategy and recurse
        let original_strategy = self.strategy.clone();
        self.strategy = chosen_strategy;
        let result = match self.strategy {
            ResolutionStrategy::Emergency => self.emergency_resolution(affirmation_strength, contention_strength, context).await,
            ResolutionStrategy::Bayesian => self.bayesian_resolution(affirmation_strength, contention_strength, context).await,
            ResolutionStrategy::MaximumLikelihood => self.maximum_likelihood_resolution(affirmation_strength, contention_strength, context).await,
            ResolutionStrategy::Conservative => self.conservative_resolution(affirmation_strength, contention_strength, context).await,
            _ => unreachable!(),
        };
        self.strategy = original_strategy;
        
        result
    }

    /// Emergency fallback when normal resolution fails
    fn emergency_fallback(&mut self, reason: &str) -> Result<ResolutionOutcome> {
        self.record_step(ProcessingStepType::FallbackTriggered, Instant::now(), ProcessingMode::Emergency);
        
        Ok(ResolutionOutcome {
            action: Action::Emergency(crate::types::EmergencyAction::EmergencyBrake),
            confidence: Confidence::new(0.9), // High confidence in safety action
            reasoning: format!("Emergency fallback triggered: {}", reason),
            evidence_summary: self.create_evidence_summary(0.0, 1.0), // Assume high contention
            timestamp: Instant::now(),
            processing_time: self.timestamp.elapsed(),
            used_fallback: true,
        })
    }

    /// Determine the appropriate action based on confidence and evidence
    fn determine_action(&self, confidence: Confidence, affirmation_strength: f64, contention_strength: f64) -> Action {
        if confidence.value() > 0.8 && affirmation_strength > 0.7 {
            // High confidence and strong affirmations - proceed with action
            // This would be determined by the specific point content
            Action::Execute(crate::types::DrivingAction::ChangeLane { 
                direction: crate::types::LaneDirection::Left, 
                urgency: crate::types::Urgency::Medium 
            })
        } else if confidence.value() < 0.3 || contention_strength > 0.7 {
            // Low confidence or strong contentions - maintain current state
            Action::Maintain
        } else {
            // Medium confidence - proceed with caution
            Action::Execute(crate::types::DrivingAction::AdjustSpeed { 
                delta_mph: -5, 
                urgency: crate::types::Urgency::Low 
            })
        }
    }

    /// Create evidence summary for resolution outcome
    fn create_evidence_summary(&self, affirmation_strength: f64, contention_strength: f64) -> EvidenceSummary {
        let strongest_affirmation = self.affirmations
            .iter()
            .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal))
            .map(|a| a.evidence.content.clone());
        
        let strongest_contention = self.contentions
            .iter()
            .max_by(|a, b| a.impact.partial_cmp(&b.impact).unwrap_or(std::cmp::Ordering::Equal))
            .map(|c| c.evidence.content.clone());
        
        let evidence_quality = if self.affirmations.is_empty() && self.contentions.is_empty() {
            0.0
        } else {
            (affirmation_strength + contention_strength) / 2.0
        };
        
        EvidenceSummary {
            affirmation_strength,
            contention_strength,
            affirmation_count: self.affirmations.len(),
            contention_count: self.contentions.len(),
            strongest_affirmation,
            strongest_contention,
            evidence_quality,
        }
    }

    /// Record a processing step
    fn record_step(&mut self, step_type: ProcessingStepType, start_time: Instant, mode: ProcessingMode) {
        let step = ProcessingStep {
            step_type,
            timestamp: Instant::now(),
            duration: start_time.elapsed(),
            result: "Completed".to_string(),
            mode,
        };
        self.processing_history.push(step);
    }

    /// Check if resolution has timed out
    pub fn is_timed_out(&self) -> bool {
        self.timestamp.elapsed() > self.timeout
    }

    /// Get processing time so far
    pub fn processing_time(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point::Point;
    use crate::types::*;

    #[test]
    fn test_resolution_creation() {
        let point = Point::new("test point", 0.8);
        let resolution = Resolution::new(&point, ResolutionStrategy::Bayesian, Duration::from_millis(100));
        
        assert_eq!(resolution.point_id, point.id);
        assert_eq!(resolution.strategy, ResolutionStrategy::Bayesian);
        assert_eq!(resolution.timeout, Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_bayesian_resolution() {
        let point = Point::new("safe to merge", 0.8);
        let mut resolution = Resolution::new(&point, ResolutionStrategy::Bayesian, Duration::from_millis(100));
        
        // Add some test evidence
        let evidence = Evidence {
            id: Uuid::new_v4(),
            content: "Clear gap detected".to_string(),
            evidence_type: EvidenceType::Sensor,
            confidence: Confidence::new(0.9),
            source: "radar".to_string(),
            timestamp: Instant::now(),
            validity_window: Duration::from_secs(5),
        };
        
        resolution.add_affirmation(Affirmation {
            evidence,
            strength: 0.8,
            relevance: 0.9,
        });
        
        let context = ExecutionContext::new(Duration::from_millis(100), Confidence::new(0.6));
        let outcome = resolution.resolve(&context).await;
        
        assert!(outcome.is_ok());
        let outcome = outcome.unwrap();
        assert!(outcome.confidence.value() > 0.0);
    }
} 