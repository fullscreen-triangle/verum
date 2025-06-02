//! Certificate implementation - pre-compiled, verifiable execution units

use crate::types::{Action, Confidence, ExecutionContext, Metadata};
use crate::point::Point;
use crate::resolution::ResolutionOutcome;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// A Certificate represents a pre-compiled, verifiable execution unit
/// that can quickly resolve specific types of points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    /// Unique identifier for this certificate
    pub id: Uuid,
    
    /// Human-readable name for this certificate
    pub name: String,
    
    /// Pattern that this certificate can match
    pub pattern: CertificatePattern,
    
    /// Pre-compiled resolution logic
    pub resolution_logic: ResolutionLogic,
    
    /// When this certificate was created
    pub created_at: Instant,
    
    /// When this certificate expires
    pub expires_at: Instant,
    
    /// Cryptographic signature for verification
    pub signature: String,
    
    /// Metadata for this certificate
    pub metadata: Metadata,
}

/// Pattern matching for certificates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertificatePattern {
    /// Exact string match
    Exact(String),
    
    /// Regular expression pattern
    Regex(String),
    
    /// Semantic similarity threshold
    Semantic { concept: String, threshold: f64 },
    
    /// Point category match
    Category(crate::point::PointCategory),
    
    /// Confidence range match
    ConfidenceRange { min: f64, max: f64 },
    
    /// Composite pattern (AND logic)
    And(Vec<CertificatePattern>),
    
    /// Alternative pattern (OR logic)
    Or(Vec<CertificatePattern>),
}

/// Pre-compiled resolution logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionLogic {
    /// Simple action mapping
    DirectAction(Action),
    
    /// Conditional logic
    Conditional {
        conditions: Vec<Condition>,
        default_action: Action,
    },
    
    /// Lookup table based on confidence ranges
    ConfidenceLookup(Vec<(f64, f64, Action)>), // (min, max, action)
    
    /// Formula-based calculation
    Formula {
        confidence_formula: String,
        action_mapping: Vec<(f64, Action)>,
    },
}

/// Condition for conditional resolution logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub check: ConditionCheck,
    pub action: Action,
}

/// Types of condition checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionCheck {
    ConfidenceAbove(f64),
    ConfidenceBelow(f64),
    ConfidenceBetween(f64, f64),
    ContentContains(String),
    CategoryMatches(crate::point::PointCategory),
    TimeConstraint(Duration),
}

impl Certificate {
    /// Check if this certificate can handle the given point
    pub fn can_handle(&self, point: &Point) -> bool {
        if self.is_expired() {
            return false;
        }
        
        self.pattern.matches(point)
    }

    /// Apply this certificate to resolve a point
    pub fn apply(&self, point: &Point, context: &ExecutionContext) -> Result<ResolutionOutcome> {
        if !self.can_handle(point) {
            return Err(Error::invalid_certificate("Certificate cannot handle this point"));
        }

        if self.is_expired() {
            return Err(Error::expired_certificate(self.expires_at));
        }

        let start_time = Instant::now();
        
        // Apply resolution logic
        let action = match &self.resolution_logic {
            ResolutionLogic::DirectAction(action) => action.clone(),
            ResolutionLogic::Conditional { conditions, default_action } => {
                self.apply_conditional_logic(point, context, conditions, default_action)?
            },
            ResolutionLogic::ConfidenceLookup(lookup_table) => {
                self.apply_confidence_lookup(point, lookup_table)?
            },
            ResolutionLogic::Formula { confidence_formula: _, action_mapping } => {
                self.apply_formula_logic(point, action_mapping)?
            },
        };

        // Certificate-based resolutions have high confidence in their pre-compiled logic
        let confidence = Confidence::new(0.95 * point.confidence.value());

        Ok(ResolutionOutcome {
            action,
            confidence,
            reasoning: format!("Certificate '{}' applied", self.name),
            evidence_summary: crate::resolution::EvidenceSummary {
                affirmation_strength: point.confidence.value(),
                contention_strength: 0.0,
                affirmation_count: 1,
                contention_count: 0,
                strongest_affirmation: Some(format!("Certificate: {}", self.name)),
                strongest_contention: None,
                evidence_quality: 0.95,
            },
            timestamp: Instant::now(),
            processing_time: start_time.elapsed(),
            used_fallback: false,
        })
    }

    /// Check if this certificate has expired
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }

    /// Apply conditional resolution logic
    fn apply_conditional_logic(
        &self,
        point: &Point,
        context: &ExecutionContext,
        conditions: &[Condition],
        default_action: &Action,
    ) -> Result<Action> {
        for condition in conditions {
            if condition.check.evaluate(point, context) {
                return Ok(condition.action.clone());
            }
        }
        Ok(default_action.clone())
    }

    /// Apply confidence lookup table
    fn apply_confidence_lookup(&self, point: &Point, lookup_table: &[(f64, f64, Action)]) -> Result<Action> {
        let confidence = point.confidence.value();
        
        for (min, max, action) in lookup_table {
            if confidence >= *min && confidence <= *max {
                return Ok(action.clone());
            }
        }
        
        // Default fallback
        Ok(Action::Maintain)
    }

    /// Apply formula-based logic (simplified)
    fn apply_formula_logic(&self, point: &Point, action_mapping: &[(f64, Action)]) -> Result<Action> {
        // Simplified formula application - just use point confidence
        let confidence = point.confidence.value();
        
        // Find the action mapping with the closest confidence threshold
        let mut best_action = &Action::Maintain;
        let mut best_diff = f64::INFINITY;
        
        for (threshold, action) in action_mapping {
            let diff = (confidence - threshold).abs();
            if diff < best_diff {
                best_diff = diff;
                best_action = action;
            }
        }
        
        Ok(best_action.clone())
    }
}

impl CertificatePattern {
    /// Check if this pattern matches the given point
    pub fn matches(&self, point: &Point) -> bool {
        match self {
            CertificatePattern::Exact(text) => point.content == *text,
            CertificatePattern::Regex(pattern) => {
                // Simplified regex matching - in practice would use regex crate
                point.content.contains(pattern)
            },
            CertificatePattern::Semantic { concept, threshold } => {
                point.fuzzy_membership(concept) >= *threshold
            },
            CertificatePattern::Category(category) => point.category == *category,
            CertificatePattern::ConfidenceRange { min, max } => {
                let conf = point.confidence.value();
                conf >= *min && conf <= *max
            },
            CertificatePattern::And(patterns) => {
                patterns.iter().all(|p| p.matches(point))
            },
            CertificatePattern::Or(patterns) => {
                patterns.iter().any(|p| p.matches(point))
            },
        }
    }
}

impl ConditionCheck {
    /// Evaluate this condition against a point and context
    pub fn evaluate(&self, point: &Point, context: &ExecutionContext) -> bool {
        match self {
            ConditionCheck::ConfidenceAbove(threshold) => point.confidence.value() > *threshold,
            ConditionCheck::ConfidenceBelow(threshold) => point.confidence.value() < *threshold,
            ConditionCheck::ConfidenceBetween(min, max) => {
                let conf = point.confidence.value();
                conf >= *min && conf <= *max
            },
            ConditionCheck::ContentContains(text) => point.content.contains(text),
            ConditionCheck::CategoryMatches(category) => point.category == *category,
            ConditionCheck::TimeConstraint(max_time) => context.remaining_time() >= *max_time,
        }
    }
}

/// Builder for creating certificates
pub struct CertificateBuilder {
    name: String,
    pattern: Option<CertificatePattern>,
    resolution_logic: Option<ResolutionLogic>,
    validity_duration: Duration,
    metadata: Metadata,
}

impl CertificateBuilder {
    /// Start building a new certificate
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            pattern: None,
            resolution_logic: None,
            validity_duration: Duration::from_secs(3600), // 1 hour default
            metadata: Metadata::new(),
        }
    }

    /// Set the pattern for this certificate
    pub fn pattern(mut self, pattern: CertificatePattern) -> Self {
        self.pattern = Some(pattern);
        self
    }

    /// Set the resolution logic
    pub fn resolution_logic(mut self, logic: ResolutionLogic) -> Self {
        self.resolution_logic = Some(logic);
        self
    }

    /// Set the validity duration
    pub fn validity_duration(mut self, duration: Duration) -> Self {
        self.validity_duration = duration;
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build the certificate
    pub fn build(self) -> Result<Certificate> {
        let pattern = self.pattern.ok_or_else(|| Error::invalid_certificate("Pattern is required"))?;
        let resolution_logic = self.resolution_logic.ok_or_else(|| Error::invalid_certificate("Resolution logic is required"))?;
        
        let now = Instant::now();
        
        Ok(Certificate {
            id: Uuid::new_v4(),
            name: self.name,
            pattern,
            resolution_logic,
            created_at: now,
            expires_at: now + self.validity_duration,
            signature: "TODO: implement cryptographic signature".to_string(),
            metadata: self.metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point::PointBuilder;

    #[test]
    fn test_certificate_pattern_matching() {
        let pattern = CertificatePattern::Exact("safe to proceed".to_string());
        let point = PointBuilder::new("safe to proceed").build();
        
        assert!(pattern.matches(&point));
    }

    #[test]
    fn test_certificate_builder() {
        let certificate = CertificateBuilder::new("test_cert")
            .pattern(CertificatePattern::Exact("test".to_string()))
            .resolution_logic(ResolutionLogic::DirectAction(Action::Maintain))
            .build();
        
        assert!(certificate.is_ok());
    }

    #[test]
    fn test_certificate_application() {
        let certificate = CertificateBuilder::new("maintain_cert")
            .pattern(CertificatePattern::Exact("maintain state".to_string()))
            .resolution_logic(ResolutionLogic::DirectAction(Action::Maintain))
            .build()
            .unwrap();
        
        let point = PointBuilder::new("maintain state").confidence(0.8).build();
        let context = ExecutionContext::new(Duration::from_millis(100), Confidence::new(0.6));
        
        let result = certificate.apply(&point, &context);
        assert!(result.is_ok());
        
        let outcome = result.unwrap();
        assert!(matches!(outcome.action, Action::Maintain));
    }
} 