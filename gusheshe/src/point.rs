//! Point implementation - irreducible semantic content with uncertainty

use crate::types::{Confidence, Evidence, Metadata, ExecutionContext};
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// A Point represents an irreducible unit of semantic content with inherent uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point {
    /// Unique identifier for this point
    pub id: Uuid,
    
    /// The semantic content of this point
    pub content: String,
    
    /// Confidence level in this point's validity (0.0 to 1.0)
    pub confidence: Confidence,
    
    /// When this point was created
    pub timestamp: Instant,
    
    /// How long this point remains valid
    pub validity_window: Duration,
    
    /// Context-specific metadata
    pub metadata: Metadata,
    
    /// Evidence that supports this point
    pub supporting_evidence: Vec<Evidence>,
    
    /// Evidence that challenges this point
    pub challenging_evidence: Vec<Evidence>,
    
    /// Semantic category of this point
    pub category: PointCategory,
    
    /// Processing priority (higher = more urgent)
    pub priority: u8,
}

/// Categories of semantic content that points can represent
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PointCategory {
    /// Immediate sensor observation
    Observation,
    
    /// Inferred state or condition
    Inference,
    
    /// Predicted future state
    Prediction,
    
    /// Goal or intention
    Intention,
    
    /// Constraint or rule
    Constraint,
    
    /// Safety-critical information
    Safety,
    
    /// Historical pattern or learned behavior
    Pattern,
    
    /// Human-provided information
    Human,
}

impl Point {
    /// Create a new point with minimal information
    pub fn new(content: impl Into<String>, confidence: impl Into<Confidence>) -> Self {
        Self {
            id: Uuid::new_v4(),
            content: content.into(),
            confidence: confidence.into(),
            timestamp: Instant::now(),
            validity_window: Duration::from_secs(60), // Default 1 minute validity
            metadata: Metadata::new(),
            supporting_evidence: Vec::new(),
            challenging_evidence: Vec::new(),
            category: PointCategory::Observation,
            priority: 128, // Mid-range priority
        }
    }

    /// Check if this point is still valid based on its timestamp and validity window
    pub fn is_valid(&self) -> bool {
        self.timestamp.elapsed() <= self.validity_window
    }

    /// Check if this point has expired
    pub fn is_expired(&self) -> bool {
        !self.is_valid()
    }

    /// Get the remaining validity time for this point
    pub fn remaining_validity(&self) -> Duration {
        self.validity_window.saturating_sub(self.timestamp.elapsed())
    }

    /// Add supporting evidence to this point
    pub fn add_supporting_evidence(&mut self, evidence: Evidence) {
        self.supporting_evidence.push(evidence);
        self.recalculate_confidence();
    }

    /// Add challenging evidence to this point
    pub fn add_challenging_evidence(&mut self, evidence: Evidence) {
        self.challenging_evidence.push(evidence);
        self.recalculate_confidence();
    }

    /// Calculate aggregate support strength from all supporting evidence
    pub fn support_strength(&self) -> f64 {
        if self.supporting_evidence.is_empty() {
            return 0.0;
        }

        let total_confidence: f64 = self.supporting_evidence
            .iter()
            .map(|e| e.confidence.value())
            .sum();
        
        total_confidence / self.supporting_evidence.len() as f64
    }

    /// Calculate aggregate challenge strength from all challenging evidence
    pub fn challenge_strength(&self) -> f64 {
        if self.challenging_evidence.is_empty() {
            return 0.0;
        }

        let total_confidence: f64 = self.challenging_evidence
            .iter()
            .map(|e| e.confidence.value())
            .sum();
        
        total_confidence / self.challenging_evidence.len() as f64
    }

    /// Recalculate confidence based on supporting and challenging evidence
    fn recalculate_confidence(&mut self) {
        let support = self.support_strength();
        let challenge = self.challenge_strength();
        
        // Bayesian-inspired confidence update
        // If we have both support and challenge, weight them
        let new_confidence = if support > 0.0 && challenge > 0.0 {
            // Weighted combination favoring stronger evidence
            (support * 2.0 - challenge) / 2.0
        } else if support > 0.0 {
            // Only supporting evidence
            (self.confidence.value() + support) / 2.0
        } else if challenge > 0.0 {
            // Only challenging evidence
            self.confidence.value() * (1.0 - challenge)
        } else {
            // No evidence change
            self.confidence.value()
        };

        self.confidence = Confidence::new(new_confidence.clamp(0.0, 1.0));
    }

    /// Check if this point meets the confidence threshold for a given context
    pub fn meets_threshold(&self, context: &ExecutionContext) -> bool {
        self.confidence.value() >= context.confidence_threshold.value()
    }

    /// Create a derived point based on this one (for inference chains)
    pub fn derive(&self, new_content: impl Into<String>, confidence_modifier: f64) -> Point {
        let derived_confidence = self.confidence.value() * confidence_modifier;
        
        Point {
            id: Uuid::new_v4(),
            content: new_content.into(),
            confidence: Confidence::new(derived_confidence),
            timestamp: Instant::now(),
            validity_window: self.validity_window,
            metadata: self.metadata.clone(),
            supporting_evidence: Vec::new(),
            challenging_evidence: Vec::new(),
            category: PointCategory::Inference,
            priority: self.priority,
        }
    }

    /// Merge this point with another compatible point
    pub fn merge_with(&self, other: &Point) -> Result<Point> {
        // Only merge points with similar content or high semantic similarity
        if self.semantic_distance(other) > 0.3 {
            return Err(Error::invalid_input(
                "points", 
                "Points too semantically different to merge"
            ));
        }

        // Combined confidence using probabilistic OR
        let combined_confidence = self.confidence.or(other.confidence);
        
        // Take the more recent timestamp
        let timestamp = if self.timestamp > other.timestamp {
            self.timestamp
        } else {
            other.timestamp
        };

        // Combine content (prefer the higher confidence point's content)
        let content = if self.confidence.value() > other.confidence.value() {
            self.content.clone()
        } else {
            other.content.clone()
        };

        // Merge evidence
        let mut supporting_evidence = self.supporting_evidence.clone();
        supporting_evidence.extend(other.supporting_evidence.clone());
        
        let mut challenging_evidence = self.challenging_evidence.clone();
        challenging_evidence.extend(other.challenging_evidence.clone());

        Ok(Point {
            id: Uuid::new_v4(),
            content,
            confidence: combined_confidence,
            timestamp,
            validity_window: std::cmp::min(self.validity_window, other.validity_window),
            metadata: self.metadata.clone(), // TODO: Merge metadata intelligently
            supporting_evidence,
            challenging_evidence,
            category: self.category.clone(),
            priority: std::cmp::max(self.priority, other.priority),
        })
    }

    /// Calculate semantic distance between two points (0.0 = identical, 1.0 = completely different)
    /// This is a simplified implementation - in practice, would use semantic embeddings
    fn semantic_distance(&self, other: &Point) -> f64 {
        // Simple Jaccard distance based on words
        let self_words: std::collections::HashSet<&str> = self.content.split_whitespace().collect();
        let other_words: std::collections::HashSet<&str> = other.content.split_whitespace().collect();
        
        let intersection = self_words.intersection(&other_words).count();
        let union = self_words.union(&other_words).count();
        
        if union == 0 {
            1.0
        } else {
            1.0 - (intersection as f64 / union as f64)
        }
    }

    /// Convert point to a fuzzy membership value for a given concept
    pub fn fuzzy_membership(&self, concept: &str) -> f64 {
        // Simple keyword-based membership - in practice would use semantic models
        let content_lower = self.content.to_lowercase();
        let concept_lower = concept.to_lowercase();
        
        if content_lower.contains(&concept_lower) {
            self.confidence.value()
        } else {
            0.0
        }
    }

    /// Check if this point contradicts another point
    pub fn contradicts(&self, other: &Point) -> bool {
        // Simple contradiction detection - look for negation words
        let self_content = self.content.to_lowercase();
        let other_content = other.content.to_lowercase();
        
        // If one contains "not" and the other doesn't, potential contradiction
        let self_has_negation = self_content.contains("not") || self_content.contains("no") || self_content.contains("false");
        let other_has_negation = other_content.contains("not") || other_content.contains("no") || other_content.contains("false");
        
        // Basic semantic overlap with opposite polarity
        if self_has_negation != other_has_negation {
            self.semantic_distance(other) < 0.5
        } else {
            false
        }
    }
}

/// Builder for creating Points with fluent API
pub struct PointBuilder {
    content: String,
    confidence: Confidence,
    category: PointCategory,
    validity_window: Duration,
    priority: u8,
    metadata: Metadata,
}

impl PointBuilder {
    /// Start building a new point
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            confidence: Confidence::new(0.5), // Default medium confidence
            category: PointCategory::Observation,
            validity_window: Duration::from_secs(60),
            priority: 128,
            metadata: Metadata::new(),
        }
    }

    /// Set the confidence level
    pub fn confidence(mut self, confidence: impl Into<Confidence>) -> Self {
        self.confidence = confidence.into();
        self
    }

    /// Set the point category
    pub fn category(mut self, category: PointCategory) -> Self {
        self.category = category;
        self
    }

    /// Set the validity window
    pub fn validity_window(mut self, duration: Duration) -> Self {
        self.validity_window = duration;
        self
    }

    /// Set the processing priority
    pub fn priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build the final Point
    pub fn build(self) -> Point {
        Point {
            id: Uuid::new_v4(),
            content: self.content,
            confidence: self.confidence,
            timestamp: Instant::now(),
            validity_window: self.validity_window,
            metadata: self.metadata,
            supporting_evidence: Vec::new(),
            challenging_evidence: Vec::new(),
            category: self.category,
            priority: self.priority,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_creation() {
        let point = Point::new("test content", 0.8);
        assert_eq!(point.content, "test content");
        assert_eq!(point.confidence.value(), 0.8);
        assert!(point.is_valid());
    }

    #[test]
    fn test_point_builder() {
        let point = PointBuilder::new("test content")
            .confidence(0.9)
            .category(PointCategory::Safety)
            .priority(255)
            .metadata("source", "test")
            .build();
        
        assert_eq!(point.confidence.value(), 0.9);
        assert_eq!(point.category, PointCategory::Safety);
        assert_eq!(point.priority, 255);
    }

    #[test]
    fn test_point_contradiction() {
        let point1 = Point::new("vehicle is safe", 0.8);
        let point2 = Point::new("vehicle is not safe", 0.7);
        
        assert!(point1.contradicts(&point2));
    }

    #[test]
    fn test_semantic_distance() {
        let point1 = Point::new("safe gap detected", 0.8);
        let point2 = Point::new("gap is safe", 0.7);
        let point3 = Point::new("weather is sunny", 0.9);
        
        assert!(point1.semantic_distance(&point2) < point1.semantic_distance(&point3));
    }
} 