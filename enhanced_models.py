"""
Pydantic models for Enhanced Feature Mapping with LLM Evaluation.

This module defines the data structures for:
- Input JSON (from vector search)
- Enhanced output JSON (with LLM context scores)
- Evaluation summaries and quality metrics
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class MatchQuality(str, Enum):
    """Quality classification for matches based on context score."""
    EXCELLENT = "excellent"           # ≥85
    GOOD = "good"                      # 70-84
    NEEDS_REVIEW = "needs_review"      # 50-69
    POOR = "poor"                      # <50
    NOT_EVALUATED = "not_evaluated"    # Above threshold, auto-accepted


class CandidateMetadata(BaseModel):
    """Metadata for each candidate from vector search."""
    model_config = ConfigDict(extra="allow")  # Allow additional fields

    feature_name: str = Field(..., description="Name of the candidate feature")
    feature_value: Optional[str] = Field(None, description="Value/description of the feature")
    notes: Optional[str] = Field(None, description="Additional semantic notes")
    car_model: Optional[str] = Field(None, description="Car model containing OEM info")

    # Additional fields that exist but should be hidden from LLM
    package_type: Optional[str] = None
    source: Optional[str] = None


class MappedCandidate(BaseModel):
    """A single candidate from vector search results."""
    model_config = ConfigDict(extra="allow")

    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Vector similarity score")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence level")
    metadata: CandidateMetadata

    # Enhanced fields (added by LLM evaluation)
    context_matching_score: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="LLM-assigned semantic matching score (0-100)"
    )
    match_quality: MatchQuality = Field(
        default=MatchQuality.NOT_EVALUATED,
        description="Quality classification"
    )
    evaluated: bool = Field(
        default=False,
        description="Whether this candidate was evaluated by LLM"
    )
    llm_reasoning: Optional[str] = Field(
        None,
        description="LLM's explanation for the score"
    )


class FeatureMapping(BaseModel):
    """A single feature with its candidate mappings."""
    model_config = ConfigDict(extra="allow")

    Feature_Name: str = Field(..., description="The target feature to map")
    mapped_list: List[MappedCandidate] = Field(
        default_factory=list,
        description="List of candidate matches from vector search"
    )

    # Best match selection (added after evaluation)
    best_match_index: Optional[int] = Field(
        None,
        description="Index of the best match in mapped_list"
    )
    best_match_score: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Score of the best match (context or similarity-based)"
    )
    best_match_type: Optional[str] = Field(
        None,
        description="Type of score used: 'context' or 'similarity'"
    )


class EvaluationSummary(BaseModel):
    """Summary statistics for the evaluation process."""
    total_features: int = Field(..., description="Total number of features processed")
    total_candidates: int = Field(..., description="Total candidates across all features")
    candidates_evaluated: int = Field(..., description="Number of candidates sent to LLM")
    candidates_auto_accepted: int = Field(..., description="High similarity candidates (≥threshold)")

    # Quality breakdown
    features_with_excellent_matches: int = Field(
        0,
        description="Features with context score ≥85"
    )
    features_with_good_matches: int = Field(
        0,
        description="Features with context score 70-84"
    )
    features_needing_review: int = Field(
        0,
        description="Features with context score 50-69"
    )
    features_with_poor_matches: int = Field(
        0,
        description="Features with context score <50"
    )

    # Cost estimation
    estimated_llm_calls: int = Field(..., description="Number of LLM API calls made")
    estimated_cost_usd: float = Field(..., description="Estimated cost in USD")


class QualityThresholds(BaseModel):
    """Configurable thresholds for quality classification."""
    auto_accept: int = Field(
        default=85,
        ge=0,
        le=100,
        description="Score ≥ this value = excellent match"
    )
    manual_review: int = Field(
        default=70,
        ge=0,
        le=100,
        description="Score between this and auto_accept = needs review"
    )
    reject: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Score < this value = poor match"
    )


class EnhancedMappingResult(BaseModel):
    """Complete result with enhanced mappings and summary."""
    features: List[FeatureMapping] = Field(
        default_factory=list,
        description="All features with enhanced candidate evaluations"
    )
    EvaluationSummary: EvaluationSummary
    FeaturesRequiringReview: List[str] = Field(
        default_factory=list,
        description="List of feature names that need manual review"
    )

    # Configuration used
    configuration: Dict[str, Any] = Field(
        default_factory=dict,
        description="Settings used for this evaluation"
    )
