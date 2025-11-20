"""
LangChain-specific Pydantic models for structured LLM output.

These models are used with LangChain's with_structured_output() feature
to guarantee valid, type-safe responses from the LLM.
"""

from pydantic import BaseModel, Field


class BiasFreeCandidateInput(BaseModel):
    """
    Sanitized candidate data shown to LLM - bias removed.

    We HIDE these fields to prevent anchoring bias:
    - similarity_score (LLM might trust vector search too much)
    - confidence (same reason)
    - source (irrelevant for semantic matching)
    - package_type (irrelevant for semantic matching)

    We SHOW only semantic data:
    - feature_name
    - feature_value
    - notes (semantic context)
    - oem (extracted from car_model)
    """
    feature_name: str = Field(..., description="Name of the candidate feature")
    feature_value: str = Field(..., description="Description or value of the feature")
    notes: str = Field(default="", description="Additional semantic notes or context")
    oem: str = Field(default="", description="OEM/manufacturer extracted from car model")


class CandidateEvaluation(BaseModel):
    """
    Structured output from LLM evaluation.

    LangChain guarantees this schema is followed exactly,
    with automatic retries on validation errors.
    """
    context_matching_score: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Semantic similarity score (0-100) where:\n"
            "- 90-100: Perfect semantic match (same concept, different wording)\n"
            "- 80-89: Very strong match (closely related concepts)\n"
            "- 70-79: Good match (related with clear connection)\n"
            "- 60-69: Moderate match (some relationship exists)\n"
            "- 40-59: Weak match (tangential relationship)\n"
            "- 20-39: Very weak match (minimal connection)\n"
            "- 0-19: No meaningful relationship"
        )
    )

    reasoning: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description=(
            "Clear explanation of why this score was assigned. "
            "Focus on semantic relationships, concept overlap, "
            "and domain-specific knowledge. Be concise but specific."
        )
    )

    key_factors: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description=(
            "1-5 key factors that influenced the score. "
            "Examples: 'Same measurement unit', 'Synonym relationship', "
            "'Different domains', 'No semantic overlap'"
        )
    )


class BatchEvaluationRequest(BaseModel):
    """
    Request structure for batch candidate evaluation.
    Used when evaluating multiple candidates for a single target feature.
    """
    target_feature_name: str = Field(..., description="The feature we're trying to match")
    target_feature_context: str = Field(
        default="",
        description="Additional context about the target feature"
    )
    candidates: list[BiasFreeCandidateInput] = Field(
        ...,
        min_length=1,
        description="List of candidates to evaluate"
    )


class BatchEvaluationResult(BaseModel):
    """
    Batch evaluation result - one evaluation per candidate.
    Currently not used (we evaluate one at a time), but available for future optimization.
    """
    evaluations: list[CandidateEvaluation] = Field(
        ...,
        description="One evaluation per candidate, in same order as input"
    )
