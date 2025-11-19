#!/usr/bin/env python3
"""
Enhanced Feature Mapper with LangChain and AWS Bedrock.

This script evaluates feature mappings using:
1. Vector similarity scores (from Pinecone or similar)
2. LLM semantic evaluation (via AWS Bedrock + LangChain)

Key Strategy:
- Candidates with similarity_score ≥ threshold → Auto-accept (no LLM call)
- Candidates with similarity_score < threshold → LLM evaluation (find semantic matches)

This approach:
✅ Saves costs (fewer LLM calls)
✅ Finds hidden matches (low vector similarity but high semantic match)
✅ Removes bias (LLM doesn't see similarity scores)
"""

import json
import logging
import sys
import time
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import boto3
from langchain_aws import ChatBedrock
from pydantic import ValidationError

from enhanced_models import (
    FeatureMapping,
    MappedCandidate,
    EnhancedMappingResult,
    EvaluationSummary,
    MatchQuality,
    QualityThresholds,
)
from langchain_models import (
    CandidateEvaluation,
    BiasFreeCandidateInput,
)


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure comprehensive logging with colored output.

    Args:
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("EnhancedMapper")
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Detailed format with timestamp
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# ============================================================================
# LANGCHAIN FEATURE MAPPER
# ============================================================================

class LangChainFeatureMapper:
    """
    Feature mapper using LangChain + AWS Bedrock for semantic evaluation.

    Architecture:
    1. Split candidates by similarity threshold
    2. Auto-accept high similarity candidates (≥threshold)
    3. Evaluate low similarity candidates with LLM (<threshold)
    4. Merge results and select best match
    """

    # Configuration
    DEFAULT_SIMILARITY_THRESHOLD = 0.85
    DEFAULT_MAX_CANDIDATES = 10
    DEFAULT_MAX_RETRIES = 3
    COST_PER_EVALUATION = 0.0003  # Estimated cost in USD

    def __init__(
        self,
        aws_region: str = "eu-central-1",
        model_id: str = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_candidates_to_evaluate: int = DEFAULT_MAX_CANDIDATES,
        quality_thresholds: Optional[QualityThresholds] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the LangChain Feature Mapper.

        Args:
            aws_region: AWS region for Bedrock (default: eu-central-1)
            model_id: Bedrock model ID (default: Claude 3.7 Sonnet)
            similarity_threshold: Split point for evaluation (default: 0.85)
            max_candidates_to_evaluate: Max candidates to send to LLM (default: 10)
            quality_thresholds: Custom quality thresholds
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)

        self.aws_region = aws_region
        self.model_id = model_id
        self.similarity_threshold = similarity_threshold
        self.max_candidates_to_evaluate = max_candidates_to_evaluate
        self.quality_thresholds = quality_thresholds or QualityThresholds()

        # Statistics
        self.stats = {
            "total_features": 0,
            "total_candidates": 0,
            "candidates_evaluated": 0,
            "candidates_auto_accepted": 0,
            "llm_calls": 0,
            "llm_errors": 0,
        }

        self.logger.info("=" * 80)
        self.logger.info("🚀 LangChain Feature Mapper Initialized")
        self.logger.info("=" * 80)
        self.logger.info(f"📍 AWS Region: {self.aws_region}")
        self.logger.info(f"🤖 Model: {self.model_id}")
        self.logger.info(f"📊 Similarity Threshold: {self.similarity_threshold}")
        self.logger.info(f"🎯 Max Candidates to Evaluate: {self.max_candidates_to_evaluate}")
        self.logger.info(f"✨ Quality Thresholds: Auto-accept={self.quality_thresholds.auto_accept}, "
                        f"Review={self.quality_thresholds.manual_review}, "
                        f"Reject={self.quality_thresholds.reject}")
        self.logger.info("=" * 80)

        # Initialize LangChain with Bedrock
        self._initialize_langchain()

    def _initialize_langchain(self) -> None:
        """Initialize LangChain with AWS Bedrock backend."""
        try:
            self.logger.info("🔧 Initializing LangChain with AWS Bedrock...")

            # Create Bedrock client using AWS config credentials
            bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.aws_region,
            )

            # Create LangChain ChatBedrock instance
            self.llm = ChatBedrock(
                client=bedrock_client,
                model_id=self.model_id,
                model_kwargs={
                    "temperature": 0.1,  # Low temperature for consistent evaluation
                    "top_p": 0.9,
                    "max_tokens": 1000,
                },
            )

            # Create structured output LLM
            self.structured_llm = self.llm.with_structured_output(CandidateEvaluation)

            self.logger.info("✅ LangChain initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LangChain: {e}")
            raise

    def _extract_oem_from_car_model(self, car_model: Optional[str]) -> str:
        """
        Extract OEM/manufacturer from car model string.

        Examples:
            "BMW 3 Series" → "BMW"
            "Mercedes-Benz C-Class" → "Mercedes-Benz"
            "Toyota Camry 2023" → "Toyota"

        Args:
            car_model: Car model string

        Returns:
            Extracted OEM or empty string
        """
        if not car_model:
            return ""

        # Simple extraction: take first word/phrase before space or number
        match = re.match(r'^([A-Za-z\-]+)', car_model.strip())
        return match.group(1) if match else car_model.split()[0] if car_model.split() else ""

    def _create_bias_free_candidate(self, candidate: MappedCandidate) -> BiasFreeCandidateInput:
        """
        Create bias-free candidate data for LLM evaluation.

        CRITICAL: We hide similarity_score, confidence, source, package_type
        to prevent anchoring bias. LLM sees only semantic data.

        Args:
            candidate: Full candidate with all fields

        Returns:
            Sanitized candidate for LLM
        """
        return BiasFreeCandidateInput(
            feature_name=candidate.metadata.feature_name,
            feature_value=candidate.metadata.feature_value or "",
            notes=candidate.metadata.notes or "",
            oem=self._extract_oem_from_car_model(candidate.metadata.car_model),
        )

    def _build_evaluation_prompt(
        self,
        target_feature_name: str,
        candidate: BiasFreeCandidateInput,
    ) -> str:
        """
        Build the prompt for LLM evaluation.

        Args:
            target_feature_name: The feature we're trying to match
            candidate: Bias-free candidate data

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert in automotive feature mapping and semantic similarity evaluation.

**TASK**: Evaluate how well the CANDIDATE feature matches the TARGET feature based on SEMANTIC MEANING only.

**TARGET FEATURE**:
Name: {target_feature_name}

**CANDIDATE FEATURE**:
Name: {candidate.feature_name}
Value: {candidate.feature_value}
Notes: {candidate.notes}
OEM: {candidate.oem}

**EVALUATION CRITERIA**:
- Focus on SEMANTIC meaning, not string similarity
- Consider synonyms (e.g., "Engine Capacity" = "Displacement")
- Consider unit conversions (e.g., "cc" = "cm³")
- Consider domain knowledge (automotive context)
- Ignore irrelevant fields like "Maps", "Navigation" when target is mechanical

**SCORING GUIDE**:
- 90-100: Perfect semantic match (same concept, different wording)
- 80-89: Very strong match (closely related concepts)
- 70-79: Good match (related with clear connection)
- 60-69: Moderate match (some relationship exists)
- 40-59: Weak match (tangential relationship)
- 20-39: Very weak match (minimal connection)
- 0-19: No meaningful relationship

**IMPORTANT**:
- Be objective and precise
- Don't be influenced by any external factors
- Base your score purely on semantic relationships
- Provide clear reasoning for your score

Evaluate the candidate and provide:
1. A context_matching_score (0-100)
2. Clear reasoning for your score
3. Key factors that influenced your decision (1-5 factors)
"""
        return prompt

    def _evaluate_candidate_with_llm(
        self,
        target_feature_name: str,
        candidate: MappedCandidate,
    ) -> CandidateEvaluation:
        """
        Evaluate a single candidate using LLM with retries.

        Args:
            target_feature_name: Target feature name
            candidate: Candidate to evaluate

        Returns:
            CandidateEvaluation with score and reasoning

        Raises:
            Exception: If all retries fail
        """
        bias_free_candidate = self._create_bias_free_candidate(candidate)
        prompt = self._build_evaluation_prompt(target_feature_name, bias_free_candidate)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"🔄 LLM evaluation attempt {attempt + 1}/{max_retries}")

                # Invoke LLM with structured output
                response: CandidateEvaluation = self.structured_llm.invoke(prompt)

                self.stats["llm_calls"] += 1

                self.logger.debug(f"✅ LLM returned score: {response.context_matching_score}")
                return response

            except ValidationError as e:
                self.logger.warning(f"⚠️ Validation error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    self.stats["llm_errors"] += 1
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

            except Exception as e:
                self.logger.error(f"❌ LLM error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    self.stats["llm_errors"] += 1
                    raise
                time.sleep(2 ** attempt)

        # Should never reach here
        raise Exception("Failed to evaluate candidate after all retries")

    def _classify_quality(self, score: int) -> MatchQuality:
        """
        Classify match quality based on context score.

        Args:
            score: Context matching score (0-100)

        Returns:
            MatchQuality enum value
        """
        if score >= self.quality_thresholds.auto_accept:
            return MatchQuality.EXCELLENT
        elif score >= self.quality_thresholds.manual_review:
            return MatchQuality.GOOD
        elif score >= self.quality_thresholds.reject:
            return MatchQuality.NEEDS_REVIEW
        else:
            return MatchQuality.POOR

    def _process_single_feature(self, feature: FeatureMapping) -> FeatureMapping:
        """
        Process a single feature: split candidates, evaluate, merge.

        Args:
            feature: Feature with candidates to evaluate

        Returns:
            Enhanced feature with context scores
        """
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"🎯 Processing Feature: {feature.Feature_Name}")
        self.logger.info("=" * 80)

        self.stats["total_features"] += 1
        self.stats["total_candidates"] += len(feature.mapped_list)

        if not feature.mapped_list:
            self.logger.warning("⚠️ No candidates found for this feature")
            return feature

        # Split candidates by threshold
        above_threshold = []
        below_threshold = []

        for candidate in feature.mapped_list:
            if candidate.similarity_score >= self.similarity_threshold:
                above_threshold.append(candidate)
            else:
                below_threshold.append(candidate)

        self.logger.info(f"📊 Total candidates: {len(feature.mapped_list)}")
        self.logger.info(f"✅ Above threshold (≥{self.similarity_threshold}): {len(above_threshold)} → Auto-accept")
        self.logger.info(f"🔍 Below threshold (<{self.similarity_threshold}): {len(below_threshold)} → LLM evaluation")

        # Process above-threshold candidates (auto-accept)
        for candidate in above_threshold:
            candidate.evaluated = False
            candidate.match_quality = MatchQuality.NOT_EVALUATED
            candidate.llm_reasoning = f"High similarity ({candidate.similarity_score:.3f}) - auto-accepted without LLM evaluation"
            self.stats["candidates_auto_accepted"] += 1

        # Process below-threshold candidates (LLM evaluation)
        candidates_to_evaluate = below_threshold[:self.max_candidates_to_evaluate]

        if candidates_to_evaluate:
            self.logger.info(f"🤖 Evaluating {len(candidates_to_evaluate)} candidates with LLM...")

        for idx, candidate in enumerate(candidates_to_evaluate, 1):
            try:
                self.logger.info(f"  [{idx}/{len(candidates_to_evaluate)}] Evaluating: {candidate.metadata.feature_name} "
                               f"(similarity: {candidate.similarity_score:.3f})")

                # Evaluate with LLM
                evaluation = self._evaluate_candidate_with_llm(
                    target_feature_name=feature.Feature_Name,
                    candidate=candidate,
                )

                # Update candidate with evaluation results
                candidate.context_matching_score = evaluation.context_matching_score
                candidate.match_quality = self._classify_quality(evaluation.context_matching_score)
                candidate.evaluated = True
                candidate.llm_reasoning = evaluation.reasoning

                self.stats["candidates_evaluated"] += 1

                # Log result
                quality_emoji = {
                    MatchQuality.EXCELLENT: "🌟",
                    MatchQuality.GOOD: "👍",
                    MatchQuality.NEEDS_REVIEW: "⚠️",
                    MatchQuality.POOR: "👎",
                }
                emoji = quality_emoji.get(candidate.match_quality, "❓")

                self.logger.info(f"      {emoji} Score: {candidate.context_matching_score}/100 ({candidate.match_quality.value})")
                self.logger.debug(f"      💭 Reasoning: {evaluation.reasoning[:100]}...")

                # Special highlight for semantic matches despite low similarity
                if (candidate.similarity_score < self.similarity_threshold and
                    candidate.context_matching_score >= self.quality_thresholds.auto_accept):
                    self.logger.info(f"      🎯 Found semantic match despite low similarity!")

            except Exception as e:
                self.logger.error(f"      ❌ Failed to evaluate candidate: {e}")
                # Fallback: assign neutral score
                candidate.context_matching_score = 50
                candidate.match_quality = MatchQuality.NEEDS_REVIEW
                candidate.evaluated = True
                candidate.llm_reasoning = f"Evaluation failed: {str(e)}"

        # Select best match
        self._select_best_match(feature)

        return feature

    def _select_best_match(self, feature: FeatureMapping) -> None:
        """
        Select the best match from evaluated candidates.

        Priority:
        1. Highest context_matching_score (if evaluated)
        2. Highest similarity_score (if not evaluated)

        Args:
            feature: Feature to update with best match
        """
        if not feature.mapped_list:
            return

        best_idx = 0
        best_score = -1
        best_type = "similarity"

        for idx, candidate in enumerate(feature.mapped_list):
            # Prefer context score if available
            if candidate.context_matching_score is not None:
                if candidate.context_matching_score > best_score:
                    best_score = candidate.context_matching_score
                    best_idx = idx
                    best_type = "context"
            else:
                # Fall back to similarity score
                similarity_as_score = int(candidate.similarity_score * 100)
                if similarity_as_score > best_score:
                    best_score = similarity_as_score
                    best_idx = idx
                    best_type = "similarity"

        feature.best_match_index = best_idx
        feature.best_match_score = best_score
        feature.best_match_type = best_type

        best_candidate = feature.mapped_list[best_idx]
        self.logger.info(f"🏆 Best Match: {best_candidate.metadata.feature_name} "
                        f"(score: {best_score}, type: {best_type})")

    def process_json(self, input_data: Dict[str, Any]) -> EnhancedMappingResult:
        """
        Process input JSON and return enhanced results.

        Args:
            input_data: Input JSON data (list of features or dict with features)

        Returns:
            EnhancedMappingResult with evaluations and summary
        """
        self.logger.info("")
        self.logger.info("🚀 Starting feature mapping process...")

        # Parse input
        if isinstance(input_data, list):
            features_data = input_data
        elif isinstance(input_data, dict) and "features" in input_data:
            features_data = input_data["features"]
        else:
            raise ValueError("Input must be a list of features or dict with 'features' key")

        # Convert to FeatureMapping objects
        features = [FeatureMapping(**f) for f in features_data]

        # Process each feature
        processed_features = []
        for feature in features:
            processed_feature = self._process_single_feature(feature)
            processed_features.append(processed_feature)

        # Build summary
        summary = self._build_summary(processed_features)

        # Identify features requiring review
        features_requiring_review = []
        for feature in processed_features:
            if feature.best_match_index is not None:
                best_candidate = feature.mapped_list[feature.best_match_index]
                if best_candidate.match_quality == MatchQuality.NEEDS_REVIEW:
                    features_requiring_review.append(feature.Feature_Name)

        # Build result
        result = EnhancedMappingResult(
            features=processed_features,
            EvaluationSummary=summary,
            FeaturesRequiringReview=features_requiring_review,
            configuration={
                "similarity_threshold": self.similarity_threshold,
                "max_candidates_to_evaluate": self.max_candidates_to_evaluate,
                "quality_thresholds": self.quality_thresholds.model_dump(),
                "aws_region": self.aws_region,
                "model_id": self.model_id,
            },
        )

        # Log final summary
        self._log_final_summary(summary)

        return result

    def _build_summary(self, features: List[FeatureMapping]) -> EvaluationSummary:
        """Build evaluation summary from processed features."""
        excellent_count = 0
        good_count = 0
        review_count = 0
        poor_count = 0

        for feature in features:
            if feature.best_match_index is not None:
                best = feature.mapped_list[feature.best_match_index]
                if best.match_quality == MatchQuality.EXCELLENT:
                    excellent_count += 1
                elif best.match_quality == MatchQuality.GOOD:
                    good_count += 1
                elif best.match_quality == MatchQuality.NEEDS_REVIEW:
                    review_count += 1
                elif best.match_quality == MatchQuality.POOR:
                    poor_count += 1

        estimated_cost = self.stats["llm_calls"] * self.COST_PER_EVALUATION

        return EvaluationSummary(
            total_features=self.stats["total_features"],
            total_candidates=self.stats["total_candidates"],
            candidates_evaluated=self.stats["candidates_evaluated"],
            candidates_auto_accepted=self.stats["candidates_auto_accepted"],
            features_with_excellent_matches=excellent_count,
            features_with_good_matches=good_count,
            features_needing_review=review_count,
            features_with_poor_matches=poor_count,
            estimated_llm_calls=self.stats["llm_calls"],
            estimated_cost_usd=round(estimated_cost, 4),
        )

    def _log_final_summary(self, summary: EvaluationSummary) -> None:
        """Log final summary statistics."""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("📊 EVALUATION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"✅ Total Features Processed: {summary.total_features}")
        self.logger.info(f"📦 Total Candidates: {summary.total_candidates}")
        self.logger.info(f"🤖 Candidates Evaluated by LLM: {summary.candidates_evaluated}")
        self.logger.info(f"⚡ Candidates Auto-Accepted: {summary.candidates_auto_accepted}")
        self.logger.info("")
        self.logger.info("🎯 Quality Breakdown:")
        self.logger.info(f"  🌟 Excellent Matches: {summary.features_with_excellent_matches}")
        self.logger.info(f"  👍 Good Matches: {summary.features_with_good_matches}")
        self.logger.info(f"  ⚠️  Needs Review: {summary.features_needing_review}")
        self.logger.info(f"  👎 Poor Matches: {summary.features_with_poor_matches}")
        self.logger.info("")
        self.logger.info(f"💰 Estimated Cost: ${summary.estimated_cost_usd:.4f} USD")
        self.logger.info(f"📞 LLM API Calls: {summary.estimated_llm_calls}")
        if self.stats["llm_errors"] > 0:
            self.logger.warning(f"⚠️  LLM Errors: {self.stats['llm_errors']}")
        self.logger.info("=" * 80)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced Feature Mapper with LLM Semantic Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with defaults
  python enhanced_mapper_langchain.py input.json

  # Custom threshold
  python enhanced_mapper_langchain.py input.json --threshold 0.80

  # Debug logging
  python enhanced_mapper_langchain.py input.json --debug

  # Custom output path
  python enhanced_mapper_langchain.py input.json --output custom_output.json
        """
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSON file with feature mappings"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: context_matching_<input>.json)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.85,
        help="Similarity threshold for LLM evaluation (default: 0.85)"
    )
    parser.add_argument(
        "--max-candidates", "-m",
        type=int,
        default=10,
        help="Max candidates to evaluate per feature (default: 10)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="eu-central-1",
        help="AWS region (default: eu-central-1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
        help="Bedrock model ID"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        input_path = Path(args.input_file)
        output_path = input_path.parent / f"context_matching_{input_path.name}"

    # Setup logging level
    log_level = logging.DEBUG if args.debug else logging.INFO

    try:
        # Load input
        logger = setup_logging(log_level)
        logger.info(f"📂 Loading input from: {args.input_file}")

        with open(args.input_file, 'r') as f:
            input_data = json.load(f)

        # Initialize mapper
        mapper = LangChainFeatureMapper(
            aws_region=args.region,
            model_id=args.model,
            similarity_threshold=args.threshold,
            max_candidates_to_evaluate=args.max_candidates,
            log_level=log_level,
        )

        # Process
        result = mapper.process_json(input_data)

        # Save output
        logger.info(f"💾 Saving enhanced output to: {output_path}")

        with open(output_path, 'w') as f:
            f.write(result.model_dump_json(indent=2, exclude_none=False))

        logger.info(f"✅ Processing complete! Output saved to: {output_path}")

        # Exit with appropriate code
        if result.EvaluationSummary.features_needing_review > 0:
            logger.warning(f"⚠️  {result.EvaluationSummary.features_needing_review} feature(s) need manual review")
            sys.exit(1)
        else:
            sys.exit(0)

    except FileNotFoundError:
        print(f"❌ Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in input file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
