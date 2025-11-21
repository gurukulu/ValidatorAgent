#!/usr/bin/env python3
"""
Enhanced Feature Mapper with AWS Bedrock.

This script evaluates feature mappings using:
1. Vector similarity scores (from Pinecone or similar)
2. LLM semantic evaluation (via AWS Bedrock)

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
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from pydantic import ValidationError

from enhanced_models import (
    FeatureMapping,
    MappedCandidate,
    EnhancedMappingResult,
    EvaluationSummary,
    MatchQuality,
    QualityThresholds,
    AgentInputWrapper,
    AgentOutputWrapper,
)
from langchain_models import (
    CandidateEvaluation,
    BiasFreeCandidateInput,
    BatchEvaluationResult,
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
# BEDROCK FEATURE MAPPER
# ============================================================================

class BedrockFeatureMapper:
    """
    Feature mapper using AWS Bedrock for semantic evaluation.

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
    DEFAULT_BATCH_SIZE = 10  # Max candidates per batch LLM call
    DEFAULT_EVALUATE_ALL = True  # Evaluate ALL candidates below threshold
    DEFAULT_SKIP_GOOD_FEATURES = False  # Skip features with good similarity matches
    DEFAULT_MAX_WORKERS = 1  # Number of parallel threads (1 = sequential)
    DEFAULT_MIN_LLM_THRESHOLD = 0.5  # Minimum similarity to warrant LLM evaluation
    COST_PER_EVALUATION = 0.0003  # Estimated cost in USD (single call)
    COST_PER_BATCH_CALL = 0.0008  # Estimated cost for batch call (amortized)

    def __init__(
        self,
        aws_region: str = "eu-central-1",
        model_id: str = "eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_candidates_to_evaluate: int = DEFAULT_MAX_CANDIDATES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_batch_evaluation: bool = True,
        evaluate_all: bool = DEFAULT_EVALUATE_ALL,
        skip_features_with_good_matches: bool = DEFAULT_SKIP_GOOD_FEATURES,
        max_workers: int = DEFAULT_MAX_WORKERS,
        min_llm_threshold: float = DEFAULT_MIN_LLM_THRESHOLD,
        quality_thresholds: Optional[QualityThresholds] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the Bedrock Feature Mapper.

        Args:
            aws_region: AWS region for Bedrock (default: eu-central-1)
            model_id: Bedrock model ID (default: Claude 3.7 Sonnet)
            similarity_threshold: Split point for evaluation (default: 0.85)
            max_candidates_to_evaluate: Max candidates if evaluate_all=False (default: 10)
            batch_size: Max candidates per batch LLM call (default: 10)
            use_batch_evaluation: Enable batch evaluation to save tokens (default: True)
            evaluate_all: Evaluate ALL candidates below threshold, ignore max limit (default: True)
            skip_features_with_good_matches: Skip features where highest similarity >= threshold (default: False)
            max_workers: Number of parallel threads for processing features (default: 1 = sequential)
            min_llm_threshold: Minimum similarity score to warrant LLM evaluation (default: 0.5)
            quality_thresholds: Custom quality thresholds
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)

        self.aws_region = aws_region
        self.model_id = model_id
        self.similarity_threshold = similarity_threshold
        self.max_candidates_to_evaluate = max_candidates_to_evaluate
        self.batch_size = batch_size
        self.use_batch_evaluation = use_batch_evaluation
        self.evaluate_all = evaluate_all
        self.skip_features_with_good_matches = skip_features_with_good_matches
        self.max_workers = max_workers
        self.min_llm_threshold = min_llm_threshold
        self.quality_thresholds = quality_thresholds or QualityThresholds()

        # Thread-safe statistics (for parallel processing)
        self.stats_lock = threading.Lock()

        # Statistics
        self.stats = {
            "total_features": 0,
            "total_candidates": 0,
            "candidates_evaluated": 0,
            "candidates_auto_accepted": 0,
            "candidates_rejected_low_similarity": 0,  # Rejected due to low similarity
            "features_skipped": 0,  # Features skipped due to good matches
            "llm_calls": 0,
            "batch_llm_calls": 0,
            "individual_llm_calls": 0,
            "llm_errors": 0,
            "batch_fallbacks": 0,  # Times we fell back to individual calls
        }

        self.logger.info("=" * 80)
        self.logger.info("🚀 Bedrock Feature Mapper Initialized")
        self.logger.info("=" * 80)
        self.logger.info(f"📍 AWS Region: {self.aws_region}")
        self.logger.info(f"🤖 Model: {self.model_id}")
        self.logger.info(f"📊 Similarity Threshold: {self.similarity_threshold}")
        self.logger.info(f"🔽 Min LLM Threshold: {self.min_llm_threshold} (candidates below this are auto-rejected)")
        if self.evaluate_all:
            self.logger.info(f"🎯 Candidate Evaluation: ALL candidates in range [{self.min_llm_threshold}, {self.similarity_threshold})")
        else:
            self.logger.info(f"🎯 Max Candidates to Evaluate: {self.max_candidates_to_evaluate}")
        self.logger.info(f"📦 Batch Evaluation: {'Enabled' if self.use_batch_evaluation else 'Disabled'} "
                        f"(batch size: {self.batch_size})")
        self.logger.info(f"⏭️  Skip Features with Good Matches: {'Enabled' if self.skip_features_with_good_matches else 'Disabled'}")
        if self.max_workers > 1:
            self.logger.info(f"⚡ Parallel Processing: Enabled ({self.max_workers} workers)")
        else:
            self.logger.info(f"⚡ Parallel Processing: Disabled (sequential)")
        self.logger.info(f"✨ Quality Thresholds: Auto-accept={self.quality_thresholds.auto_accept}, "
                        f"Review={self.quality_thresholds.manual_review}, "
                        f"Reject={self.quality_thresholds.reject}")
        self.logger.info("=" * 80)

        # Initialize AWS Bedrock client
        self._initialize_bedrock()

    def _initialize_bedrock(self) -> None:
        """Initialize AWS Bedrock client."""
        try:
            self.logger.info("🔧 Initializing AWS Bedrock client...")

            # Create Bedrock runtime client using AWS config credentials
            self.bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.aws_region,
            )

            # Model configuration
            self.model_config = {
                "temperature": 0.1,  # Low temperature for consistent evaluation
                "top_p": 0.9,
                "max_tokens": 4000,  # Increased for batch evaluations
            }

            self.logger.info("✅ AWS Bedrock client initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize AWS Bedrock client: {e}")
            raise

    def _invoke_bedrock(self, prompt: str) -> str:
        """
        Invoke AWS Bedrock with a prompt and return the response text.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Response text from the model

        Raises:
            Exception: If the API call fails
        """
        # Prepare request body for Claude models
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.model_config["max_tokens"],
            "temperature": self.model_config["temperature"],
            "top_p": self.model_config["top_p"],
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        # Invoke the model
        response = self.bedrock_client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body)
        )

        # Parse response
        response_body = json.loads(response['body'].read())

        # Extract text from Claude response format
        if 'content' in response_body and len(response_body['content']) > 0:
            return response_body['content'][0]['text']
        else:
            raise ValueError("Unexpected response format from Bedrock")

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from LLM response.

        The LLM might wrap JSON in markdown code blocks or add explanatory text.
        This method extracts the JSON and parses it.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed JSON as dictionary

        Raises:
            ValueError: If JSON cannot be extracted or parsed
        """
        # Try to find JSON in code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON (starts with { and ends with })
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Assume the entire response is JSON
                json_str = response_text.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from response: {e}\nResponse: {response_text[:200]}")

    def _sanitize_evaluation_data(self, data: Dict[str, Any], max_reasoning_length: int = 2000) -> Dict[str, Any]:
        """
        Sanitize evaluation data to ensure it meets Pydantic constraints.

        This prevents validation errors by truncating fields that are too long.

        Args:
            data: Raw evaluation data from LLM
            max_reasoning_length: Maximum length for reasoning field

        Returns:
            Sanitized data that will pass Pydantic validation
        """
        # Truncate reasoning if too long
        if "reasoning" in data and len(data["reasoning"]) > max_reasoning_length:
            self.logger.warning(
                f"⚠️ Reasoning too long ({len(data['reasoning'])} chars), "
                f"truncating to {max_reasoning_length} chars"
            )
            data["reasoning"] = data["reasoning"][:max_reasoning_length-3] + "..."

        # Limit key_factors to 5 items
        if "key_factors" in data and len(data["key_factors"]) > 5:
            self.logger.warning(f"⚠️ Too many key_factors ({len(data['key_factors'])}), limiting to 5")
            data["key_factors"] = data["key_factors"][:5]

        return data

    def _sanitize_batch_data(self, data: Dict[str, Any], max_reasoning_length: int = 2000) -> Dict[str, Any]:
        """
        Sanitize batch evaluation data to ensure all evaluations meet Pydantic constraints.

        Args:
            data: Raw batch evaluation data from LLM
            max_reasoning_length: Maximum length for reasoning field

        Returns:
            Sanitized data that will pass Pydantic validation
        """
        if "evaluations" in data and isinstance(data["evaluations"], list):
            for eval_data in data["evaluations"]:
                self._sanitize_evaluation_data(eval_data, max_reasoning_length)

        return data

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
        prompt = f"""You are a senior automotive engineer with 15+ years of experience in vehicle systems design,
specializing in powertrain, safety systems, infotainment, ADAS, and chassis technologies.

**YOUR ROLE**: Critically evaluate this feature match with EXTREME PRECISION. Reject non-matches decisively.
MOST candidates will NOT match - only TRUE technical equivalents deserve high scores.

**CRITICAL: DISQUALIFYING FACTORS** (Automatic LOW score if ANY apply):
1. ❌ Different subsystems (Braking ≠ Infotainment, Powertrain ≠ Safety, Electrical ≠ Mechanical)
2. ❌ Different measurement types (Torque ≠ Power, Weight ≠ Volume, Pressure ≠ Temperature)
3. ❌ Different components (Rotors ≠ Pads, Engine ≠ Transmission, Display ≠ Camera)
4. ❌ Different vehicle systems (Body ≠ Chassis, Interior ≠ Exterior)
5. ❌ Vague semantic similarity without functional equivalence

**MULTILINGUAL MATCHING**:
✅ Features in different languages CAN match IF they are semantic equivalents
✅ Examples of valid cross-language matches:
   - "Motor" (German) = "Engine" (English) → Score 95-100 ✓
   - "Bremsscheiben" (German) = "Brake Rotors" (English) → Score 95-100 ✓
   - "Getriebe" (German) = "Transmission" (English) → Score 95-100 ✓
✅ ALWAYS translate non-English terms to English in your reasoning
✅ Score based on semantic/technical equivalence, NOT language difference

**TARGET FEATURE**:
Name: {target_feature_name}

**CANDIDATE FEATURE**:
Name: {candidate.feature_name}
Value: {candidate.feature_value}
Notes: {candidate.notes}
OEM: {candidate.oem}

**STRICT EVALUATION PROCESS**:
Step 1: Translate any non-English terms to English (e.g., "Netztrennwan" → "Network Separation Wall")
Step 2: Identify target's subsystem (Powertrain, Braking, Safety, Infotainment, ADAS, Chassis, Body, Electrical, etc.)
Step 3: For candidate:
   a) Does candidate belong to SAME subsystem? If NO → Score 0-20
   b) Does candidate serve SAME functional purpose? If NO → Score 0-30
   c) Are specifications compatible? If NO → Score 0-40
   d) Only if YES to all: Consider 70+ score
Step 4: Apply conservative scoring

**NEGATIVE EXAMPLES** (What NOT to match - Score 0-10):
- "Brake Rotors" ≠ "Netztrennwan/Network Separation Wall" (Braking ≠ Electrical, different subsystems)
- "Engine Capacity" ≠ "Torque" (Capacity ≠ Force, different measurements)
- "Airbags" ≠ "Seat Belts" (Both safety, but different components)
- "Navigation System" ≠ "Parking Sensors" (Both ADAS, but different functions)
- "Leather Seats" ≠ "Leather Steering Wheel" (Both interior, but different components)

**POSITIVE CROSS-LANGUAGE EXAMPLES** (Score 95-100):
- "Motor" (German) = "Engine" (English) → Same component, same subsystem ✓
- "Bremsscheiben" (German) = "Brake Rotors" (English) → Same component, same subsystem ✓
- "Hubraum" (German) = "Engine Displacement" (English) → Same measurement, same subsystem ✓

**SUBSYSTEM TAXONOMY** (Features must match subsystem - with multilingual examples):
- Powertrain: Engine/Motor, Transmission/Getriebe, Drivetrain, Fuel System, Displacement/Hubraum
- Braking: Rotors/Bremsscheiben, Pads/Bremsbeläge, Calipers, ABS, Brake Assist
- Safety: Airbags, Seat Belts/Sicherheitsgurte, Collision Warning, Emergency Brake
- Infotainment: Display, Audio, Navigation, Connectivity, Media System
- ADAS: Cameras, Sensors, Autopilot, Lane Assist, Parking Assist
- Chassis: Suspension, Wheels/Räder, Tires/Reifen, Steering/Lenkung
- Electrical: Battery/Batterie, Charging, Wiring, Fuses, Network Components/Netztrennwan
- Body: Doors/Türen, Windows/Fenster, Roof/Dach, Paint/Lackierung, Trim

**SCORING GUIDE** (VERY Conservative):
- 95-100: IDENTICAL features (exact same component, just different wording)
- 85-94: TRUE functional equivalents (same subsystem, same function, compatible specs)
- 70-84: Related features within same subsystem (e.g., "ABS" and "Brake Assist")
- 40-69: Same subsystem but different components/functions
- 20-39: Different subsystems but vague similarity
- 0-19: Completely unrelated or different subsystems

**VERIFICATION CHECKLIST** (Must pass ALL for 85+ score):
✓ Same subsystem?
✓ Same functional purpose?
✓ Same measurement type (if applicable)?
✓ Compatible specifications?
✓ Clear technical equivalence (not just semantic similarity)?

**CRITICAL RULES**:
- Most candidates will score 0-50 (this is NORMAL and EXPECTED)
- Scores 90+ should be RARE (only true equivalents)
- Different subsystems = automatic score <20
- When uncertain about equivalence, score 30-50 (not 70-80)
- Precision is CRITICAL - false positives are worse than false negatives

Evaluate the candidate and provide your response as a JSON object with this exact structure:
{{
  "context_matching_score": <integer 0-100>,
  "reasoning": "<your reasoning here - be concise, max 2-3 sentences>",
  "key_factors": ["factor1", "factor2", ...] // 1-5 factors
}}

IMPORTANT:
- Keep reasoning concise (2-3 sentences max, under 500 characters)
- ALWAYS translate non-English terms to English in your reasoning (e.g., "Netztrennwan (Network Separation Wall)")
- Provide ONLY the JSON object, no additional text or formatting
"""
        return prompt

    def _build_batch_evaluation_prompt(
        self,
        target_feature_name: str,
        candidates: List[BiasFreeCandidateInput],
    ) -> str:
        """
        Build the prompt for batch LLM evaluation.

        Args:
            target_feature_name: The feature we're trying to match
            candidates: List of bias-free candidate data

        Returns:
            Formatted prompt string for batch evaluation
        """
        # Build candidate list
        candidates_text = ""
        for idx, candidate in enumerate(candidates, 1):
            candidates_text += f"""
**CANDIDATE {idx}**:
Name: {candidate.feature_name}
Value: {candidate.feature_value}
Notes: {candidate.notes}
OEM: {candidate.oem}
"""

        prompt = f"""You are a senior automotive engineer with 15+ years of experience in vehicle systems design,
specializing in powertrain, safety systems, infotainment, ADAS, and chassis technologies.

**YOUR ROLE**: Critically evaluate feature matches with EXTREME PRECISION. Reject non-matches decisively.
MOST candidates will NOT match - only TRUE technical equivalents deserve high scores.

**CRITICAL: DISQUALIFYING FACTORS** (Automatic LOW score if ANY apply):
1. ❌ Different subsystems (Braking ≠ Infotainment, Powertrain ≠ Safety, Electrical ≠ Mechanical)
2. ❌ Different measurement types (Torque ≠ Power, Weight ≠ Volume, Pressure ≠ Temperature)
3. ❌ Different components (Rotors ≠ Pads, Engine ≠ Transmission, Display ≠ Camera)
4. ❌ Different vehicle systems (Body ≠ Chassis, Interior ≠ Exterior)
5. ❌ Vague semantic similarity without functional equivalence

**MULTILINGUAL MATCHING**:
✅ Features in different languages CAN match IF they are semantic equivalents
✅ Examples of valid cross-language matches:
   - "Motor" (German) = "Engine" (English) → Score 95-100 ✓
   - "Bremsscheiben" (German) = "Brake Rotors" (English) → Score 95-100 ✓
   - "Getriebe" (German) = "Transmission" (English) → Score 95-100 ✓
✅ ALWAYS translate non-English terms to English in your reasoning
✅ Score based on semantic/technical equivalence, NOT language difference

**TARGET FEATURE**:
Name: {target_feature_name}

**CANDIDATES TO EVALUATE**:
{candidates_text}

**STRICT EVALUATION PROCESS**:
Step 1: Identify target's subsystem (Powertrain, Braking, Safety, Infotainment, ADAS, Chassis, Body, Electrical, etc.)
Step 2: For EACH candidate:
   a) Does candidate belong to SAME subsystem? If NO → Score 0-20
   b) Does candidate serve SAME functional purpose? If NO → Score 0-30
   c) Are specifications compatible? If NO → Score 0-40
   d) Only if YES to all: Consider 70+ score
Step 3: Apply conservative scoring

**NEGATIVE EXAMPLES** (What NOT to match - Score 0-10):
- "Brake Rotors" ≠ "Netztrennwan/Network Separation Wall" (Braking ≠ Electrical, different subsystems)
- "Engine Capacity" ≠ "Torque" (Capacity ≠ Force, different measurements)
- "Airbags" ≠ "Seat Belts" (Both safety, but different components)
- "Navigation System" ≠ "Parking Sensors" (Both ADAS, but different functions)
- "Leather Seats" ≠ "Leather Steering Wheel" (Both interior, but different components)

**POSITIVE CROSS-LANGUAGE EXAMPLES** (Score 95-100):
- "Motor" (German) = "Engine" (English) → Same component, same subsystem ✓
- "Bremsscheiben" (German) = "Brake Rotors" (English) → Same component, same subsystem ✓
- "Hubraum" (German) = "Engine Displacement" (English) → Same measurement, same subsystem ✓

**SUBSYSTEM TAXONOMY** (Features must match subsystem - with multilingual examples):
- Powertrain: Engine/Motor, Transmission/Getriebe, Drivetrain, Fuel System, Displacement/Hubraum
- Braking: Rotors/Bremsscheiben, Pads/Bremsbeläge, Calipers, ABS, Brake Assist
- Safety: Airbags, Seat Belts/Sicherheitsgurte, Collision Warning, Emergency Brake
- Infotainment: Display, Audio, Navigation, Connectivity, Media System
- ADAS: Cameras, Sensors, Autopilot, Lane Assist, Parking Assist
- Chassis: Suspension, Wheels/Räder, Tires/Reifen, Steering/Lenkung
- Electrical: Battery/Batterie, Charging, Wiring, Fuses, Network Components/Netztrennwan
- Body: Doors/Türen, Windows/Fenster, Roof/Dach, Paint/Lackierung, Trim

**SCORING GUIDE** (VERY Conservative):
- 95-100: IDENTICAL features (exact same component, just different wording)
- 85-94: TRUE functional equivalents (same subsystem, same function, compatible specs)
- 70-84: Related features within same subsystem (e.g., "ABS" and "Brake Assist")
- 40-69: Same subsystem but different components/functions
- 20-39: Different subsystems but vague similarity
- 0-19: Completely unrelated or different subsystems

**VERIFICATION CHECKLIST** (Must pass ALL for 85+ score):
✓ Same subsystem?
✓ Same functional purpose?
✓ Same measurement type (if applicable)?
✓ Compatible specifications?
✓ Clear technical equivalence (not just semantic similarity)?

**CRITICAL RULES**:
- Most candidates will score 0-50 (this is NORMAL and EXPECTED)
- Scores 90+ should be RARE (only true equivalents)
- Different subsystems = automatic score <20
- When uncertain about equivalence, score 30-50 (not 70-80)
- Precision is CRITICAL - false positives are worse than false negatives
- Evaluate EACH candidate independently

Evaluate ALL {len(candidates)} candidates and provide your response as a JSON object with this exact structure:
{{
  "evaluations": [
    {{
      "context_matching_score": <integer 0-100>,
      "reasoning": "<reasoning for candidate 1 - be concise, 2-3 sentences max>",
      "key_factors": ["factor1", "factor2", ...] // 1-5 factors
    }},
    {{
      "context_matching_score": <integer 0-100>,
      "reasoning": "<reasoning for candidate 2 - be concise, 2-3 sentences max>",
      "key_factors": ["factor1", "factor2", ...] // 1-5 factors
    }}
    // ... one evaluation object per candidate
  ]
}}

CRITICAL:
- Return evaluations in the SAME ORDER as the candidates (Candidate 1, 2, 3, etc.)
- Keep each reasoning concise (2-3 sentences max, under 500 characters)
- Provide 1-5 key_factors per candidate
Provide ONLY the JSON object, no additional text or formatting.
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

                # Invoke Bedrock
                response_text = self._invoke_bedrock(prompt)

                # Parse JSON response
                response_data = self._parse_json_response(response_text)

                # Sanitize data to prevent validation errors
                response_data = self._sanitize_evaluation_data(response_data)

                # Validate with Pydantic
                evaluation = CandidateEvaluation(**response_data)

                # Track stats (thread-safe)
                with self.stats_lock:
                    self.stats["llm_calls"] += 1

                self.logger.debug(f"✅ LLM returned score: {evaluation.context_matching_score}")
                return evaluation

            except ValidationError as e:
                self.logger.warning(f"⚠️ Validation error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    with self.stats_lock:
                        self.stats["llm_errors"] += 1
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

            except Exception as e:
                self.logger.error(f"❌ LLM error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    with self.stats_lock:
                        self.stats["llm_errors"] += 1
                    raise
                time.sleep(2 ** attempt)

        # Should never reach here
        raise Exception("Failed to evaluate candidate after all retries")

    def _evaluate_candidates_batch(
        self,
        target_feature_name: str,
        candidates: List[MappedCandidate],
    ) -> List[CandidateEvaluation]:
        """
        Evaluate multiple candidates in a single LLM call (batch evaluation).

        This is more token-efficient than individual calls because the prompt
        instructions are sent only once for all candidates.

        Args:
            target_feature_name: Target feature name
            candidates: List of candidates to evaluate

        Returns:
            List of CandidateEvaluation (one per candidate, in same order)

        Raises:
            Exception: If all retries fail
        """
        if not candidates:
            return []

        # Create bias-free candidates
        bias_free_candidates = [
            self._create_bias_free_candidate(candidate)
            for candidate in candidates
        ]

        # Build batch prompt
        prompt = self._build_batch_evaluation_prompt(
            target_feature_name,
            bias_free_candidates,
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.debug(
                    f"🔄 Batch LLM evaluation attempt {attempt + 1}/{max_retries} "
                    f"for {len(candidates)} candidates"
                )

                # Invoke Bedrock
                response_text = self._invoke_bedrock(prompt)

                # Parse JSON response
                response_data = self._parse_json_response(response_text)

                # Sanitize data to prevent validation errors
                response_data = self._sanitize_batch_data(response_data)

                # Validate with Pydantic
                batch_result = BatchEvaluationResult(**response_data)

                # Validate that we got the right number of evaluations
                if len(batch_result.evaluations) != len(candidates):
                    raise ValueError(
                        f"Expected {len(candidates)} evaluations, "
                        f"got {len(batch_result.evaluations)}"
                    )

                # Track stats (thread-safe)
                with self.stats_lock:
                    self.stats["batch_llm_calls"] += 1
                    self.stats["llm_calls"] += 1

                self.logger.debug(
                    f"✅ Batch LLM returned {len(batch_result.evaluations)} evaluations"
                )
                return batch_result.evaluations

            except ValidationError as e:
                self.logger.warning(
                    f"⚠️ Batch validation error on attempt {attempt + 1}: {e}"
                )
                if attempt == max_retries - 1:
                    with self.stats_lock:
                        self.stats["llm_errors"] += 1
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                self.logger.error(
                    f"❌ Batch LLM error on attempt {attempt + 1}: {e}"
                )
                if attempt == max_retries - 1:
                    with self.stats_lock:
                        self.stats["llm_errors"] += 1
                    raise
                time.sleep(2 ** attempt)

        # Should never reach here
        raise Exception("Failed to evaluate candidates in batch after all retries")

    def _evaluate_candidates_in_chunks(
        self,
        target_feature_name: str,
        candidates: List[MappedCandidate],
    ) -> List[CandidateEvaluation]:
        """
        Evaluate candidates in chunks to avoid max token issues.

        This method splits candidates into batches of self.batch_size and
        processes each batch separately, then combines results.

        Args:
            target_feature_name: Target feature name
            candidates: List of candidates to evaluate

        Returns:
            List of CandidateEvaluation (one per candidate, in same order)
        """
        if not candidates:
            return []

        all_evaluations = []
        total_candidates = len(candidates)

        # Process candidates in chunks
        for chunk_start in range(0, total_candidates, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, total_candidates)
            chunk = candidates[chunk_start:chunk_end]

            self.logger.info(
                f"  📦 Processing chunk {chunk_start//self.batch_size + 1} "
                f"({len(chunk)} candidates: {chunk_start+1}-{chunk_end} of {total_candidates})"
            )

            # Evaluate this chunk
            chunk_evaluations = self._evaluate_candidates_batch(
                target_feature_name,
                chunk,
            )

            all_evaluations.extend(chunk_evaluations)

        return all_evaluations

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

    def _calculate_confidence(self, candidate: MappedCandidate) -> float:
        """
        Calculate normalized confidence score (0-1) with 2 decimal places.

        Logic:
        - If evaluated (has context_matching_score): normalize to 0-1 (score/100)
        - If not evaluated (auto-accepted): use similarity_score

        Args:
            candidate: Candidate with scores

        Returns:
            Confidence score between 0.0 and 1.0, rounded to 2 decimal places
        """
        if candidate.context_matching_score is not None:
            # Evaluated: normalize context_matching_score (0-100) to (0-1)
            confidence = candidate.context_matching_score / 100.0
        else:
            # Not evaluated (auto-accepted): use similarity_score
            confidence = candidate.similarity_score

        # Round to 2 decimal places
        return round(confidence, 2)

    def _evaluate_candidates_individually(
        self,
        feature: FeatureMapping,
        candidates_to_evaluate: List[MappedCandidate],
    ) -> None:
        """
        Evaluate candidates one-by-one with individual LLM calls.

        This is used as a fallback when batch evaluation fails or is disabled.

        Args:
            feature: The feature being processed
            candidates_to_evaluate: List of candidates to evaluate
        """
        for idx, candidate in enumerate(candidates_to_evaluate, 1):
            try:
                self.logger.info(
                    f"  [{idx}/{len(candidates_to_evaluate)}] Evaluating: {candidate.metadata.feature_name} "
                    f"(similarity: {candidate.similarity_score:.3f})"
                )

                # Evaluate with LLM
                evaluation = self._evaluate_candidate_with_llm(
                    target_feature_name=feature.Feature_Name,
                    candidate=candidate,
                )

                # Update candidate with evaluation results
                candidate.context_matching_score = evaluation.context_matching_score
                candidate.match_quality = self._classify_quality(evaluation.context_matching_score)
                candidate.confidence = self._calculate_confidence(candidate)
                candidate.evaluated = True
                candidate.llm_reasoning = evaluation.reasoning

                # Track stats (thread-safe)
                with self.stats_lock:
                    self.stats["candidates_evaluated"] += 1
                    self.stats["individual_llm_calls"] += 1

                # Log result
                quality_emoji = {
                    MatchQuality.EXCELLENT: "🌟",
                    MatchQuality.GOOD: "👍",
                    MatchQuality.NEEDS_REVIEW: "⚠️",
                    MatchQuality.POOR: "👎",
                }
                emoji = quality_emoji.get(candidate.match_quality, "❓")

                self.logger.info(
                    f"      {emoji} Context: {candidate.context_matching_score}/100, "
                    f"Confidence: {candidate.confidence:.2f} ({candidate.match_quality.value})"
                )
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
                candidate.confidence = self._calculate_confidence(candidate)
                candidate.evaluated = True
                candidate.llm_reasoning = f"Evaluation failed: {str(e)}"

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

        with self.stats_lock:
            self.stats["total_features"] += 1
            self.stats["total_candidates"] += len(feature.mapped_list)

        if not feature.mapped_list:
            self.logger.warning("⚠️ No candidates found for this feature")
            return feature

        # Check if we should skip this feature (feature-level filtering)
        if self.skip_features_with_good_matches and feature.mapped_list:
            max_similarity = max(c.similarity_score for c in feature.mapped_list)
            if max_similarity >= self.similarity_threshold:
                self.logger.info(
                    f"⏭️  Skipping feature - highest similarity ({max_similarity:.3f}) >= threshold ({self.similarity_threshold})"
                )
                # Process all candidates without LLM, assign confidence scores
                for candidate in feature.mapped_list:
                    candidate.evaluated = False
                    candidate.match_quality = MatchQuality.NOT_EVALUATED
                    candidate.confidence = self._calculate_confidence(candidate)
                    candidate.llm_reasoning = (
                        f"Feature skipped - highest similarity ({max_similarity:.3f}) >= threshold"
                    )

                # Track stats (thread-safe)
                with self.stats_lock:
                    self.stats["candidates_auto_accepted"] += len(feature.mapped_list)
                    self.stats["features_skipped"] += 1

                # Select best match and return
                self._select_best_match(feature)
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
        self.logger.info(f"🔍 Below threshold (<{self.similarity_threshold}): {len(below_threshold)} → Check min threshold")

        # Process above-threshold candidates (auto-accept)
        for candidate in above_threshold:
            candidate.evaluated = False
            candidate.match_quality = MatchQuality.NOT_EVALUATED
            candidate.confidence = self._calculate_confidence(candidate)
            candidate.llm_reasoning = f"High similarity ({candidate.similarity_score:.3f}) - auto-accepted without LLM evaluation"

        # Track auto-accepted candidates (thread-safe)
        if above_threshold:
            with self.stats_lock:
                self.stats["candidates_auto_accepted"] += len(above_threshold)

        # Split below-threshold candidates: LLM-worthy vs too low
        candidates_for_llm = []
        candidates_rejected = []

        for candidate in below_threshold:
            if candidate.similarity_score >= self.min_llm_threshold:
                candidates_for_llm.append(candidate)
            else:
                candidates_rejected.append(candidate)

        # Process rejected candidates (too low for LLM evaluation)
        if candidates_rejected:
            self.logger.info(
                f"❌ Rejecting {len(candidates_rejected)} candidates (similarity < {self.min_llm_threshold}) "
                f"→ Auto-reject without LLM"
            )
            for candidate in candidates_rejected:
                candidate.evaluated = False
                candidate.match_quality = MatchQuality.POOR
                candidate.confidence = self._calculate_confidence(candidate)
                candidate.llm_reasoning = (
                    f"Auto-rejected: similarity ({candidate.similarity_score:.3f}) "
                    f"below minimum LLM threshold ({self.min_llm_threshold})"
                )

            # Track rejected candidates (thread-safe)
            with self.stats_lock:
                self.stats["candidates_rejected_low_similarity"] += len(candidates_rejected)

        # Process candidates worthy of LLM evaluation
        if self.evaluate_all:
            candidates_to_evaluate = candidates_for_llm
            if len(candidates_for_llm) > 0:
                self.logger.info(f"🎯 Evaluating ALL {len(candidates_to_evaluate)} candidates with LLM")
        else:
            candidates_to_evaluate = candidates_for_llm[:self.max_candidates_to_evaluate]
            if len(candidates_for_llm) > self.max_candidates_to_evaluate:
                self.logger.info(
                    f"⚠️ {len(candidates_for_llm)} candidates for LLM, "
                    f"limiting to {self.max_candidates_to_evaluate}"
                )

        if candidates_to_evaluate:
            self.logger.info(f"🤖 Evaluating {len(candidates_to_evaluate)} candidates with LLM...")

            # Try batch evaluation first (if enabled)
            if self.use_batch_evaluation and len(candidates_to_evaluate) > 1:
                try:
                    # Use chunking if candidates exceed batch size
                    if len(candidates_to_evaluate) > self.batch_size:
                        self.logger.info(
                            f"📦 Using chunked batch evaluation "
                            f"({len(candidates_to_evaluate)} candidates in chunks of {self.batch_size})"
                        )
                        evaluations = self._evaluate_candidates_in_chunks(
                            target_feature_name=feature.Feature_Name,
                            candidates=candidates_to_evaluate,
                        )
                    else:
                        self.logger.info(f"📦 Using batch evaluation for {len(candidates_to_evaluate)} candidates")
                        evaluations = self._evaluate_candidates_batch(
                            target_feature_name=feature.Feature_Name,
                            candidates=candidates_to_evaluate,
                        )

                    # Update candidates with batch evaluation results
                    for idx, (candidate, evaluation) in enumerate(zip(candidates_to_evaluate, evaluations), 1):
                        candidate.context_matching_score = evaluation.context_matching_score
                        candidate.match_quality = self._classify_quality(evaluation.context_matching_score)
                        candidate.confidence = self._calculate_confidence(candidate)
                        candidate.evaluated = True
                        candidate.llm_reasoning = evaluation.reasoning

                        # Log result
                        quality_emoji = {
                            MatchQuality.EXCELLENT: "🌟",
                            MatchQuality.GOOD: "👍",
                            MatchQuality.NEEDS_REVIEW: "⚠️",
                            MatchQuality.POOR: "👎",
                        }
                        emoji = quality_emoji.get(candidate.match_quality, "❓")

                        self.logger.info(
                            f"  [{idx}/{len(candidates_to_evaluate)}] {candidate.metadata.feature_name} "
                            f"(similarity: {candidate.similarity_score:.2f}) "
                            f"{emoji} Context: {candidate.context_matching_score}/100, Confidence: {candidate.confidence:.2f} ({candidate.match_quality.value})"
                        )

                        # Special highlight for semantic matches despite low similarity
                        if (candidate.similarity_score < self.similarity_threshold and
                            candidate.context_matching_score >= self.quality_thresholds.auto_accept):
                            self.logger.info(f"      🎯 Found semantic match despite low similarity!")

                    # Track batch evaluation stats (thread-safe)
                    with self.stats_lock:
                        self.stats["candidates_evaluated"] += len(evaluations)

                except Exception as e:
                    # Batch evaluation failed - fall back to individual evaluation
                    self.logger.warning(f"⚠️ Batch evaluation failed: {e}")
                    self.logger.info("🔄 Falling back to individual evaluation...")
                    with self.stats_lock:
                        self.stats["batch_fallbacks"] += 1

                    # Process individually as fallback
                    self._evaluate_candidates_individually(
                        feature, candidates_to_evaluate
                    )
            else:
                # Use individual evaluation (batch disabled or single candidate)
                if len(candidates_to_evaluate) == 1:
                    self.logger.info("Single candidate - using individual evaluation")
                else:
                    self.logger.info("Batch evaluation disabled - using individual evaluation")

                self._evaluate_candidates_individually(
                    feature, candidates_to_evaluate
                )

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
            input_data: Input JSON data (list of features, dict with 'features',
                       or dict with 'AgentInterimOutput')

        Returns:
            EnhancedMappingResult with evaluations and summary
        """
        self.logger.info("")
        self.logger.info("🚀 Starting feature mapping process...")

        # Parse input - support multiple formats
        if isinstance(input_data, list):
            # Format 1: Direct list of features
            features_data = input_data
        elif isinstance(input_data, dict):
            if "AgentInterimOutput" in input_data:
                # Format 2: New wrapper format with ExecutionID, RequestID, etc.
                self.logger.info("📦 Detected new wrapper format with AgentInterimOutput")
                features_data = input_data["AgentInterimOutput"]
            elif "features" in input_data:
                # Format 3: Old format with 'features' key
                features_data = input_data["features"]
            else:
                raise ValueError(
                    "Input dict must contain 'AgentInterimOutput' or 'features' key"
                )
        else:
            raise ValueError(
                "Input must be a list of features or dict with 'features'/'AgentInterimOutput' key"
            )

        # Convert to FeatureMapping objects
        features = [FeatureMapping(**f) for f in features_data]

        # Process features (parallel or sequential based on max_workers)
        if self.max_workers > 1:
            # Parallel processing with ThreadPoolExecutor
            self.logger.info(f"⚡ Processing {len(features)} features in parallel ({self.max_workers} workers)...")
            processed_features = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all features for processing
                future_to_feature = {
                    executor.submit(self._process_single_feature, feature): feature
                    for feature in features
                }

                # Collect results as they complete
                for future in as_completed(future_to_feature):
                    feature = future_to_feature[future]
                    try:
                        processed_feature = future.result()
                        processed_features.append(processed_feature)
                    except Exception as e:
                        self.logger.error(f"❌ Error processing feature {feature.Feature_Name}: {e}")
                        # Add feature without processing
                        processed_features.append(feature)
        else:
            # Sequential processing (original behavior)
            self.logger.info(f"🔄 Processing {len(features)} features sequentially...")
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

    def process_json_with_wrapper(self, input_data: Dict[str, Any]) -> AgentOutputWrapper:
        """
        Process input JSON with wrapper format and return wrapped output.

        This method is designed for the new input format with ExecutionID,
        RequestID, Timestamp, and AgentInterimOutput.

        Args:
            input_data: Input JSON data with wrapper structure

        Returns:
            AgentOutputWrapper with all original metadata preserved
        """
        # Parse input wrapper
        if isinstance(input_data, dict) and "AgentInterimOutput" in input_data:
            input_wrapper = AgentInputWrapper(**input_data)
        else:
            # If not in wrapper format, create a minimal wrapper
            input_wrapper = AgentInputWrapper(
                ExecutionID=None,
                RequestID=None,
                Timestamp=None,
                AgentInterimOutput=input_data if isinstance(input_data, list) else input_data.get("features", [])
            )

        # Process features
        result = self.process_json(input_data)

        # Create output wrapper preserving original metadata
        output_wrapper = AgentOutputWrapper(
            ExecutionID=input_wrapper.ExecutionID,
            RequestID=input_wrapper.RequestID,
            Timestamp=input_wrapper.Timestamp,
            AgentInterimOutput=result.features,
            evaluation_summary=result.EvaluationSummary,
            FeaturesRequiringReview=result.FeaturesRequiringReview,
            configuration=result.configuration,
        )

        return output_wrapper

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

        # Calculate cost: batch calls are more expensive but amortized, individual calls are cheaper
        batch_cost = self.stats["batch_llm_calls"] * self.COST_PER_BATCH_CALL
        individual_cost = self.stats["individual_llm_calls"] * self.COST_PER_EVALUATION
        estimated_cost = batch_cost + individual_cost

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
        if self.stats["candidates_rejected_low_similarity"] > 0:
            self.logger.info(f"❌ Candidates Auto-Rejected (low similarity < {self.min_llm_threshold}): {self.stats['candidates_rejected_low_similarity']}")
        if self.stats["features_skipped"] > 0:
            self.logger.info(f"⏭️  Features Skipped (good matches): {self.stats['features_skipped']}")
        self.logger.info("")
        self.logger.info("🎯 Quality Breakdown:")
        self.logger.info(f"  🌟 Excellent Matches: {summary.features_with_excellent_matches}")
        self.logger.info(f"  👍 Good Matches: {summary.features_with_good_matches}")
        self.logger.info(f"  ⚠️  Needs Review: {summary.features_needing_review}")
        self.logger.info(f"  👎 Poor Matches: {summary.features_with_poor_matches}")
        self.logger.info("")
        self.logger.info("📞 LLM API Calls:")
        self.logger.info(f"  📦 Batch Calls: {self.stats['batch_llm_calls']}")
        self.logger.info(f"  🔹 Individual Calls: {self.stats['individual_llm_calls']}")
        self.logger.info(f"  📊 Total API Calls: {summary.estimated_llm_calls}")
        if self.stats["batch_fallbacks"] > 0:
            self.logger.warning(f"  🔄 Batch Fallbacks: {self.stats['batch_fallbacks']}")
        self.logger.info("")
        self.logger.info(f"💰 Estimated Cost: ${summary.estimated_cost_usd:.4f} USD")

        # Calculate savings if using batch
        if self.stats["batch_llm_calls"] > 0:
            # What it would have cost with individual calls
            hypothetical_individual_cost = summary.candidates_evaluated * self.COST_PER_EVALUATION
            savings = hypothetical_individual_cost - summary.estimated_cost_usd
            savings_pct = (savings / hypothetical_individual_cost * 100) if hypothetical_individual_cost > 0 else 0
            if savings > 0:
                self.logger.info(f"💡 Batch Savings: ${savings:.4f} USD ({savings_pct:.1f}% reduction)")

        # Calculate savings from min threshold rejection
        if self.stats["candidates_rejected_low_similarity"] > 0:
            rejected_savings = self.stats["candidates_rejected_low_similarity"] * self.COST_PER_EVALUATION
            self.logger.info(f"💡 Min Threshold Savings: ${rejected_savings:.4f} USD ({self.stats['candidates_rejected_low_similarity']} candidates not evaluated)")

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
        description="Enhanced Feature Mapper with AWS Bedrock LLM Semantic Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with defaults (batch evaluation + evaluate all candidates below threshold)
  python enhanced_mapper_langchain.py input.json

  # Parallel processing for 10x speedup (process 10 features at once)
  python enhanced_mapper_langchain.py input.json --max-workers 10

  # Custom threshold
  python enhanced_mapper_langchain.py input.json --threshold 0.80

  # Set minimum LLM threshold (auto-reject candidates below 0.5 similarity)
  python enhanced_mapper_langchain.py input.json --min-llm-threshold 0.5

  # Skip features with good matches (only process features with all candidates below threshold)
  python enhanced_mapper_langchain.py input.json --skip-features-with-good-matches

  # Limit to first 10 candidates (instead of evaluating all)
  python enhanced_mapper_langchain.py input.json --no-evaluate-all --max-candidates 10

  # Disable batch evaluation (use individual LLM calls)
  python enhanced_mapper_langchain.py input.json --no-batch

  # Custom batch size for chunking (avoid token limits)
  python enhanced_mapper_langchain.py input.json --batch-size 5

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
        "--batch-size", "-b",
        type=int,
        default=10,
        help="Max candidates per batch LLM call (default: 10)"
    )
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Disable batch evaluation (use individual LLM calls)"
    )
    parser.add_argument(
        "--no-evaluate-all",
        action="store_true",
        help="Limit to max-candidates instead of evaluating all below threshold"
    )
    parser.add_argument(
        "--skip-features-with-good-matches",
        action="store_true",
        help="Skip entire feature if highest similarity >= threshold (only process features with all candidates below threshold)"
    )
    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=1,
        help="Number of parallel threads for processing features (default: 1 = sequential, recommended: 5-10 for speedup)"
    )
    parser.add_argument(
        "--min-llm-threshold",
        type=float,
        default=0.5,
        help="Minimum similarity score to warrant LLM evaluation (default: 0.5, candidates below this are auto-rejected)"
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

        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        # Initialize mapper
        mapper = BedrockFeatureMapper(
            aws_region=args.region,
            model_id=args.model,
            similarity_threshold=args.threshold,
            max_candidates_to_evaluate=args.max_candidates,
            batch_size=args.batch_size,
            use_batch_evaluation=not args.no_batch,
            evaluate_all=not args.no_evaluate_all,
            skip_features_with_good_matches=args.skip_features_with_good_matches,
            max_workers=args.max_workers,
            min_llm_threshold=args.min_llm_threshold,
            log_level=log_level,
        )

        # Detect input format and process accordingly
        has_wrapper = isinstance(input_data, dict) and "AgentInterimOutput" in input_data

        if has_wrapper:
            # Use wrapper-based processing (preserves ExecutionID, RequestID, etc.)
            logger.info("📦 Using wrapper-based processing")
            output = mapper.process_json_with_wrapper(input_data)
            summary = output.evaluation_summary
        else:
            # Use standard processing (legacy format)
            logger.info("📄 Using standard processing")
            result = mapper.process_json(input_data)
            output = result
            summary = result.EvaluationSummary

        # Save output
        logger.info(f"💾 Saving enhanced output to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output.model_dump_json(indent=2, exclude_none=False))

        logger.info(f"✅ Processing complete! Output saved to: {output_path}")

        # Exit with appropriate code
        if summary.features_needing_review > 0:
            logger.warning(f"⚠️  {summary.features_needing_review} feature(s) need manual review")
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
