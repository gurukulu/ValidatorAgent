# Enhanced Feature Mapper with LLM Semantic Evaluation

🎯 **Smart feature mapping using vector similarity + LLM semantic evaluation**

## Overview

This system combines vector search (e.g., Pinecone) with LLM-based semantic evaluation to find accurate feature matches. The key innovation: **we only evaluate LOW similarity candidates with the LLM**, saving costs while discovering hidden semantic matches.

### The Problem

Vector search can miss semantic relationships:

```
Target: "Engine Capacity (cc)"

Vector Search Results:
1. Engine Displacement (0.842) ✅ Good
2. BMW Maps (0.792)           ❌ Wrong! (high score but irrelevant)
3. Displacement (cm³) (0.65)   ✅ Perfect! (low score but correct)
```

Vector similarity doesn't understand that:
- "BMW Maps" has NOTHING to do with engine capacity (despite 0.792 similarity)
- "Displacement (cm³)" is IDENTICAL to "Engine Capacity (cc)" (despite only 0.65 similarity)

### The Solution

**Two-stage evaluation:**

```
┌────────────────────────────┐
│  High Similarity (≥0.85)   │
│  → Auto-Accept             │
│  → No LLM needed           │
│  → Save money              │
└────────────────────────────┘
              ↓
          [MERGE]
              ↓
┌────────────────────────────┐
│  Low Similarity (<0.85)    │
│  → Send to LLM             │
│  → Get semantic score      │
│  → Find hidden matches     │
└────────────────────────────┘
```

**Benefits:**
- ✅ Cost-efficient: Don't waste LLM calls on obvious matches
- ✅ Find gems: Discover semantic matches despite low vector similarity
- ✅ Bias-free: LLM doesn't see similarity scores (no anchoring bias)
- ✅ Type-safe: LangChain guarantees valid JSON responses

## Architecture

### Key Components

1. **enhanced_models.py** - Pydantic schemas for input/output data
2. **langchain_models.py** - Pydantic schemas for LLM structured output
3. **enhanced_mapper_langchain.py** - Main implementation with LangChain + AWS Bedrock

### How It Works

```
INPUT JSON
    ↓
Split by Threshold (0.85)
    ↓
┌─────────────────┬─────────────────┐
│   Above (≥0.85) │  Below (<0.85)  │
│   Auto-accept   │  LLM evaluate   │
│   Cost: $0      │  Cost: ~$0.0003 │
└────────┬────────┴────────┬────────┘
         │                 │
         │    ┌────────────┘
         │    │
         │    ▼
         │  LangChain Structured Output
         │    │ - Pydantic validation
         │    │ - Auto-retry
         │    │ - Guaranteed JSON
         │    ↓
         │  Context Scores (0-100)
         │    │
         └────┴──────┐
                     ↓
            Merge & Select Best
                     ↓
        context_matching_<input>.json
```

### Bias Removal

**Critical Design Decision:** We hide similarity scores from the LLM to prevent anchoring bias.

**What we HIDE from LLM:**
- ❌ `similarity_score`
- ❌ `confidence`
- ❌ `source`
- ❌ `package_type`

**What we SHOW to LLM:**
- ✅ `feature_name`
- ✅ `feature_value`
- ✅ `notes`
- ✅ `oem` (extracted from `car_model`)

**Why?** LLMs suffer from anchoring bias. If they see a high similarity score, they might convince themselves there's a relationship even when there isn't.

## Installation

### Requirements

- Python 3.11 or higher
- AWS account with Bedrock access
- AWS credentials configured (via `~/.aws/credentials` or environment variables)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd ValidatorAgent

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials (if not already done)
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region (eu-central-1)
```

## Usage

### Basic Usage

```bash
python enhanced_mapper_langchain.py sample_input.json
```

Output will be saved to: `context_matching_sample_input.json`

### Advanced Options

```bash
# Custom threshold (evaluate more aggressively)
python enhanced_mapper_langchain.py input.json --threshold 0.80

# Custom output path
python enhanced_mapper_langchain.py input.json --output results.json

# Limit candidates evaluated per feature
python enhanced_mapper_langchain.py input.json --max-candidates 5

# Debug logging
python enhanced_mapper_langchain.py input.json --debug

# Different AWS region/model
python enhanced_mapper_langchain.py input.json \
  --region us-east-1 \
  --model anthropic.claude-3-sonnet-20240229-v1:0
```

### Programmatic Usage

```python
from enhanced_mapper_langchain import LangChainFeatureMapper
import json

# Initialize
mapper = LangChainFeatureMapper(
    aws_region="eu-central-1",
    model_id="eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
    similarity_threshold=0.85,
    max_candidates_to_evaluate=10,
)

# Load input
with open('input.json') as f:
    data = json.load(f)

# Process
result = mapper.process_json(data)

# Save output
with open('output.json', 'w') as f:
    f.write(result.model_dump_json(indent=2))

# Check results
print(f"Features: {result.EvaluationSummary.total_features}")
print(f"Excellent: {result.EvaluationSummary.features_with_excellent_matches}")
print(f"Review needed: {len(result.FeaturesRequiringReview)}")
```

## Input Format

```json
[
  {
    "Feature_Name": "Engine Capacity (cc)",
    "mapped_list": [
      {
        "similarity_score": 0.842,
        "confidence": 0.91,
        "metadata": {
          "feature_name": "Engine Displacement",
          "feature_value": "Total volume of all engine cylinders",
          "notes": "Measured in cubic centimeters (cc) or liters",
          "car_model": "BMW 3 Series",
          "package_type": "Technical Specifications",
          "source": "manufacturer_datasheet"
        }
      }
    ]
  }
]
```

## Output Format

Enhanced output preserves ALL original data and adds new fields:

```json
{
  "features": [
    {
      "Feature_Name": "Engine Capacity (cc)",
      "mapped_list": [
        {
          // ORIGINAL (preserved)
          "similarity_score": 0.842,
          "confidence": 0.91,
          "metadata": {...},

          // NEW (added by LLM)
          "context_matching_score": null,
          "match_quality": "not_evaluated",
          "evaluated": false,
          "llm_reasoning": "High similarity (0.842) - auto-accepted without LLM evaluation"
        },
        {
          // ORIGINAL (preserved)
          "similarity_score": 0.65,
          "metadata": {...},

          // NEW (added by LLM)
          "context_matching_score": 98,
          "match_quality": "excellent",
          "evaluated": true,
          "llm_reasoning": "Perfect semantic match - same concept with different unit notation"
        }
      ],
      "best_match_index": 1,
      "best_match_score": 98,
      "best_match_type": "context"
    }
  ],
  "EvaluationSummary": {
    "total_features": 1,
    "total_candidates": 2,
    "candidates_evaluated": 1,
    "candidates_auto_accepted": 1,
    "features_with_excellent_matches": 1,
    "estimated_llm_calls": 1,
    "estimated_cost_usd": 0.0003
  },
  "FeaturesRequiringReview": []
}
```

## Configuration

### Similarity Threshold

Controls which candidates get evaluated by LLM:

```python
# Conservative (fewer LLM calls, may miss matches)
mapper = LangChainFeatureMapper(similarity_threshold=0.90)

# Balanced (default)
mapper = LangChainFeatureMapper(similarity_threshold=0.85)

# Aggressive (more LLM calls, find more hidden matches)
mapper = LangChainFeatureMapper(similarity_threshold=0.70)
```

### Quality Thresholds

Controls quality classification:

```python
from enhanced_models import QualityThresholds

mapper = LangChainFeatureMapper(
    quality_thresholds=QualityThresholds(
        auto_accept=85,      # ≥85 → excellent
        manual_review=70,    # 70-84 → good
        reject=50            # <50 → poor
    )
)
```

### Max Candidates

Limit number of candidates evaluated per feature (cost control):

```python
mapper = LangChainFeatureMapper(
    max_candidates_to_evaluate=5  # Only evaluate top 5 below-threshold candidates
)
```

## Error Handling

The system includes robust error handling:

1. **Validation Errors**: Automatic retry up to 3 times with exponential backoff
2. **API Throttling**: Exponential backoff on 429 errors
3. **Network Errors**: Retry with backoff
4. **LLM Failures**: Fallback score of 50, continue processing
5. **Authentication Errors**: Fail immediately with clear message

## Logging

Comprehensive logging at multiple levels:

```bash
# Standard logging
python enhanced_mapper_langchain.py input.json

# Debug logging (shows LLM prompts and responses)
python enhanced_mapper_langchain.py input.json --debug
```

**Log Output Includes:**
- 🚀 Initialization details
- 🎯 Feature processing progress
- 🤖 LLM evaluation results
- 🏆 Best match selection
- 📊 Final summary statistics
- ⚠️ Warnings and errors

## Cost Estimation

**Estimated cost per LLM evaluation:** ~$0.0003 USD

**Example scenarios:**

| Scenario | LLM Calls | Cost |
|----------|-----------|------|
| 5 features, 3 below-threshold each | 15 | $0.0045 |
| 10 features, 5 below-threshold each | 50 | $0.0150 |
| 100 features, 2 below-threshold each | 200 | $0.0600 |

**Cost-saving tips:**
- Increase `similarity_threshold` to reduce LLM calls
- Decrease `max_candidates_to_evaluate` to limit evaluations per feature
- Use high-quality vector embeddings to reduce false positives

## Example Walkthrough

### Input

```json
[
  {
    "Feature_Name": "Engine Capacity (cc)",
    "mapped_list": [
      {
        "similarity_score": 0.842,
        "metadata": {"feature_name": "Engine Displacement", ...}
      },
      {
        "similarity_score": 0.792,
        "metadata": {"feature_name": "BMW Maps", ...}
      },
      {
        "similarity_score": 0.65,
        "metadata": {"feature_name": "Displacement (cm³)", ...}
      }
    ]
  }
]
```

### Processing

1. **Candidate 1** (similarity: 0.842)
   - ✅ Above threshold (≥0.85)? No, but close
   - Actually below, so → LLM evaluation
   - LLM Score: 95/100 (excellent)

2. **Candidate 2** (similarity: 0.792)
   - ❌ Below threshold → LLM evaluation
   - LLM Score: 5/100 (poor - "Maps have nothing to do with engine capacity")

3. **Candidate 3** (similarity: 0.65)
   - ❌ Below threshold → LLM evaluation
   - LLM Score: 98/100 (excellent - "Perfect semantic match despite different notation")

### Output

```
🏆 Best Match: Displacement (cm³) (score: 98, type: context)
```

**Key insight:** Vector search ranked it 3rd with 0.65 similarity, but LLM found it's the best semantic match!

## Troubleshooting

### AWS Credentials Error

```
Error: Unable to locate credentials
```

**Solution:**
```bash
aws configure
# Enter your Access Key ID and Secret Access Key
```

### Model Not Found

```
Error: Could not access model eu.anthropic.claude-3-7-sonnet-20250219-v1:0
```

**Solutions:**
1. Check model availability in your AWS region
2. Request Bedrock model access in AWS Console
3. Use alternative model: `--model anthropic.claude-3-sonnet-20240229-v1:0`

### Rate Limiting

```
Warning: Throttling error (429)
```

**Solution:** The system automatically retries with exponential backoff. For future prevention:
- Reduce `max_candidates_to_evaluate`
- Add delays between features (future enhancement)

## Technical Details

### Why LangChain?

**Without LangChain:**
```python
response = bedrock.invoke_model(...)
text = json.loads(response.body)['content'][0]['text']
# LLM might return: "Here's my eval: {score: 98}" ❌
# Or: {"score": "ninety-eight"} ❌
# Or: Malformed JSON ❌

try:
    parsed = json.loads(text)
    score = int(parsed["score"])
    if score < 0 or score > 100:
        score = 50
except:
    score = 50
```

**With LangChain:**
```python
class CandidateEvaluation(BaseModel):
    context_matching_score: int = Field(..., ge=0, le=100)

structured_llm = llm.with_structured_output(CandidateEvaluation)
response = structured_llm.invoke(prompt)
# response.context_matching_score is GUARANTEED to be int 0-100 ✅
```

### Match Quality Classification

| Score Range | Quality | Action |
|-------------|---------|--------|
| 85-100 | Excellent | Auto-accept |
| 70-84 | Good | Accept with confidence |
| 50-69 | Needs Review | Manual review recommended |
| 0-49 | Poor | Likely reject |

## Future Enhancements

Potential improvements (not yet implemented):

- [ ] Batch evaluation (evaluate multiple candidates in one LLM call)
- [ ] Parallel feature processing
- [ ] Rate limiting to respect Bedrock quotas
- [ ] Caching of LLM evaluations
- [ ] Cost tracking dashboard
- [ ] A/B testing framework for threshold tuning
- [ ] Support for other LLM providers (OpenAI, Anthropic direct API)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

[Specify your license here]

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Built with:**
- 🦜 LangChain (structured outputs)
- 🤖 AWS Bedrock (Claude 3.7 Sonnet)
- ✨ Pydantic v2 (data validation)
- 🐍 Python 3.11+
