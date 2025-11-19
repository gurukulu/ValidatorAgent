# Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- Python 3.11+
- AWS account with Bedrock access
- AWS credentials configured

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure AWS credentials (if not done)
aws configure
# Region: eu-central-1
```

## Basic Usage

```bash
# Run with sample data
python enhanced_mapper_langchain.py sample_input.json
```

Output saved to: `context_matching_sample_input.json`

## Understanding the Output

The system evaluates each candidate and adds these fields:

- **context_matching_score** (0-100): LLM's semantic similarity score
- **match_quality**: excellent | good | needs_review | poor | not_evaluated
- **evaluated**: Whether LLM evaluated this candidate
- **llm_reasoning**: Explanation for the score

## Key Concepts

### Similarity Threshold (default: 0.85)

- **Above threshold** (≥0.85): Auto-accepted, no LLM call
- **Below threshold** (<0.85): Sent to LLM for evaluation

This saves money and finds hidden semantic matches!

### Example

```
Candidate: "Displacement (cm³)"
Similarity Score: 0.65 (below threshold)
→ Sent to LLM
LLM Context Score: 98/100
Quality: excellent
Reasoning: "Perfect semantic match - same concept, different notation"
```

Despite low vector similarity, LLM found it's semantically identical!

## Common Use Cases

### 1. Process Your Own Data

```bash
python enhanced_mapper_langchain.py your_data.json
```

### 2. Adjust Threshold

```bash
# More aggressive (evaluate more candidates)
python enhanced_mapper_langchain.py input.json --threshold 0.70

# More conservative (evaluate fewer candidates)
python enhanced_mapper_langchain.py input.json --threshold 0.90
```

### 3. Debug Mode

```bash
python enhanced_mapper_langchain.py input.json --debug
```

Shows detailed LLM prompts and responses.

## Expected Input Format

```json
[
  {
    "Feature_Name": "Your Target Feature",
    "mapped_list": [
      {
        "similarity_score": 0.85,
        "metadata": {
          "feature_name": "Candidate Feature",
          "feature_value": "Description",
          "notes": "Additional context",
          "car_model": "Brand Model"
        }
      }
    ]
  }
]
```

## Troubleshooting

### AWS Credentials Error

```bash
aws configure
# Enter Access Key ID, Secret Key, Region
```

### Model Access Error

Request Bedrock model access in AWS Console:
1. Go to AWS Bedrock
2. Model access
3. Request access to Claude models

## Next Steps

- Read [README.md](README.md) for full documentation
- Customize quality thresholds
- Integrate into your pipeline
- Review features requiring manual review

## Cost Estimate

**~$0.0003 per candidate evaluated**

Example:
- 10 features
- 5 below-threshold candidates each
- Total: 50 evaluations × $0.0003 = **$0.015**

Very cost-effective! 💰

## Support

Questions? Check the [README.md](README.md) or open an issue.

---

Happy mapping! 🚀
