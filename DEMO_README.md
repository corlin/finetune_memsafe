# Enhanced Qwen3 Checkpoint-450 Evaluation Demo

This demo program tests the fine-tuned `enhanced-qwen3-finetuned/checkpoint-450` model using QA data from `data/raw` and leverages the industry model evaluation capabilities.

## Features

- **Automated QA Testing**: Loads questions from markdown files and generates answers using the fine-tuned model
- **Industry Evaluation**: Uses specialized evaluators for knowledge accuracy, terminology usage, and reasoning quality
- **Comprehensive Reporting**: Generates detailed evaluation reports with category-wise performance analysis
- **Flexible Configuration**: Supports various parameters for model generation and evaluation

## Quick Start

### Option 1: Simple Run
```bash
python run_demo.py
```

### Option 2: Direct Execution
```bash
python demo_checkpoint_evaluation.py --max-samples 10
```

### Option 3: Custom Configuration
```bash
python demo_checkpoint_evaluation.py \
    --checkpoint enhanced-qwen3-finetuned/checkpoint-450 \
    --max-samples 20 \
    --output my_evaluation_report.json
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- PyYAML (optional, for configuration)

Install requirements:
```bash
pip install torch transformers pyyaml
```

## Input Data Structure

The demo expects QA data in markdown format in the `data/raw` directory:

```
data/raw/
├── QA1.md
├── QA2.md
├── QA3.md
└── ...
```

Each QA file should follow this format:
```markdown
## Category Name

### Q1: Question text here?
A1: Answer text here.

### Q2: Another question?
A2: Another answer.
```

## Output

The demo generates a JSON report with:

- **Summary Statistics**: Overall performance metrics
- **Category Analysis**: Performance breakdown by question category
- **Detailed Results**: Individual question-answer pairs with scores
- **Evaluation Details**: Specific evaluation metrics from industry evaluators

Example output structure:
```json
{
  "summary": {
    "total_samples": 20,
    "average_score": 0.756,
    "max_score": 0.923,
    "min_score": 0.445,
    "category_averages": {
      "基础概念类": 0.812,
      "密码应用等级类": 0.734,
      "技术要求类": 0.689
    }
  },
  "detailed_results": [...]
}
```

## Evaluation Metrics

The demo uses multiple evaluation dimensions:

1. **Knowledge Accuracy** (40% weight): Measures factual correctness and domain knowledge
2. **Terminology Accuracy** (30% weight): Evaluates proper use of technical terms
3. **Reasoning Quality** (30% weight): Assesses logical reasoning and explanation quality

## Configuration

You can customize the evaluation using `demo_config.yaml`:

```yaml
checkpoint_path: "enhanced-qwen3-finetuned/checkpoint-450"
max_samples: 20
generation_config:
  max_length: 512
  temperature: 0.7
evaluation_weights:
  knowledge_accuracy: 0.4
  terminology_accuracy: 0.3
  reasoning_quality: 0.3
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure the checkpoint directory exists and contains required files
2. **CUDA Memory Error**: Reduce batch size or use CPU-only mode
3. **Missing QA Data**: Verify QA*.md files exist in data/raw directory

### Debug Mode

Run with verbose logging:
```bash
python demo_checkpoint_evaluation.py --max-samples 5 2>&1 | tee demo.log
```

## Performance Tips

- Start with fewer samples (`--max-samples 5`) for quick testing
- Use GPU acceleration if available
- Monitor memory usage for large datasets

## Integration with Industry Evaluation Framework

The demo automatically detects and uses the industry evaluation framework if available:

- `industry_evaluation.evaluators.knowledge_evaluator`
- `industry_evaluation.evaluators.terminology_evaluator`
- `industry_evaluation.evaluators.reasoning_evaluator`

If these modules are not available, it falls back to simple similarity-based evaluation.

## Example Usage

```bash
# Quick test with 5 samples
python demo_checkpoint_evaluation.py --max-samples 5

# Full evaluation with custom output
python demo_checkpoint_evaluation.py --max-samples 50 --output full_report.json

# Test specific checkpoint
python demo_checkpoint_evaluation.py --checkpoint enhanced-qwen3-finetuned/checkpoint-300
```

## Results Interpretation

- **Score Range**: 0.0 to 1.0 (higher is better)
- **Good Performance**: > 0.7
- **Acceptable Performance**: 0.5 - 0.7
- **Needs Improvement**: < 0.5

Category-specific performance helps identify model strengths and weaknesses in different knowledge areas.