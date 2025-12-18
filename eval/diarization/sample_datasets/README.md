# Sample Diarization Validation Datasets

This folder contains sample validation datasets for diarization evaluation in RTTM format following ADR-003 specifications.

## RTTM Format

The RTTM (Rich Transcription Time Marked) format is used to represent speaker diarization annotations:

```text
SPEAKER <file-id> <channel> <start-time> <duration> <ortho> <stype> <name> <conf>
```

Where:

- `file-id`: Recording identifier
- `channel`: Audio channel (typically 1)
- `start-time`: Segment start time in seconds
- `duration`: Segment duration in seconds
- `ortho`: Orthographic field (not used, typically "\<NA\>")
- `stype`: Speaker type (not used, typically "\<NA\>")
- `name`: Speaker identifier
- `conf`: Confidence score (optional, defaults to 1.0)

## Sample Datasets

- `english_diarization_reference.rttm`: Ground truth English speaker diarization
- `english_diarization_hypothesis.rttm`: Example system predictions for English
- `arabic_diarization_reference.rttm`: Ground truth Arabic speaker diarization
- `arabic_diarization_hypothesis.rttm`: Example system predictions for Arabic

## Evaluation Metrics

The diarization evaluator calculates:

1. **DER (Diarization Error Rate)**: Overall error rate including missed speech, false alarms, and speaker confusion
2. **JER (Jaccard Error Rate)**: Error based on Jaccard similarity between speaker segments
3. **Component metrics**: Individual rates for missed speech, false alarms, and speaker confusion

## Usage

```python
from eval.diarization.main import load_rttm_file, evaluate_diarization_model

# Load RTTM files
ref_segments = load_rttm_file("english_diarization_reference.rttm")
hyp_segments = load_rttm_file("english_diarization_hypothesis.rttm")

# Evaluate
results = evaluate_diarization_model(ref_segments, hyp_segments, collar=0.25)
print(f"DER: {results.der:.3f}")
print(f"JER: {results.jer:.3f}")
```
