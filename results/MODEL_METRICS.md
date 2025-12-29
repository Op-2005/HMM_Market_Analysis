# Model Performance Metrics

## Configuration
- **States**: 5
- **Observations**: 20
- **Strategy**: equal_freq
- **Direct States**: True
- **Feature**: sp500 high-low
- **Threshold**: 0.4
- **Training Steps**: 60

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.6612 (66.12%) |
| Precision | 0.7083 (70.83%) |
| Recall | 0.7133 (71.33%) |
| F1 Score | 0.7108 (71.08%) |

## Confusion Matrix

```
                 Predicted
                 Bear    Bull
Actual Bear      180     126
       Bull      123     306
```

## State Interpretations

| State | Bull Ratio | Mean Return | Correlation | Type |
|-------|------------|-------------|-------------|------|
| 0 | 0.764 | 0.0036 | 0.263 | Bull Market |
| 1 | 0.485 | 0.0045 | -0.081 | Mixed |
| 2 | 0.632 | 0.0060 | 0.056 | Bull Market |
| 3 | 0.427 | 0.0074 | -0.166 | Mixed |
| 4 | 0.140 | 0.0124 | -0.225 | Bear Market |
