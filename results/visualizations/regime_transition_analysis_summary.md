# Regime Transition Pattern Analysis Summary

## 1. Regime Persistence

Expected duration (time steps) spent in each state:

- **State 0**: 17.82 time steps
- **State 1**: 3.61 time steps
- **State 2**: 3.88 time steps
- **State 3**: 10.45 time steps
- **State 4**: 10.99 time steps

## 2. Top State Transitions

Most probable transitions between states:

| From | To | Probability |
|------|----|----|
| State 2.0 | State 1.0 | 0.214 |
| State 1.0 | State 2.0 | 0.206 |

## 3. State-Regime Correlations

Relationship between predicted states and actual market regimes:

| State | Bull Ratio | Mean Return | Correlation | Count |
|-------|------------|-------------|-------------|-------|
| 0.0 | 0.676 | 0.0008 | 0.106 | 1283.0 |
| 1.0 | 0.656 | 0.0007 | 0.045 | 567.0 |
| 2.0 | 0.536 | 0.0004 | -0.059 | 543.0 |
| 3.0 | 0.615 | 0.0009 | 0.011 | 877.0 |
| 4.0 | 0.382 | -0.0021 | -0.161 | 406.0 |

## 4. Regime Classification

- **State 0.0**: Bull Market (Bull ratio: 0.68)
- **State 1.0**: Bull Market (Bull ratio: 0.66)
- **State 2.0**: Mixed/Transitional (Bull ratio: 0.54)
- **State 3.0**: Bull Market (Bull ratio: 0.61)
- **State 4.0**: Bear Market (Bull ratio: 0.38)
