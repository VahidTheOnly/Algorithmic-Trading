---
tags:
  - concept
  - algo_trading
  - risk_management
---
# Data Leakage in Financial Machine Learning

## Definition
Data Leakage (specifically **Look-ahead Bias** in algorithmic trading) occurs when a model inadvertently uses information from the future to make predictions about the past or present. This creates an illusion of high performance during backtesting, leading to catastrophic financial losses when deployed in live markets.

## The DataLoader Shuffle Trap
In standard computer vision or NLP tasks, the `DataLoader` is often instantiated with `shuffle=True` to break correlations in mini-batches. 

In financial time series, using `shuffle=True` for Validation or Test sets destroys the temporal chronology. Even during training, shuffling can be dangerous if overlapping sliding windows leak future target values into the training features of adjacent windows.

### Strict Engineering Rules:
1.  **Validation/Test DataLoaders**: `shuffle=False` is strictly required.
2.  **Splitting**: Train/Val/Test splits must be chronological (e.g., Train: 2018-2020, Val: 2021, Test: 2022). Random $K$-Fold Cross Validation is invalid; methods like Purged Walk-Forward Validation must be used instead.
3.  **Normalization**: Global normalization (using the mean/std of the entire dataset) causes leakage. Features must be normalized using only strictly historical rolling statistics.

## Related Concepts
- [[Custom Datasets and DataLoaders]]