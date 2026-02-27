---
tags:
  - concept
  - deep_learning
  - time_series
---
# Sliding Window Mechanism in Time Series

## Definition
In financial time series and Reinforcement Learning, an agent cannot make robust decisions based solely on instantaneous data $x_t$. The Sliding Window mechanism transforms a 1D sequence of feature vectors into a 2D matrix representing the historical context (state), capturing trends and volatility.

## Mathematical Formulation
Given a raw time-series tensor $X = (x_1, x_2, \dots, x_T)$, where $x_t \in \mathbb{R}^d$:

Let $W$ be the `look_back` window and $H$ be the `look_ahead` horizon. For a given current time step $t$, the state $S_t$ and target $y_t$ are defined as:
$$S_t = X_{[t - W : t]}$$
$$y_t = X_{[t : t + H]}$$

## Implementation Standard (Type-Hinted & Target-Isolated)
```python
def __getitem__(self, idx: int):
    if idx < 0:
        idx += len(self)
        
    t = idx + self.look_back
    x = self.data[idx : t, self.input_indices]
    y = self.data[t : t + self.look_ahead, self.target_indices]
    return x, y
```

## Related Concepts
- [[Custom Datasets and DataLoaders]]
- [[Data Leakage in Finance]]