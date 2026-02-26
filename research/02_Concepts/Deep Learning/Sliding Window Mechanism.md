---
tags:
  - concept
  - deep_learning
  - time_series
---
# Sliding Window Mechanism in Time Series

## Definition
In financial time series and Reinforcement Learning, an agent cannot make robust decisions based solely on instantaneous data $x_t$. The Sliding Window mechanism  transforms a 1D sequence of feature vectors into a 2D matrix representing the historical context (state), capturing trends and volatility.

## Mathematical Formulation
Given a raw time-series tensor $X = (x_1, x_2, \dots, x_T)$, where $x_t \in \mathbb{R}^d$:

Let $W$ be the `look_back` window and $H$ be the `look_ahead` horizon. For a given current time step $t$, the state $S_t$ and target $y_t$ are defined as:
$$S_t = X_{[t - W : t]} \in \mathbb{R}^{W \times d}$$
$$y_t = X_{[t : t + H]} \in \mathbb{R}^{H \times d}$$

### Boundary Logic
To prevent `IndexError: out of bounds`, the maximum number of valid $(S_t, y_t)$ pairs that can be extracted is strictly:
$$N = T - W - H + 1$$

## Implementation Standard
By mapping the raw dataset `idx` to the current time step $t$, the mathematical formulation is directly translated into clean code:
```python
def __getitem__(self, idx):
    # Map abstract index to time 't'
    t = idx + self.look_back - 1
    
    # Extract historical window (State)
    x = self.data[idx : t + 1]
    
    # Extract future window (Target)
    y = self.data[t + 1 : t + self.look_ahead + 1]
    
    return x, y
````

## Related Concepts
- [[Custom Datasets and DataLoaders]]
- [[Data Leakage in Finance]]