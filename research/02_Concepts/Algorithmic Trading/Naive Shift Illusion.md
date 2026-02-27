---
tags:
  - concept
  - algo_trading
  - time_series
---
# Naive Shift Illusion (Shifted Random Walk)


## Definition
The Naive Shift Illusion occurs when a predictive model (especially one optimized via MSE) evaluated on a highly stochastic time series (like financial prices) learns that the most mathematically optimal prediction for $t+1$ is simply the value at $t$. 

## Mechanism
Financial markets often resemble a Random Walk:
$$x_{t+1} = x_t + \epsilon_t$$
where $\epsilon_t$ is unpredictable noise. When an MSE-driven model attempts to minimize $(\hat{y}_{t+1} - x_{t+1})^2$, it realizes that predicting $\hat{y}_{t+1} = x_t$ yields the lowest possible reliable error, as the expected value of the noise is zero.

## Visual Symptoms
On a plot of true vs. predicted prices, the predictions (orange dots) will look like an exact replica of the true prices (blue dots), but visibly shifted one step to the right. The model has learned absolutely no underlying patterns; it has merely learned to become a "Lag-1" copycat.

## Related Concepts
- [[Data Leakage in Finance]]