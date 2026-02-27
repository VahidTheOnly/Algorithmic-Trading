---
tags:
  - concept
  - deep_learning
  - optimization
---
# Weight Initialization in Neural Networks

## The Danger of Raw Randomness
Initializing weights using a standard normal distribution (`torch.randn`) without scaling is catastrophic for financial data. If $x \in \mathbb{R}^d$ and $w \sim \mathcal{N}(0, 1)$, the variance of the output $y = w^\top x$ grows linearly with the input dimension $d$:
$$Var(y) = d \cdot Var(w) \cdot Var(x)$$
Large initial variances lead to massive initial predictions, resulting in exploding gradients during the first backward pass (`Loss = 2,903,359`).

## Scaling the Variance
To maintain stability, the initial weights must be scaled down. While techniques like Kaiming or Xavier initialization are standard, manually multiplying by a small constant (e.g., $0.1$ or $0.01$) shrinks the initial variance, preventing the network from getting trapped in severe local minima early in training.
```python
# Scaling variance down
self.w = nn.Parameter(torch.randn(input_dim, look_ahead) * 0.01)

```

## Related Concepts

- [[Weight Decay]]