---
tags:
  - concept
  - deep_learning
  - optimization
  - math
---
# Weight Decay (L2 Regularization)

## Mathematical Formulation
Weight Decay adds a penalty term to the loss function based on the squared magnitude of the weights, encouraging the network to keep weights small and preventing overfitting to noisy financial features.
$$L(\mathbf{w}, b) = MSE(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

The gradient update mathematically forces the weights to decay by a factor of $(1 - \eta \lambda)$ before applying the error gradient:
$$\mathbf{w} \leftarrow \mathbf{w}(1 - \eta \lambda) - \eta \nabla_{\mathbf{w}} MSE$$

## The Bias Penalty Fallacy
**Crucial Rule:** The bias term ($b$) must NEVER be included in the $L_2$ norm calculation. 
- **Reasoning:** The bias allows the neural network to shift its output to match the mean of the target distribution (e.g., shifting the baseline to \$50,000 for Bitcoin). Penalizing the bias forces it towards zero, artificially dragging the model's predictions down and destroying accuracy. Regularization is strictly meant to limit the influence of individual *features* by penalizing their multiplicative weights ($\mathbf{w}$).

## Related Concepts
- [[Weight Initialization]]