---
tags: [concept, deep_learning, optimization, reinforcement_learning]
aliases: [Gradient Norm Clipping, Exploding Gradients]
---
# Gradient Clipping

## 📌 Definition
Gradient Clipping is a crucial stabilization technique used during neural network training to prevent the **Exploding Gradient Problem**. It forcibly caps the gradients of the model's parameters before the optimizer takes a step, ensuring that the updates do not become uncontrollably large.

## 🧮 The Mathematics
There are two main approaches: **Value Clipping** and **Norm Clipping**. Norm clipping is the industry standard.
If the $L_2$ norm of the gradient vector $g$ exceeds a predefined threshold $c$, the gradients are scaled down:
$$g \leftarrow c \frac{g}{||g||_2}$$
This preserves the *direction* of the gradient vector but strictly bounds its *magnitude*.

## 🎯 Why is it Critical for This Thesis?
Financial time-series data is notoriously noisy. In Reinforcement Learning architectures like PPO or Actor-Critic, sudden spikes in the loss function can cause catastrophic divergence in the policy network. Clipping prevents a single bad batch from destroying days of continual learning progress.

## 💻 PyTorch Implementation
In PyTorch, this must be called *after* `loss.backward()` but *before* `optimizer.step()`:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## 🔗 Related Topics
- [[Autograd]]
- [[Computational Graphs]]
- [[Optimizer Mechanics]]