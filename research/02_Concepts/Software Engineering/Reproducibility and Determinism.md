---
aliases: [Determinism, PRNG Seeding]
tags: [reproducibility, software-engineering, mlops]
date: 2026-02-24
status: verified
---

# Reproducibility and Determinism in Deep Learning

## Mathematical Context
Deep learning relies heavily on stochasticity. Weight initialization (e.g., $W \sim \mathcal{U}(-\frac{1}{\sqrt{k}}, \frac{1}{\sqrt{k}})$), data shuffling, and dropout mechanisms all utilize Pseudo-Random Number Generators (PRNGs). 
In Reinforcement Learning, the environment itself often contains stochastic transitions $P(s' | s, a)$. Without fixing the seed ($S_0$), the variance is too high to distinguish algorithmic improvements from statistical noise.

## PyTorch Implementation Requirements
To enforce strict determinism across all execution layers, the following PRNGs must be locked:
1. `random.seed(seed)`: Standard Python PRNG.
2. `np.random.seed(seed)`: NumPy PRNG.
3. `torch.manual_seed(seed)`: PyTorch CPU operations.
4. `torch.cuda.manual_seed_all(seed)`: PyTorch Multi-GPU operations.
5. `torch.backends.cudnn.deterministic = True`: Forces CuDNN to use deterministic algorithms.
6. `torch.backends.cudnn.benchmark = False`: Prevents CuDNN from auto-tuning and selecting stochastic but faster algorithms dynamically at runtime.

---
**Related Concepts:**
- [[Day_004_Experiment_Infrastructure]]
- [[Experiment Tracking]]