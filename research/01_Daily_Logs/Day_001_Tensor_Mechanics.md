---
tags:
  - daily_log
  - pytorch
  - autograd
  - configs
date: 2026-02-20
---
# Day 1: Tensor Mechanics & Autograd

## 🎯 Goal
Understand PyTorch tensor initialization, computational graphs, and config-driven design.

---

## ✅ What I Implemented
- Loaded raw BTCUSDT data using Pandas and converted to `torch.float32`.
- Explored the Autograd engine internals (`requires_grad`, `backward()`).
- Implemented a manual Gradient Accumulation loop.
- Built a YAML config parser to manage hyperparameters dynamically.
- Tested `retain_graph=True` using a custom asymmetric trading loss function.

---

## 🧠 Key Insights
- Gradients naturally accumulate in PyTorch unless explicitly cleared with `zero_()`.
- `retain_graph=True` is powerful but can cause memory leaks if misused; calling `backward()` twice without it destroys the computational graph.
- Config-driven architecture decouples parameters from code, making experiments scalable.

---

## 🔗 Related Concepts
- [[Autograd]]
- [[Computational Graphs]]
- [[Gradient Accumulation]]
- [[YAML Config Architecture]]