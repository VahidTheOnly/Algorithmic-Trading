---
tags: [daily_log, pytorch, modules, optimization]
date: 2026-02-21
---
# Day 2: Modules & Manual Training Loop

## 🎯 Goal
Debug leaf node errors, compute financial features via advanced indexing, and build a custom neural network module.

---

## ✅ What I Implemented
- Deliberately triggered and resolved an in-place operation error on a leaf tensor.
- Used advanced tensor indexing to compute the Typical Price and normalize volume data.
- Designed a custom `LinearRewardNet` inheriting from `nn.Module`.
- Registered model parameters (`nn.Parameter`) and un-optimized buffers (`register_buffer`).
- Implemented a manual training loop comparing SGD and Adam optimizers.

---

## 🧠 Key Insights
- Modifying a `requires_grad=True` tensor in-place ruins the backward pass. Always use `with torch.no_grad():` for manual weight updates.
- Normalizing financial data (Z-score) drastically stabilizes and accelerates the Adam optimizer.
- `register_buffer` is vital for holding state variables (like profit targets) that shouldn't receive gradients.

---

## 🔗 Related Concepts
- [[In-Place Operations]]
- [[Leaf Nodes]]
- [[Feature Normalization]]
- [[PyTorch Module Internals]]