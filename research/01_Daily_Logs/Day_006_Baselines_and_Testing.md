---
tags:
  - daily_log
  - testing
  - baseline
  - math
---
# Day 006: Object-Oriented Baselines, Unit Testing, and Math Foundations

## Objectives Completed
1. **Data Pipeline Robustness:** Moved `SlidingWindowDataset` to `core/data/dataset.py`. Implemented rigorous `pytest` unit tests to mathematically prove the absence of Data Leakage.
2. **Python Package Infrastructure:** Resolved `ModuleNotFoundError` by creating a `setup.py` and installing the `core` directory as an editable package (`pip install -e .`), eliminating hacky `sys.path.append` logic.
3. **Training Loop Architecture Fixes:** - Moved `optimizer.zero_grad()` and `optimizer.step()` strictly inside the mini-batch loop to prevent catastrophic gradient accumulation.
   - Fixed memory leaks by extracting scalar values via `loss.item()` instead of accumulating computational graphs.
4. **Inference Optimization:** Replaced $O(N^2)$ `np.append` patterns with list accumulation and a single `torch.cat()` operation.
5. **Mathematical Baselines:** Implemented `SimpleLinearBaseline` and a custom $L_2$ Regularization loss function from scratch to understand the mechanics of Weight Decay without relying on PyTorch abstractions.

## Key Insights
- **The Tautological Test Trap:** Discovered that repeating internal code formulas in assertions creates false-positive tests. Tests must use hardcoded, pre-calculated scenarios.
- **Initialization matters:** Using raw `torch.randn` for weights causes gradient explosions in financial data due to massive initial variances.
- **Bias Exclusion in Regularization:** Mathematically verified why the bias term must NEVER be penalized in $L_2$ regularization.

## Links
- [[Tautological Testing]]
- [[Weight Initialization]]
- [[Weight Decay]]
- [[Naive Shift Illusion]]