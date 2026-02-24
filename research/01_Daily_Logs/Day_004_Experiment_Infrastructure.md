---
tags: [daily-log, mlops, reproducibility, debugging]
date: 2026-02-24
project: Thesis-RL-Trading
---
## Objective
Establish a robust, reproducible, and trackable experimental framework for Deep Learning and RL using PyTorch, WandB, and YAML configurations. Transition from notebook-based scripting to core modular engineering.

## Key Achievements
1. **Core Refactoring:** Migrated the `Trainer` class from Jupyter notebooks to a centralized `core/training/trainer.py` module.
2. **Resolved Silent Bug:** Identified and fixed a critical [[Device Mismatch Bug]] where the optimizer tracked CPU parameters before the model was moved to the GPU.
3. **Determinism:** Implemented `set_seed` to lock Python, NumPy, and PyTorch PRNGs, enforcing exact [[Reproducibility and Determinism]] across runs.
4. **MLOps Integration:** Integrated Weights & Biases (WandB) for continuous [[Experiment Tracking]].
5. **Network-Gapped Workflow:** Engineered an offline WandB workflow (`mode="offline"`) to bypass network restrictions (403 Forbidden errors), effectively mimicking secure cluster environments.
6. **Config-Driven Architecture:** Decoupled hyperparameters from execution logic using YAML configurations to prevent hardcoding.

---
**Related Logs:**
- Previous: [[Day_003_Training_System_Engineering]]
- Next: [[Day 005 Financial Time Series]] (Planned)