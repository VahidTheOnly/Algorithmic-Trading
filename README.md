# Algorithmic Trading with Multi-Agent Continual RL

**Author:** Vahid Maleki 
**Institution:** K. N. Toosi University of Technology 
**Target:** Master's Thesis & TRL4 Lab Prototype 

## 📌 The Journey

This repository documents my 365-day, 12-month execution plan to engineer a research-grade Algorithmic Trading system using **Multi-Agent Continual Reinforcement Learning**. 

The goal is to solve catastrophic forgetting in non-stationary financial markets by transitioning from fundamental PyTorch engineering to a fully functioning multi-agent ecosystem.

## 🧭 Current Phase: Phase 1 - Engineering Foundation

am currently in **Month 1-2**, focusing on PyTorch mastery and building robust training systems from scratch. 

My current focus areas:
- Computational graph internals and Tensor mechanics 
- Custom loss functions and gradient accumulation 
- Mixed precision training and gradient clipping
- Checkpointing and config-driven experiments 

*Note: Exploratory code lives in `notebooks/`. Stable, production-ready components are continuously migrated to `core/`.*

## 🗂 Repository Architecture

- `configs/` -> YAML configurations for reproducible experiments.
- `core/` -> Production-ready modules (Agents, Trainers, Utils).
- `data/` -> Raw OHLCV market data and processed features.
- `experiments/` -> Isolated run logs and model weights.
- `notebooks/` -> Prototyping and deep-dive learning.
- `research/` -> Obsidian PKM vault (Daily logs, Literature Review, Thesis Drafts).
- `scripts/` -> Execution scripts for the main pipeline.

## 📈 Engineering Philosophy

1. Understand the math before calling the API.
2. Design clean, modular systems.
3. Commit daily and document concepts thoroughly.