---
date: 2026-02-26
tags:
  - daily_log
  - pytorch
  - data_engineering
  - time_series
project: Thesis_Multi_Agent_RL
phase: 1
chapter: 1.6
---
# Day 005: Time-Series Data Pipeline Engineering

## 1. Mission Summary
Today's objective was to transition from flat tensor data to a mathematically sound, rolling window architecture suitable for financial time-series forecasting and Reinforcement Learning state representation. I successfully implemented a custom PyTorch `Dataset` and integrated it with a `DataLoader` for batching.

## 2. Implementation Progress
* **Basic Map-Style Dataset**: Implemented `BasicMarketDataset` to map indices to next-step targets.
* **Sliding Window Architecture**: Developed `SlidingWindowDataset` integrating both `look_back` ($W$) and `look_ahead` ($H$) parameters.
* **Batching Integration**: Wrapped the dataset in a `torch.utils.data.DataLoader` (with `shuffle=False`) and successfully verified output tensor dimensions (e.g., `[32, 4, 5]` for $X$ and `[32, 3, 5]` for $y$).

## 3. Key Engineering Insights & Math
* **Boundary Mathematics (Off-by-One Error)**: The total valid samples in a dataset of length $T$ with a lookback $W$ and lookahead $H$ is strictly $T - W - H + 1$. Omitting the `+ 1` results in truncating the final valid trading window.
* **Time-Centric Indexing ($t$)**: Instead of treating `idx` as the start of the window, `idx` was mathematically mapped to $t$ (current time). This aligns precisely with RL formulations where the agent observes $S_t$ to predict or act upon $y_t$.
* **Stateless `__getitem__`**: Realized that storing $t$ as a class attribute (`self.t = ...`) is a critical flaw. In multi-processing environments (`num_workers > 0`), this causes race conditions. `__getitem__` must remain strictly pure and stateless.

## 4. Related Concepts
* [[Custom Datasets and DataLoaders]]
* [[Sliding Window Mechanism]]
* [[Data Leakage in Finance]]

## 5. Next Steps
Move towards deeper neural network architectures (MLP, CNN) that can consume these 3D batched tensors, transitioning to Chapter 2 of the roadmap.