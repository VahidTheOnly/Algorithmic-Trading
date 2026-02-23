---
tags: [concept, data_engineering, algorithmic_trading, deep_learning]
aliases: [Data Scaling, Standardization]
---
# Feature Normalization in Trading

## 📌 Definition
Feature Normalization is the process of scaling input data to a standard range or distribution. Deep Learning optimizers (like Adam or SGD) converge much faster and avoid local minima when all input features have a similar scale. 

## ⚖️ Types of Normalization
1. **Z-Score Standardization**: Centers data around a mean of $0$ with a standard deviation of $1$. Best for data with outliers (like financial returns).
   $$z = \frac{x - \mu}{\sigma}$$
2. **Min-Max Scaling**: Scales data to a fixed range, usually $[0, 1]$ or $[-1, 1]$. Often used for volumes or bounding indicators like RSI.

## ⚠️ The Look-Ahead Bias (Crucial for Trading)
In algorithmic trading, you **cannot** normalize your dataset using the global mean and standard deviation of the entire dataset. Doing so introduces **Look-Ahead Bias (Data Leakage)**, meaning your agent uses future data statistics to make decisions in the past.

**Solution: Rolling Normalization**
To properly scale financial time-series for RL agents, compute the rolling mean and variance using only a backward-looking window (e.g., the last 100 periods).

## 🔗 Related Topics
- [[Look_Ahead_Bias]]
- [[Optimizer_Mechanics]]