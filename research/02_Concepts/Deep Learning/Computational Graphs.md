---
tags: [concept, deep_learning, math, architecture]
aliases: [DAG, Dynamic Computation Graph]
---
# Computational Graphs

## 📌 Definition
A Computational Graph is a Directed Acyclic Graph (DAG) mathematical model used to represent algorithms. In Deep Learning, they are used to compute derivatives automatically via the chain rule. 


## 🏗️ Structure
- **Nodes**: Represent variables (tensors, scalars) or mathematical operations (addition, matrix multiplication).
- **Edges**: Represent the flow of data (tensors) from one operation to the next.

## 🔄 Static vs. Dynamic Graphs
- **Static Graphs (e.g., TensorFlow 1.x)**: Define the entire graph architecture first, compile it, and then feed data into it. Faster, but harder to debug and less flexible for varying sequence lengths.
- **Dynamic Graphs (Define-by-Run) (e.g., PyTorch)**: The graph is built on-the-fly during the forward pass. Every iteration can have a completely different graph structure. This is highly beneficial for Reinforcement Learning (RL) and Recurrent Neural Networks (RNNs).

## ⚠️ Memory Implications
In PyTorch, the graph is destroyed immediately after `.backward()` is called to free up GPU memory (VRAM). If you need to backpropagate through the same graph multiple times (e.g., in some Actor-Critic RL algorithms), you must use `loss.backward(retain_graph=True)`.

## 🔗 Related Topics
- [[Autograd]]
- [[Gradient Clipping]]