---
tags: [concept, pytorch, gradients, autograd]
aliases: [Automatic Differentiation]
---
# Autograd in PyTorch

## 📌 Definition
`Autograd` is PyTorch’s automatic differentiation engine that powers neural network training. It completely abstracts the complex calculus required for backpropagation by tracking operations on tensors and computing gradients automatically.

## ⚙️ How It Works
When you set `requires_grad=True` on a tensor, PyTorch starts tracking every operation performed on it. 
- During the **forward pass**, it calculates the output and simultaneously builds a [[Computational Graphs|Computational Graph]] in the background.
- During the **backward pass** (triggered by calling `.backward()` on the loss tensor), Autograd traverses this graph in reverse, applying the chain rule of calculus to compute the gradients.

## 💻 Key Mechanics
- **`tensor.grad`**: After calling `.backward()`, the computed gradients are accumulated in the `.grad` attribute of the respective leaf tensors.
- **Gradient Accumulation**: By default, PyTorch *adds* gradients to the `.grad` attribute on multiple backward passes. This is why we must call `optimizer.zero_grad()` before a new step, unless we are intentionally doing [[Gradient Accumulation]].
- **`torch.no_grad()`**: A context manager that disables gradient tracking. Essential during model evaluation or manual weight updates to prevent memory leaks and speed up computations.

## 🔗 Related Topics
- [[Computational Graphs]]
- [[In-Place Operations]]
- [[Leaf Nodes]]
- [[Optimization Algorithms]]