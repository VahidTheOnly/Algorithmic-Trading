---
tags: [concept, pytorch, computational_graph, autograd]
aliases: [Leaf Tensors]
---
# Leaf Nodes (Leaf Tensors)

## 📌 Definition
In a PyTorch [[Computational Graphs|Computational Graph]], a **Leaf Node** is a tensor that was created directly by the user (or initialized by PyTorch's `nn.Module`), not as the result of a mathematical operation tracked by Autograd. 



## 🔑 Key Characteristics
- Tensors with `requires_grad=False` are always leaf nodes.
- Tensors with `requires_grad=True` are leaf nodes **only if** they were created directly by the user (e.g., `w = torch.randn(5, requires_grad=True)`).
- If a tensor is the output of an operation (e.g., `y = w * x`), it is an **Intermediate Node**, not a leaf node.

## ⚙️ Why Do They Matter?
During the `.backward()` pass, PyTorch only populates the `.grad` attribute for **Leaf Nodes** that have `requires_grad=True`. Intermediate nodes have their gradients computed during backprop, but PyTorch immediately discards them to save memory. 

If you ever need to inspect the gradient of an intermediate node for debugging, you must use `tensor.retain_grad()`.

## 🔗 Related Topics
- [[Autograd]]
- [[Computational Graphs]]
- [[In-Place Operations]]