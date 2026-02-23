---
tags: [concept, pytorch, memory_management, autograd]
aliases: [In-place Modifying]
---
# In-Place Operations in PyTorch

## 📌 Definition
An in-place operation is one that directly changes the data of a tensor without making a copy, thus saving memory. In PyTorch, these are typically denoted by functions ending with an underscore (e.g., `tensor.add_()`, `tensor.zero_()`) or by using augmented assignment operators (`+=`, `*=`).

## ⚠️ The Danger with Autograd
While in-place operations save memory, they are extremely dangerous when working with [[Autograd]]. If a tensor's value is modified in-place, the forward computational graph loses the original value. If the backward pass needs that original value to compute the derivative via the chain rule, PyTorch will throw a `RuntimeError`.

## 🛡️ Best Practices
- **Updating Weights:** When manually updating weights (e.g., `w -= lr * w.grad`), you **must** wrap it in `with torch.no_grad():`. Otherwise, PyTorch thinks you are trying to add a mathematical operation to the computational graph of a [[Leaf Nodes|Leaf Node]].
- **Clearing Gradients:** `optimizer.zero_grad()` uses in-place zeroing (`.zero_()`) safely because gradients themselves don't require gradients.

## 💻 Example
```python
# BAD: Will cause a RuntimeError during .backward()
w = torch.tensor([1.0], requires_grad=True)
y = w * 2
w += 1.0  # In-place modification of a leaf variable
y.backward() 

# GOOD: Safe weight update
with torch.no_grad():
    w -= 0.01 * w.grad    
```

## 🔗 Related Topics
- [[Autograd]]
- [[Computational Graphs]]
- [[Leaf Nodes]]