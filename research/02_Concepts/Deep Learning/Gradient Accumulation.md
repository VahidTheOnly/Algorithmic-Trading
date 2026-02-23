---
tags: [concept, deep_learning, optimization, memory_management]
aliases: [Accumulated Gradients]
---
# Gradient Accumulation

## 📌 Definition
Gradient Accumulation is an engineering technique used to bypass hardware memory limitations. It simulates training with a large batch size by running multiple forward and backward passes with smaller micro-batches, accumulating their gradients, and only updating the model weights after $N$ steps.

## 🧮 The Logic
Hardware (like a standard GPU) might only fit a batch size of 16 into VRAM, but statistical stability in RL might require a batch size of 64. 
Instead of crashing with out-of-memory (OOM) errors, we process 4 consecutive batches of 16.

**Algorithm:**
1. Forward pass micro-batch.
2. Compute loss. **(Divide loss by $N$ if averaging is required)**
3. `loss.backward()` (PyTorch naturally adds this to existing `.grad`).
4. Repeat 1-3 for $N$ steps.
5. Call `optimizer.step()`.
6. Call `optimizer.zero_grad()` to clear for the next macro-batch.

## 💻 PyTorch Implementation Template
```python
accumulation_steps = 4
for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward() # Accumulates gradients
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 🔗 Related Topics
- [[Autograd]]
- [[Automatic Mixed Precision AMP]]
- [[Optimizer_Mechanics]]