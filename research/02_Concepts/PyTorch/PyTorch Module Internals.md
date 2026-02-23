---
tags: [concept, pytorch, architecture, oop]
aliases: [nn.Module]
---
# PyTorch Module Internals

## 📌 Definition
`torch.nn.Module` is the base class for all neural network models in PyTorch. It provides an Object-Oriented interface to manage layers, weights, biases, and the forward propagation logic.

## 🏗️ Core Components

### 1. `nn.Parameter`
When you assign a tensor wrapped in `nn.Parameter()` to a module, it is automatically added to the module's list of parameters. These are the tensors that the optimizer will update during training.

### 2. `register_buffer()`
Buffers are tensors that belong to the model's state but are **not** updated by the optimizer (e.g., running mean in BatchNorm, or a fixed `profit_target` in a custom loss).
- They are moved to the GPU alongside the model when calling `.to(device)`.
- They are saved in the `state_dict` during checkpointing.

### 3. `state_dict()`
A Python dictionary mapping each layer to its parameter/buffer tensor. This is what you actually save to disk when checkpointing a model, not the class object itself.

## 🔄 `train()` vs `eval()` modes
Calling `model.train()` or `model.eval()` does not magically run the training loop. It simply switches the internal boolean flags of certain layers (like `Dropout` and `BatchNorm`) so they behave correctly during inference versus training.

## 🔗 Related Topics
- [[Model Checkpointing Systems]]