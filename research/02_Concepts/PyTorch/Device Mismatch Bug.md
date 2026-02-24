---
aliases: [Silent Bug, Optimizer Device Mismatch]
tags: [pytorch, debugging, tensor-mechanics]
date: 2026-02-24
status: verified
---

# Device Mismatch and Optimizer Pointers in PyTorch

## The Core Issue
A highly common and critical "silent bug" occurs when a model's parameters are passed to an optimizer *before* the model is moved to a target device (e.g., GPU). This relates directly to how PyTorch handles [[Computational Graphs]] and [[Leaf Nodes]].

## Mechanism
When `optimizer = torch.optim.SGD(model.parameters(), lr=0.1)` is executed, the optimizer stores direct memory pointers to the current tensors, which initially reside on the CPU. 
If `model.to(device)` is called *afterward*, PyTorch allocates new memory and creates new tensors on the GPU. The optimizer is now completely disconnected from the active computational graph and will continuously attempt to update the orphaned CPU tensors. Loss will stagnate, or it will throw a device mismatch runtime error during backpropagation.

## Engineering Standard
**Rule:** Always cast the model to the target hardware device before initializing the optimizer. This ensures the optimizer tracks the correct [[Autograd]] tracking nodes on the GPU.

---
**Related Concepts:**
- [[Day_004_Experiment_Infrastructure]]
- [[In-Place Operations]]