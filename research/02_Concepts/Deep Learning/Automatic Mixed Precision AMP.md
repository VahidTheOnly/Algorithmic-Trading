---
tags: [concept, deep_learning, hardware, optimization]
aliases: [AMP, Mixed Precision, FP16]
---
# Automatic Mixed Precision (AMP)

## 📌 Definition
Automatic Mixed Precision (AMP) is a hardware-acceleration technique that speeds up training and reduces VRAM usage[cite: 498]. It works by dynamically executing operations in Half-Precision (`float16`) where it's safe (e.g., matrix multiplications) while keeping critical operations in Single-Precision (`float32`) to maintain numerical stability (e.g., gradient accumulations and loss calculations).

## ⚙️ Core Components in PyTorch
1. **`torch.autocast`**: A context manager that automatically chooses the correct precision for each forward pass operation.
2. **`GradScaler` (`torch.amp.GradScaler`)**: The most important component. Because `float16` has a narrow representable range, small gradient values can underflow (become zero). The Scaler multiplies the loss by a large factor before backpropagation, then un-scales the gradients before the optimizer updates the weights.

## 💻 Implementation Template
```python
scaler = torch.amp.GradScaler('cuda')

for inputs, targets in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in mixed precision
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
    # Backward pass with scaled gradients
    scaler.scale(loss).backward()
    
    # Unscale and step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
````

## 🔗 Related Topics

- [[Gradient Clipping]]
- [[Model Checkpointing Systems]]