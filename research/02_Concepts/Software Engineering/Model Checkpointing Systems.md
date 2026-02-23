---
tags: [concept, software_engineering, mlops, state_management]
aliases: [Saving Models, Checkpoints]
---
# Model Checkpointing Systems

## 📌 Definition
A Model Checkpointing System is an engineering pattern used to save the complete state of a training process to disk at specific intervals. This allows for resuming interrupted training, performing inference later, or evaluating intermediate models (early stopping).

## ⚠️ Beyond the Model Weights
A common beginner mistake is saving *only* the model weights. To truly resume training—especially in Continual Learning frameworks—you must save a composite dictionary containing all active states.

### What Must Be Saved:
1. **`model.state_dict()`**: The actual learned weights and buffers of the neural network.
2. **`optimizer.state_dict()`**: Crucial for optimizers like Adam, which maintain moving averages (momentum and variance) for every single parameter. Without this, the optimizer resets and destabilizes the resumed training.
3. **`scaler.state_dict()`**: If using [[Automatic Mixed Precision AMP]], the `GradScaler`'s scaling factor must be preserved.
4. **`epoch` / `step`**: The current training iteration.

## 💻 Implementation Example
```python
# Saving
checkpoint = {
    'epoch': current_epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scaler_state': scaler.state_dict()
}
torch.save(checkpoint, 'checkpoints/model_epoch_10.pth')

# Loading
checkpoint = torch.load('checkpoints/model_epoch_10.pth')
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
scaler.load_state_dict(checkpoint['scaler_state'])
````

## 🔗 Related Topics

- [[PyTorch Module Internals]]
- [[YAML Config Architecture]]