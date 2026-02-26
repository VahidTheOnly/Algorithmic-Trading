---
tags:
  - concept
  - pytorch
  - engineering
---
# PyTorch Custom Datasets and DataLoaders

## Definition
The `torch.utils.data.Dataset` is an abstract class representing a dataset. A custom dataset must inherit from this class and override specific magic methods to create a standard interface for the `DataLoader`, which handles batching, shuffling, and multi-process data loading.

## Core Magic Methods (Map-Style)
1.  `__init__(self, ...)`: Instantiates the dataset object, loads raw data into memory (or creates pointers to disk files for lazy loading), and sets hyperparameters.
2.  `__len__(self)`: Returns the absolute number of *valid* samples in the dataset. Calling `len(self)` inside this method causes a `RecursionError`.
3.  `__getitem__(self, idx)`: A pure function that maps an integer index to a specific data sample and its corresponding target/label.

## Engineering Constraint: Statelessness
The `__getitem__` method must be **stateless**. It should not mutate class attributes (e.g., `self.current_idx = idx`). When `DataLoader` uses multiple workers (`num_workers > 0`), shared states lead to race conditions, causing silent data corruption.

## Minimal Implementation
```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        # Pure mapping logic
        x = self.data[idx]
        y = self.data[idx + 1]
        return x, y
```

## Related Concepts
- [[Sliding Window Mechanism]]