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
1. `__init__(self, ...)`: Instantiates the dataset object, loads raw data into memory, and sets hyperparameters. **Engineering Standard:** Always use explicit Type Hinting for arguments.
2. `__len__(self) -> int`: Returns the absolute number of *valid* samples in the dataset.
3. `__getitem__(self, idx: int)`: A pure function that maps an integer index to a specific data sample and its corresponding target/label.

## Engineering Constraint: Statelessness & Feature Selection
The `__getitem__` method must be **stateless**. Furthermore, in financial datasets, input features $x$ and target variables $y$ must be strictly separated using indexing (`input_indices` vs `target_indices`) to prevent target leakage into the feature space.

## Related Concepts
- [[Sliding Window Mechanism]]
- [[Data Leakage in Finance]]