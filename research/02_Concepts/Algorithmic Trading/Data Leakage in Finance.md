---
tags:
  - concept
  - algo_trading
  - risk_management
  - testing
---
# Data Leakage in Financial Machine Learning

## Definition
Data Leakage (specifically **Look-ahead Bias**) occurs when a model inadvertently uses information from the future to make predictions. This creates an illusion of high performance during backtesting, leading to catastrophic financial losses live.

## Defending Against Leakage via Unit Testing
Beyond disabling `shuffle=True`, software engineering principles mandate unit testing to prevent indexing errors from leaking future data into historical states.
**The Leakage Test:** The last element of state $x$ must strictly precede the first element of target $y$ chronologically.
```python
def test_data_leakage():
    data = torch.arange(100).view(100, 1).float() # Sequential data
    # ... initialize dataset ...
    x_0, y_0 = dataset[0]
    # Mathematically prove no overlap:
    assert y_0[0, 0] == x_0[-1, 0] + 1
```

## Related Concepts
- [[Tautological Testing]]