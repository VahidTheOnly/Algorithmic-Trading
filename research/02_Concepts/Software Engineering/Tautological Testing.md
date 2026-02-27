---
tags:
  - concept
  - software_engineering
  - testing
---
# Tautological Testing (Anti-Pattern)


## Definition
A Tautological Test is a flawed testing practice where the test merely repeats the exact same logic or formula used in the production code. Consequently, if the underlying logic is flawed, the test will still pass (False Positive), providing a dangerous illusion of correctness.

## The Problem
```python
# BAD (Tautological)
expected_len = len(data) - look_back - look_ahead + 1
assert len(dataset) == expected_len

```

If the engineer's mental model of the formula is wrong, both the code and the test share the exact same bug.

## The Solution: Implementation-Agnostic Testing

A robust unit test treats the code as a black box and relies on hardcoded, manually verified scenarios.

```python
# GOOD (Deterministic & Hardcoded)
data_length = 10
look_back = 3
look_ahead = 2
# Math on paper: 10 - 3 - 2 + 1 = 6 windows
assert len(dataset) == 6, "Expected exactly 6 windows"

```

## Related Concepts

- [[Data Leakage in Finance]]
