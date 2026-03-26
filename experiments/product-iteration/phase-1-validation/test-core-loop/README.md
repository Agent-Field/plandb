# Fibonacci Module

This project provides two Python implementations of the Fibonacci sequence:

- `fibonacci_iterative(n)` uses a loop.
- `fibonacci_recursive(n)` uses straightforward recursion.

Both functions return the `n`th Fibonacci number for non-negative integers and
raise `ValueError` for negative input.

## Usage

```python
from fibonacci import fibonacci_iterative, fibonacci_recursive

print(fibonacci_iterative(10))  # 55
print(fibonacci_recursive(10))  # 55
```

You can also run the module directly:

```bash
python3 fibonacci.py
```

## Tests

Run the test suite with:

```bash
pytest
```
