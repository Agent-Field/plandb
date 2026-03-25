"""Fibonacci number implementations."""


def _validate_input(n: int) -> None:
    if n < 0:
        raise ValueError("n must be non-negative")


def fibonacci_iterative(n: int) -> int:
    """Return the nth Fibonacci number using iteration."""
    _validate_input(n)
    if n < 2:
        return n

    previous, current = 0, 1
    for _ in range(2, n + 1):
        previous, current = current, previous + current
    return current


def fibonacci_recursive(n: int) -> int:
    """Return the nth Fibonacci number using recursion."""
    _validate_input(n)
    if n < 2:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


if __name__ == "__main__":
    for index in range(10):
        print(f"{index}: {fibonacci_iterative(index)}")
