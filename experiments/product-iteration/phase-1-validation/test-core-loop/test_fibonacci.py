import pytest

from fibonacci import fibonacci_iterative, fibonacci_recursive


@pytest.mark.parametrize(
    ("function", "n", "expected"),
    [
        (fibonacci_iterative, 0, 0),
        (fibonacci_iterative, 1, 1),
        (fibonacci_iterative, 7, 13),
        (fibonacci_iterative, 10, 55),
        (fibonacci_recursive, 0, 0),
        (fibonacci_recursive, 1, 1),
        (fibonacci_recursive, 7, 13),
        (fibonacci_recursive, 10, 55),
    ],
)
def test_fibonacci_values(function, n, expected):
    assert function(n) == expected


@pytest.mark.parametrize("function", [fibonacci_iterative, fibonacci_recursive])
def test_negative_input_raises_value_error(function):
    with pytest.raises(ValueError):
        function(-1)
