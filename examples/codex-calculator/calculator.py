#!/usr/bin/env python3

import argparse
from typing import Callable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A simple command-line calculator."
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    for operation in ("add", "subtract", "multiply", "divide"):
        operation_parser = subparsers.add_parser(
            operation, help=f"{operation} two numbers"
        )
        operation_parser.add_argument("left", type=float)
        operation_parser.add_argument("right", type=float)

    return parser


def add(left: float, right: float) -> float:
    return left + right


def subtract(left: float, right: float) -> float:
    return left - right


def multiply(left: float, right: float) -> float:
    return left * right


def divide(left: float, right: float) -> float:
    if right == 0:
        raise ValueError("Cannot divide by zero.")
    return left / right


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    operations: dict[str, Callable[[float, float], float]] = {
        "add": add,
        "subtract": subtract,
        "multiply": multiply,
        "divide": divide,
    }

    try:
        result = operations[args.operation](args.left, args.right)
    except ValueError as error:
        parser.exit(status=1, message=f"{error}\n")

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
