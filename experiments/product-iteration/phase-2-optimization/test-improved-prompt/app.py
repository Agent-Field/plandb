from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request


SUPPORTED_OPERATIONS = {"add", "subtract", "multiply", "divide"}


def create_app() -> Flask:
    app = Flask(__name__)

    @app.post("/calculate")
    def calculate():
        payload = request.get_json(silent=True)
        validation_error = _validate_payload(payload)
        if validation_error is not None:
            return jsonify(validation_error), 400

        assert payload is not None
        operation = payload["operation"]
        a = payload["a"]
        b = payload["b"]

        if operation == "divide" and b == 0:
            return (
                jsonify(
                    {
                        "error": "Division by zero",
                        "message": "The divisor 'b' must not be zero for division.",
                    }
                ),
                400,
            )

        result = _perform_operation(operation, a, b)
        return jsonify({"operation": operation, "a": a, "b": b, "result": result})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


def _validate_payload(payload: Any) -> dict[str, str] | None:
    if not isinstance(payload, dict):
        return {
            "error": "Invalid request",
            "message": "Request body must include numeric 'a', numeric 'b', and a supported 'operation'.",
        }

    operation = payload.get("operation")
    a = payload.get("a")
    b = payload.get("b")

    if operation not in SUPPORTED_OPERATIONS:
        return {
            "error": "Invalid request",
            "message": "Unsupported operation. Use add, subtract, multiply, or divide.",
        }

    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return {
            "error": "Invalid request",
            "message": "Request body must include numeric 'a', numeric 'b', and a supported 'operation'.",
        }

    return None


def _perform_operation(operation: str, a: float, b: float) -> float:
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    return a / b


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
