from app import create_app


def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_health_endpoint():
    response = client().get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_addition():
    response = client().post("/calculate", json={"operation": "add", "a": 10, "b": 5})

    assert response.status_code == 200
    assert response.get_json()["result"] == 15


def test_subtraction():
    response = client().post(
        "/calculate", json={"operation": "subtract", "a": 10, "b": 5}
    )

    assert response.status_code == 200
    assert response.get_json()["result"] == 5


def test_multiplication():
    response = client().post(
        "/calculate", json={"operation": "multiply", "a": 10, "b": 5}
    )

    assert response.status_code == 200
    assert response.get_json()["result"] == 50


def test_division():
    response = client().post(
        "/calculate", json={"operation": "divide", "a": 10, "b": 5}
    )

    assert response.status_code == 200
    assert response.get_json()["result"] == 2


def test_division_by_zero():
    response = client().post(
        "/calculate", json={"operation": "divide", "a": 10, "b": 0}
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Division by zero",
        "message": "The divisor 'b' must not be zero for division.",
    }


def test_unsupported_operation():
    response = client().post("/calculate", json={"operation": "mod", "a": 10, "b": 3})

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Invalid request",
        "message": "Unsupported operation. Use add, subtract, multiply, or divide.",
    }


def test_missing_fields():
    response = client().post("/calculate", json={"operation": "add", "a": 10})

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Invalid request",
        "message": "Request body must include numeric 'a', numeric 'b', and a supported 'operation'.",
    }


def test_invalid_json_body():
    response = client().post(
        "/calculate",
        data="not-json",
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Invalid request",
        "message": "Request body must include numeric 'a', numeric 'b', and a supported 'operation'.",
    }
