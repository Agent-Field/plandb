# Calculator API Design

## Overview

This service exposes a small JSON REST API for arithmetic operations.

## Endpoints

- `POST /calculate`
- `GET /health`

## Request Body

```json
{
  "operation": "add",
  "a": 10,
  "b": 5
}
```

## Supported Operations

- `add`
- `subtract`
- `multiply`
- `divide`

## Success Response

HTTP `200 OK`

```json
{
  "operation": "add",
  "a": 10,
  "b": 5,
  "result": 15
}
```

### Health Response

HTTP `200 OK`

```json
{
  "status": "ok"
}
```

## Error Handling

### Invalid JSON or missing fields

HTTP `400 Bad Request`

```json
{
  "error": "Invalid request",
  "message": "Request body must include numeric 'a', numeric 'b', and a supported 'operation'."
}
```

### Unsupported operation

HTTP `400 Bad Request`

```json
{
  "error": "Invalid request",
  "message": "Unsupported operation. Use add, subtract, multiply, or divide."
}
```

### Division by zero

HTTP `400 Bad Request`

```json
{
  "error": "Division by zero",
  "message": "The divisor 'b' must not be zero for division."
}
```

## Notes

- Inputs are accepted as integers or floats.
- Responses are JSON for both success and failure cases.
- Flask's test client will be used for pytest coverage.
