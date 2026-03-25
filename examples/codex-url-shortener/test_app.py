from __future__ import annotations

from pathlib import Path

import pytest

from app import create_app


@pytest.fixture
def client(tmp_path: Path):
    database_path = tmp_path / "test.db"
    app = create_app({"TESTING": True, "DATABASE": str(database_path)})
    with app.test_client() as client:
        yield client


def test_shorten_url_creates_new_code(client):
    response = client.post(
        "/api/shorten",
        json={"url": "https://example.com/articles/flask"},
    )

    assert response.status_code == 201
    assert response.json["code"] == "1"
    assert response.json["short_url"].endswith("/1")


def test_shorten_url_reuses_existing_code(client):
    first = client.post("/api/shorten", json={"url": "https://example.com/reused"})
    second = client.post("/api/shorten", json={"url": "https://example.com/reused"})

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json["code"] == second.json["code"]


def test_shorten_url_rejects_invalid_input(client):
    response = client.post("/api/shorten", json={"url": "not-a-url"})

    assert response.status_code == 400
    assert "valid http or https URL" in response.json["error"]


def test_redirects_to_original_url(client):
    create_response = client.post(
        "/api/shorten",
        json={"url": "https://example.com/redirect-target"},
    )

    response = client.get(f"/{create_response.json['code']}")

    assert response.status_code == 302
    assert response.headers["Location"] == "https://example.com/redirect-target"


def test_lookup_returns_metadata(client):
    create_response = client.post(
        "/api/shorten",
        json={"url": "https://example.com/metadata"},
    )

    response = client.get(f"/api/urls/{create_response.json['code']}")

    assert response.status_code == 200
    assert response.json["url"] == "https://example.com/metadata"
    assert response.json["code"] == create_response.json["code"]


def test_lookup_returns_404_for_missing_code(client):
    response = client.get("/api/urls/missing")

    assert response.status_code == 404
