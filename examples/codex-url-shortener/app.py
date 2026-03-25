from __future__ import annotations

import os
from urllib.parse import urlparse

from flask import Flask, abort, jsonify, redirect, request

from storage import SQLiteURLRepository


def is_valid_url(candidate: str) -> bool:
    parsed = urlparse(candidate)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)

    if test_config is None:
        test_config = {}

    app.config.from_mapping(
        DATABASE=os.path.join(app.instance_path, "url_shortener.db"),
    )
    app.config.update(test_config)

    os.makedirs(app.instance_path, exist_ok=True)
    repository = SQLiteURLRepository(app.config["DATABASE"])
    app.config["REPOSITORY"] = repository

    @app.post("/api/shorten")
    def shorten_url():
        payload = request.get_json(silent=True) or {}
        long_url = payload.get("url", "").strip()
        if not long_url or not is_valid_url(long_url):
            return jsonify({"error": "A valid http or https URL is required."}), 400

        record, created = repository.create_or_get(long_url)
        status_code = 201 if created else 200
        return (
            jsonify(
                {
                    "code": record.short_code,
                    "short_url": request.host_url.rstrip("/") + f"/{record.short_code}",
                    "url": record.long_url,
                }
            ),
            status_code,
        )

    @app.get("/api/urls/<code>")
    def get_url(code: str):
        record = repository.get_by_short_code(code)
        if record is None:
            return jsonify({"error": "Short code not found."}), 404
        return jsonify(
            {
                "code": record.short_code,
                "url": record.long_url,
                "created_at": record.created_at,
            }
        )

    @app.get("/<code>")
    def resolve(code: str):
        record = repository.get_by_short_code(code)
        if record is None:
            abort(404)
        return redirect(record.long_url, code=302)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
