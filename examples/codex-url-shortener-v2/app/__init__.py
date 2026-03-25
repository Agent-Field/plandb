from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, abort, jsonify, redirect, render_template, request


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        DATABASE=Path(app.root_path).parent / "shortener.db",
        SECRET_KEY="dev",
        TESTING=False,
    )

    if test_config:
        app.config.update(test_config)

    from .storage import create_or_get_short_url, get_url_by_code, init_db

    init_db(app.config["DATABASE"])

    def is_valid_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def wants_json_response() -> bool:
        if request.is_json:
            return True
        accepted = request.accept_mimetypes
        return accepted.best == "application/json" and accepted[accepted.best] > accepted["text/html"]

    def build_short_url(short_code: str) -> str:
        return request.host_url.rstrip("/") + f"/{short_code}"

    @app.get("/")
    def index() -> str:
        return render_template("index.html", short_url=None, error=None, original_url="")

    @app.post("/shorten")
    def shorten():
        payload = request.get_json(silent=True) if request.is_json else request.form
        original_url = (payload.get("url") or "").strip()

        if not is_valid_url(original_url):
            if wants_json_response():
                return jsonify({"error": "Please provide a valid http or https URL."}), 400
            return (
                render_template(
                    "index.html",
                    short_url=None,
                    error="Please provide a valid http or https URL.",
                    original_url=original_url,
                ),
                400,
            )

        row = create_or_get_short_url(app.config["DATABASE"], original_url)
        short_url = build_short_url(row["short_code"])

        if wants_json_response():
            return jsonify(
                {
                    "long_url": row["long_url"],
                    "short_code": row["short_code"],
                    "short_url": short_url,
                }
            )

        return render_template(
            "index.html",
            short_url=short_url,
            error=None,
            original_url=row["long_url"],
        )

    @app.get("/<code>")
    def redirect_to_url(code: str):
        row = get_url_by_code(app.config["DATABASE"], code)
        if row is None:
            abort(404)
        return redirect(row["long_url"])

    return app
