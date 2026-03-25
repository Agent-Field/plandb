# URL Shortener Service

Small Flask service that shortens URLs and stores them in SQLite.

## Research summary

- Hash-based codes can be deterministic for identical URLs, but collision handling and code-length changes add complexity.
- Counter-based codes are straightforward with SQLite: insert a row, encode the row ID in Base62, and persist the result.
- For a single-node local service, counter-based generation is simpler and more predictable.

## API design

### `POST /api/shorten`

Creates a short URL for a submitted long URL.

Request body:

```json
{
  "url": "https://example.com/some/long/path"
}
```

Response body:

```json
{
  "code": "b",
  "short_url": "http://localhost/b",
  "url": "https://example.com/some/long/path"
}
```

- Returns `201` when a new short code is created.
- Returns `200` when the URL already exists and its previous code is reused.
- Returns `400` for invalid or missing URLs.

### `GET /<code>`

Redirects to the stored long URL with status `302`.

### `GET /api/urls/<code>`

Returns JSON metadata for a stored short code.

## Storage design

SQLite table:

```sql
CREATE TABLE urls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  long_url TEXT NOT NULL UNIQUE,
  short_code TEXT UNIQUE,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

Flow:

1. Validate the incoming URL.
2. Reuse an existing row when the long URL has already been shortened.
3. Otherwise insert a new row, encode its integer ID to Base62, and update `short_code`.
4. Resolve redirects by looking up `short_code`.

## Running

```bash
python3 -m flask --app app run
```

## Testing

```bash
python3 -m pytest
```
