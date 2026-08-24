# Flask

## What you'll learn

How Flask turns Python logic into something reachable over HTTP by any client, any language, any
machine — the request/response cycle, routing as a URL-pattern-to-function lookup table, and WSGI
as the standard interface that lets a production server (Gunicorn) run the same app the dev server
does.

## Why it matters

A trained model, a business calculation, or any piece of Python logic is only usable by the exact
process that has it loaded — unusable by another service, a mobile app, or a teammate on a
different machine, until it's exposed over the network. Flask automates the boilerplate every HTTP
endpoint needs (URL matching, request parsing, response formatting) so only the endpoint-specific
logic needs to be written by hand.

## Prerequisites

- `08-mlops-deployment/06-bentoml` — the from-scratch stdlib `http.server` endpoint this topic's
  from-scratch section cites directly; read its "From-scratch implementation" section first for the
  fuller request/preprocess/predict/postprocess pipeline this topic's routing-only version builds
  on.
- `11-memory-management` — Flask's dev server is exactly the kind of long-running process where
  memory leaks matter.

## What you'll build

- A hand-rolled URL router using only stdlib `http.server` — a `@route(pattern)` decorator, a plain
  list of `(regex, handler)` tuples, and manual dispatch — actually run and hit with real `curl`
  requests
- The existing Flask apps (`app.py` template-rendering app, `rest.py` JSON REST API), actually run
  as a local dev server and hit with real requests (`GET`/`POST`/`PUT`/`DELETE`, a redirect flow,
  a 404 case) — including one genuine template-mismatch bug found by running it for real, not
  silently patched

See [`notes.md`](notes.md) for the full write-up including all real captured output,
[`raw_http_server.py`](raw_http_server.py) for the from-scratch router, and
[`app.py`](app.py)/[`rest.py`](rest.py) for the practical Flask apps.

## Where it shows up in real systems

Small-to-medium APIs and internal tools; `08-mlops-deployment/06-bentoml` builds the ML-specific
version of the same idea (a model behind an HTTP endpoint) using a framework purpose-built for model
serving instead of Flask's general-purpose routing.

## What's next

`13-streamlit` — a different way to expose Python logic interactively, this time as a UI rather than
an HTTP API — and this section's last topic.
