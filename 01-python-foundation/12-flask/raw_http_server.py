"""
From-scratch: a minimal HTTP request handler using only the standard library
(`http.server`), with hand-rolled routing -- BEFORE Flask's `@app.route`
automates exactly this.

This is the same technique `08-mlops-deployment/06-bentoml/server.py` uses to serve
a real model over HTTP without any framework -- see that file's notes.md
("From-scratch implementation") for the fuller version (model loading,
preprocess/predict/postprocess). This file strips it down further, to just the
routing/dispatch mechanism Flask automates, since that's this topic's point.

Run directly: .venv/bin/python raw_http_server.py
Then, from another terminal:
    curl http://127.0.0.1:8010/
    curl http://127.0.0.1:8010/hello/World
    curl http://127.0.0.1:8010/nope        -> 404
"""
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

# The routing table Flask's @app.route decorator builds for you automatically.
# Here it's built by hand: a list of (compiled regex pattern, handler function).
ROUTES = []


def route(pattern):
    """A hand-rolled version of Flask's @app.route -- decorator that registers
    a URL pattern -> handler-function mapping in ROUTES, nothing more."""
    compiled = re.compile(pattern)

    def register(func):
        ROUTES.append((compiled, func))
        return func

    return register


@route(r"^/$")
def index(match):
    return 200, "Hello from a raw http.server handler (no Flask).\n"


@route(r"^/hello/(?P<name>[\w]+)$")
def hello(match):
    name = match.group("name")
    return 200, f"Hello, {name}! (dynamic segment parsed by hand with a regex)\n"


class RawRoutingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Manual dispatch: try each registered route in order until one matches
        # the request path -- this loop, plus the ROUTES list above, is exactly
        # what Flask's routing/URL-matching machinery automates.
        for pattern, handler in ROUTES:
            match = pattern.match(self.path)
            if match:
                status, body = handler(match)
                self.send_response(status)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(body.encode())
                return
        # No route matched.
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Not Found (no route matched this path)\n")

    def log_message(self, format, *args):
        pass  # keep demo output quiet


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8010), RawRoutingHandler)
    print("raw_http_server.py listening on http://127.0.0.1:8010")
    server.serve_forever()
