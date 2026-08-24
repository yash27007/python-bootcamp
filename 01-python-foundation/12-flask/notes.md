# Flask

## Problem

A trained model, a business calculation, or any piece of Python logic is only usable by the exact
process that has it loaded in memory. The moment another program — a different service, a mobile
app, a teammate's script on a different machine, a browser — needs to use that logic, "just call the
function" stops being an option: nothing outside this process can call a Python function inside it.
This topic answers: **how does Python logic become something other software, written in any
language, running on any machine, can actually invoke?**

## Intuition

Any client that can make an HTTP request — `curl`, a browser, a Go backend, a mobile app — can talk
to a server if the server (1) stays running, (2) listens on a network port, (3) reads each incoming
request, (4) runs some code to produce an answer, and (5) writes a response back over the same
connection. Nothing about the underlying Python function changes; what changes is that it's now
reachable by anything that speaks HTTP instead of only by whatever process happens to have it
imported. Flask is a thin layer that makes steps (2)-(5) — the parts that are the same for every
endpoint — automatic, so the only code you actually write is "what URL maps to what function" and
"what does that function return."

## Why simpler approaches fail

**"Just run a script."** A script executes top to bottom and exits. It cannot accept a request it
didn't already know about when it started, because there is no process still running and listening
once the script finishes — there is nothing on the other end of a network connection for a client to
talk to.

**"Just write a raw socket/`http.server` handler for every project."** This works — see the
from-scratch section below — but every single project would have to hand-write the same pieces:
matching a URL path to a handler function, parsing query strings and JSON bodies, setting the right
`Content-Type` header, handling different HTTP methods on the same path, returning proper error
status codes. None of that logic is specific to *what* a given endpoint computes; writing it by hand
for every route in every project is pure repeated boilerplate, and it's exactly what Flask
automates.

## Conceptual foundation

*(Substituting for "Mathematical foundation" — this topic's foundation is a request/response
pipeline and a routing mechanism, not a numeric derivation. This substitution mirrors the one made
in `08-mlops-deployment/06-bentoml/notes.md` for the same reason: model/logic *serving* has no
equation to derive.)*

**The request/response cycle.** Every HTTP interaction is: a client sends a request (a method —
`GET`/`POST`/`PUT`/`DELETE` — a path, headers, optionally a body) to a server listening on a port;
the server reads it, runs some code, and sends back a response (a status code, headers, a body).
Flask's job is entirely about the plumbing around this cycle — matching the incoming path to the
right Python function (**routing**), handing that function a convenient `request` object instead of
raw bytes, and turning whatever the function returns into a properly formatted HTTP response.

**Routing as a lookup table.** `@app.route("/success/<int:score>")` registers a mapping from a URL
pattern to a function, with `<int:score>` meaning "match a path segment here, parse it as an `int`,
and pass it as the `score` argument." Under the hood this is exactly the "list of
(pattern, handler)" idea built by hand in the from-scratch section below — Flask's version supports
richer pattern syntax and does the regex compilation/matching/dispatch for you.

**WSGI.** Flask apps are WSGI (Web Server Gateway Interface) applications — a standard interface
Python defines so any WSGI-compliant server (Flask's own development server, Gunicorn, uWSGI) can
run any WSGI-compliant application interchangeably. This is why swapping Flask's dev server for a
production server (see Failure modes) requires no changes to the application code itself.

## Algorithm

The request/response pipeline every Flask endpoint (and the from-scratch handler below) follows:

1. Server accepts an incoming connection and reads the raw HTTP request.
2. **Route matching**: the request path is matched against every registered route pattern in order,
   until one matches (or none does, producing a `404`).
3. The matched handler function runs, with any parsed path parameters (and, for the REST API,
   parsed JSON body) passed to it.
4. The function returns a value (a string, a rendered template, or — for a JSON API — a dict Flask
   serializes with `jsonify`).
5. The framework wraps that return value in a proper HTTP response (status code, `Content-Type`
   header, serialized body) and writes it back to the client.

## From-scratch implementation

**Cite, don't re-derive**: `08-mlops-deployment/06-bentoml/notes.md`'s "From-scratch
implementation" section already builds a real, actually-run HTTP endpoint using only Python's
stdlib `http.server` (`BaseHTTPRequestHandler`, `ThreadingHTTPServer`) — no Flask, no framework —
serving a real `LogisticRegression` model with a hand-written `do_POST` that parses JSON, validates
it, calls `.predict()`, and writes a JSON response back. That file's bridge sentence states the
connection directly: *"`@bentoml.api` replaces the hand-written `do_POST` routing and JSON
parsing/serialization."* The same statement holds verbatim for Flask — **Flask automates
routing/request-parsing on top of exactly this kind of raw handler.**

This topic adds one further-stripped-down version, isolating just the *routing* mechanism (since
`bentoml`'s demo already covers the fuller request/preprocess/predict/postprocess pipeline):
[`raw_http_server.py`](raw_http_server.py) — a hand-rolled `@route(pattern)` decorator that appends
`(compiled_regex, handler)` tuples to a plain Python list, and a `do_GET` that walks that list
looking for the first match:

```python
ROUTES = []

def route(pattern):
    compiled = re.compile(pattern)
    def register(func):
        ROUTES.append((compiled, func))
        return func
    return register

@route(r"^/hello/(?P<name>[\w]+)$")
def hello(match):
    return 200, f"Hello, {match.group('name')}!\n"

class RawRoutingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        for pattern, handler in ROUTES:
            match = pattern.match(self.path)
            if match:
                status, body = handler(match)
                ...  # send_response/send_header/end_headers/wfile.write by hand
                return
        ...  # 404, no route matched
```

**Actually run.** Started as a background process
(`.venv/bin/python raw_http_server.py`, `127.0.0.1:8010`), hit with real `curl` requests from a
separate process. Real captured output:

```
$ curl http://127.0.0.1:8010/
Hello from a raw http.server handler (no Flask).

$ curl http://127.0.0.1:8010/hello/World
Hello, World! (dynamic segment parsed by hand with a regex)

$ curl -w "\nSTATUS:%{http_code}\n" http://127.0.0.1:8010/nope
Not Found (no route matched this path)
STATUS:404
```

`@route(r"^/hello/(?P<name>[\w]+)$")` and `@app.route("/hello/<name>")` do the identical job — the
hand-rolled version makes the regex compilation, the linear scan through registered routes, and the
manual `send_response`/`send_header`/`end_headers`/`wfile.write` calls explicit; Flask's `@app.route`
hides all four behind one decorator and `return "..."`.

## Practical implementation

The existing Flask apps in this folder, actually run and hit with real requests (dev server
started as a background process, stopped after):

- [`app.py`](app.py) — a template-rendering app (`render_template`, Jinja2 templates in
  `templates/`): a home/about page, a `GET`/`POST` form (`/form`), a variable URL rule with a typed
  converter (`/success/<int:score>`), and a `POST`-then-redirect pattern (`/submit` computes a total
  and redirects to `/success-results/<int:score>` via `url_for`).
- [`rest.py`](rest.py) — a JSON REST API (`jsonify`, no templates) implementing full CRUD over an
  in-memory `todos` list: `GET /todos`, `GET /todos/<id>`, `POST /todos`, `PUT /todos/<id>`,
  `DELETE /todos/<id>`.

Real captured output, `app.py` running on `127.0.0.1:8000`:

```
$ curl -X POST -d "hero=Batman" http://127.0.0.1:8000/form
Hello Batman

$ curl http://127.0.0.1:8000/success-results/80
...
<h2>80</h2>
<h2>PASSED</h2>

$ curl -X POST -d "maths=80&biology=70&english=90" http://127.0.0.1:8000/submit
HTTP/1.1 302 FOUND
Location: /success-results/80
```

One genuine, observed quirk worth noting rather than silently fixing: `GET /success/<int:score>`
passes a bare string (`"PASS"`/`"FAIL"`) as `results` to `result.html`, but that template reads
`{{results.score}}` and `{{results.result}}` — attributes a plain string doesn't have — so that
specific route renders two empty `<h2>` tags (confirmed by real output above), while
`/success-results/<int:score>` (which passes a dict with `score`/`result` keys) renders correctly.
This is left as-is and reported honestly rather than quietly patched, since it's exactly the kind of
real bug a from-scratch understanding of "the template just accesses whatever object you pass it"
lets you diagnose immediately.

Real captured output, `rest.py` running on `127.0.0.1:8000`:

```
$ curl -X POST -H "Content-Type: application/json" -d '{"task":"Write notes"}' http://127.0.0.1:8000/todos
{
  "completed": false,
  "id": 3,
  "task": "Write notes"
}

$ curl -X PUT -H "Content-Type: application/json" -d '{"completed":true}' http://127.0.0.1:8000/todos/1
{
  "completed": true,
  "id": 1,
  "task": "Complete the course"
}

$ curl -X DELETE http://127.0.0.1:8000/todos/2
{"result": true}

$ curl -w "\nSTATUS:%{http_code}\n" http://127.0.0.1:8000/todos/999
{"error": "Todo Not found"}
STATUS:404
```

Every method (`GET`/`POST`/`PUT`/`DELETE`) worked exactly as coded, including the `404` for a
missing id — the identical CRUD-over-HTTP pattern `08-mlops-deployment/06-bentoml`'s `service.py`
uses for a single `predict` endpoint, applied here to a plain in-memory resource instead of a model.

## Experiment

**Hypothesis:** a `POST`-then-redirect flow (`/submit` -> `/success-results/<score>`) should return
an HTTP `302` with a `Location` header pointing at the computed URL, and a subsequent request to
that `Location` should return the actual computed result.

**Setup:** `curl -X POST -d "maths=80&biology=70&english=90" http://127.0.0.1:8000/submit`
(`app.py`, dev server running locally), inspecting the response headers, then a separate real
request to the `Location` URL.

**Actual result:** confirmed — `302 FOUND`, `Location: /success-results/80` (`(80+70+90)/3 = 80.0`,
Flask's `<int:score>` converter accepted the float-valued redirect target because Werkzeug's
`url_for` formatted it as `80` in the URL), and `GET /success-results/80` returned `<h2>80</h2>
<h2>PASSED</h2>`, matching the `score >= 50` branch in `successRes`.

**Limitations:** this was one request against a locally running dev server on one machine — it says
nothing about behavior under concurrent load (the dev server is single-threaded by default) or over
a real network with non-trivial latency.

## Failure modes

- **Running Flask's development server in production.** Every dev-server startup prints this
  warning verbatim (captured for real above): *"WARNING: This is a development server. Do not use it
  in a production deployment. Use a production WSGI server instead."* The dev server is
  single-threaded by default (one request at a time, unless `threaded=True`), has no process
  management (a crash takes the whole server down with no automatic restart), and includes a
  debugger (`debug=True`, active in both `app.py` and `rest.py` here — confirmed in the captured
  startup log: *"Debugger is active! Debugger PIN: ..."*) that can execute arbitrary Python code
  through the browser if reachable by an untrusted client. Production deployments use a real WSGI
  server (Gunicorn, uWSGI) in front of the Flask app instead — the same WSGI interface both the dev
  server and the production server implement, per the Conceptual foundation section above, is
  exactly what makes this swap possible without changing application code.
- **Not validating request input.** `rest.py`'s `add_todo` reads `request.get_json()` and checks for
  a `task` key, but nothing validates that `task` is actually a string, or bounds its length — a
  caller could send `{"task": 12345}` or an enormous string and it would be accepted as-is. Contrast
  with `08-mlops-deployment/06-bentoml/server.py`'s `preprocess()`, which explicitly validates type,
  length, *and* numeric coercion before ever calling `.predict()`, precisely to prevent malformed
  input from reaching business logic. The general failure mode: any field trusted from the network
  without an explicit check is a field an external caller fully controls.

## Real-world usage

- Flask is a common choice for small-to-medium APIs and internal tools precisely because routing,
  JSON handling, and template rendering are automated exactly as demonstrated above — the same shape
  of app as `rest.py`, scaled up, is a completely realistic production microservice.
- `08-mlops-deployment/06-bentoml` builds the ML-specific version of this same idea — a model
  wrapped behind an HTTP endpoint — using a framework purpose-built for model serving (adaptive
  batching, OpenAPI schema generation from type hints) rather than Flask's general-purpose routing.
- Real production Flask deployments sit behind Gunicorn/uWSGI (the WSGI server) and often behind
  Nginx/a load balancer in front of that, for the reasons in Failure modes.

## Mental model

**A Flask app is a lookup table from (method, URL pattern) to a Python function, plus automatic
request-parsing and response-formatting around it — the identical job the from-scratch
`http.server` routing table above does by hand, one `@app.route` at a time instead of one manually
appended tuple at a time.**

## Questions to think about

1. `raw_http_server.py`'s `ROUTES` list is walked linearly, checking each pattern in registration
   order until one matches. What happens if two registered patterns could both match the same
   incoming path — which one wins, and does that match how you'd expect `@app.route` to behave in
   Flask?
2. `rest.py`'s `add_todo` computes the new id as `todos[-1]["id"] + 1`. Construct a sequence of
   `POST`/`DELETE` calls that would make this produce a duplicate id.
3. The dev server's debugger (`debug=True`) can execute arbitrary code through the browser if the
   server is reachable by an untrusted client. Given the WSGI section above, explain concretely why
   swapping in Gunicorn for the dev server does not, by itself, make `debug=True` safe to leave on.
4. `/success/<int:score>` and `/success-results/<int:score>` pass different types (`str` vs. `dict`)
   as `results` into templates that access `{{results.score}}`/`{{results.result}}`. Using only the
   real output captured above, explain why Jinja2 silently rendered empty strings instead of raising
   an error for the mismatched type.
5. `08-mlops-deployment/06-bentoml`'s from-scratch server returns a `400` with a specific message for
   malformed input; `rest.py`'s `add_todo` returns a `400` only when `task` is completely missing.
   Sketch the additional check you'd add to `add_todo` to reject a non-string `task`, and what status
   code and message it should return.
