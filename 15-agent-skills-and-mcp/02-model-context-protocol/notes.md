# 02 — Model Context Protocol (MCP)

## Problem

Topic 1 (`01-agent-skills`) solved how an agent knows a *capability exists* and
loads its *instructions* only when needed — progressive disclosure for
knowledge. But instructions alone don't let an agent *do* anything in the
world: read a file, query a database, hit a calendar API, run a calculation
outside its own weights. For that, an agent needs to actually **call external
tools** and **read external resources** — and it needs to do so through some
concrete, standardized message format, the same way `14-multi-agent-systems`
established that two agents need a standard message shape before they can
exchange *anything* meaningful (that topic's notes.md: FIPA-ACL's
**performatives**, a fixed vocabulary of communicative-intent labels like
`inform`/`request`/`propose`, exist so "a receiver can dispatch on *intent*
without having to parse or guess what the sender meant from the payload's
shape alone"). The problem this topic solves is the same underlying need —
a standard message shape — applied specifically to **tool use**: how does an
agent discover what tools/resources/prompts an external system offers, and
call them, without every agent and every tool inventing its own bespoke way of
talking to each other?

## Intuition

Imagine every appliance in your house needing its own uniquely shaped wall
socket — the toaster's plug doesn't fit the lamp's outlet, the lamp's outlet
doesn't fit the fridge's socket. You'd need a different wall built for every
appliance. A standard electrical outlet solves this once: any compliant
appliance plugs into any compliant outlet, because both sides agree on a
fixed physical/electrical contract (voltage, plug shape) regardless of what's
on either end.

**MCP (Model Context Protocol)** is that standard socket for AI agents and
external tools/data. An MCP *server* exposes tools ("things you can call"),
resources ("things you can read"), and prompts ("templates you can reuse")
behind one fixed wire format. An MCP *client* (the agent's harness) speaks
that same fixed format to *any* compliant server — a calculator server, a
calendar server, a database server — without needing custom glue code
written specifically for each one.

## Why simpler approaches fail

**"Just write custom integration code for each agent-tool pairing."** This is
what teams do before adopting a shared protocol, and it's the natural first
instinct: write a Python function that calls the calendar API directly, wire
it into this one agent, done. It breaks down for a purely combinatorial
reason:

- With $N$ agents (or agent frameworks) and $M$ external tools, a bespoke
  integration for every pairing that actually needs to talk to a tool
  requires up to $N \times M$ pieces of glue code in the worst case — each
  one hand-written, hand-maintained, and hand-tested against that tool's
  particular quirks (its auth scheme, its error format, its argument
  naming). Add a new tool, and potentially every agent needs new integration
  code. Add a new agent framework, and potentially every tool needs
  re-wiring for it.
- A standard protocol turns this into an $N + M$ problem: each of the $M$
  tools is built once as a compliant MCP *server* (regardless of which
  agents will ever use it), and each of the $N$ agent frameworks is built
  once as a compliant MCP *client* (regardless of which tools it will ever
  talk to). Any client can then talk to any server, because both sides
  target the same fixed contract instead of each other directly. This is
  exactly the electrical-outlet intuition above, formalized: $N \times M$
  custom adapters collapse to $N + M$ standard plugs.
- Beyond the raw count, bespoke integrations also don't compose well even
  pairwise: without a standard discovery mechanism, an agent that wants to
  know "what can this tool actually do, and what arguments does it expect"
  has no uniform way to ask — it either hard-codes that knowledge (brittle
  the moment the tool changes) or the tool author has to hand-write
  bespoke documentation for every agent framework that might integrate with
  it. MCP's `list_tools` (see below) makes discovery itself part of the
  standard, not an afterthought each integration reinvents.

## Conceptual foundation (documented substitution for "Mathematical foundation")

Like `14-multi-agent-systems` and `01-agent-skills`, this is a
systems/architecture topic — what follows is a precise structural
definition of MCP's real architecture, not a derivation.

### The wire format: JSON-RPC 2.0

MCP messages are real [JSON-RPC 2.0](https://www.jsonrpc.org/specification)
— a lightweight remote-procedure-call protocol encoded as JSON. Three
message shapes matter here:

**Request** (client → server, expects a response):

```json
{"jsonrpc": "2.0", "id": 1, "method": "list_tools", "params": {}}
```

**Success response** (server → client):

```json
{"jsonrpc": "2.0", "id": 1, "result": {"tools": [...]}}
```

**Error response** (server → client):

```json
{"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "invalid params"}}
```

Every field matters: `jsonrpc` pins the protocol version (`"2.0"`, always);
`id` correlates a response back to the request that produced it (the client
may have several requests in flight); `method` names the operation being
invoked; `params` carries that operation's arguments; a response carries
*either* `result` (success) *or* `error` (failure) — never both. This is
precisely the "standard message shape" argument from the Problem section,
made concrete: a JSON-RPC `method` name is doing exactly the job a FIPA-ACL
*performative* does — telling the receiver what kind of thing is being asked
for, so it can dispatch on that field alone rather than guessing from the
payload's shape.

### The three MCP primitives

An MCP server exposes some combination of three primitive kinds of capability:

1. **Tools** — invocable functions. Each tool has a `name`, a human-readable
   `description`, and an `inputSchema` — a real JSON-Schema object
   (`{"type": "object", "properties": {...}, "required": [...]}`) declaring
   exactly what arguments a call needs and what type each one must be. A
   client discovers a server's tools via the `list_tools` method, then
   invokes one via `call_tool` (passing `name` and `arguments`).
2. **Resources** — readable data the server exposes (a file, a database
   row, a document) — addressed by a URI, discovered via `list_resources`,
   read via `read_resource`. Not implemented in this topic's from-scratch
   code (only tools are implemented — see below), but structurally
   identical to tools in shape: a discovery method paired with an action
   method.
3. **Prompts** — reusable prompt templates the server provides, so a client
   doesn't have to hand-author a prompt for a workflow the server author
   already knows how to phrase well. Discovered via `list_prompts`, fetched
   via `get_prompt`. Also not implemented here, for the same reason: this
   topic's scope is the tools primitive, since "calling an external
   function with validated arguments" is the sharpest illustration of what
   a standard protocol buys over bespoke integration.

### The discovery-before-call flow

A well-behaved MCP client never calls `call_tool` (or `read_resource`, or
`get_prompt`) blind. It first calls the corresponding `list_*` method to
learn what the server actually offers and what shape each call must take:

```
client -> server:  list_tools()
server -> client:  {tools: [{name, description, inputSchema}, ...]}
client -> server:  call_tool(name="add", arguments={"a": 17, "b": 25})
server -> client:  {result: {value: 42}}
```

This ordering matters for the same reason a REST client fetches an OpenAPI
spec (or a GraphQL client introspects a schema) before making arbitrary
calls: without discovery, the client would have to already know, out of
band, exactly which tools exist and exactly what arguments each one takes —
which is precisely the bespoke-integration problem this whole topic exists
to avoid.

## Algorithm

**Server-side (registration and dispatch):**
1. Register each tool as `(name, description, inputSchema)` in a lookup
   table, alongside the actual Python function that implements it.
2. On a `list_tools` request, return every registered tool's `name`,
   `description`, and `inputSchema` — nothing more (the function
   implementation itself is never sent to the client; only its contract
   is).
3. On a `call_tool` request: look up the named tool; if it doesn't exist,
   return a JSON-RPC error. Otherwise, validate `arguments` against that
   tool's `inputSchema` — check every `required` field is present, and that
   present fields match their declared `type`. If validation fails, return
   a JSON-RPC error *without ever calling the underlying function*. If
   validation passes, call the function, and wrap its return value in a
   JSON-RPC `result`.

**Client-side (discovery then invocation):**
1. Send a `list_tools` request; parse the returned tool list and each
   tool's `inputSchema`.
2. Construct a `call_tool` request whose `arguments` are shaped to match
   the chosen tool's `inputSchema` (in a real agent system, an LLM reads
   the schema and produces matching arguments; see the "Honest
   substitution" note under From-scratch implementation).
3. Send the request; branch on whether the response carries `result` or
   `error`.

## From-scratch implementation

Built and actually run in
[001_json_rpc_server_client.ipynb](001_json_rpc_server_client.ipynb), backed
by [mcp_server.py](mcp_server.py):

- **A real server, in a real subprocess.** `mcp_server.py` is a standalone
  Python script, spawned by the notebook via `subprocess.Popen` — a genuine
  child process with its own PID, its own stdin/stdout, communicating with
  the notebook (the client) purely over those two pipes. No network socket
  is opened anywhere by either side. This is the real MCP **stdio
  transport**: newline-delimited JSON-RPC 2.0 messages, one request per
  line written to the child's stdin, one response per line read back from
  its stdout.
- **Three registered tools**, each with a real JSON-Schema `inputSchema`:
  `add(a, b)` (both `number`, both `required`), `word_count(text)` (`text`
  is `string`, `required`), and `reverse_string(text)` (same shape as
  `word_count`).
- **Two real JSON-RPC methods**: `list_tools` (discovery — returns all
  three tools' names, descriptions, and schemas) and `call_tool`
  (invocation — validates arguments against the target tool's schema,
  then dispatches to the underlying Python function).
- **A minimal client**, `send_request()` in the notebook, that builds a real
  JSON-RPC 2.0 request dict, serializes it, writes it to the subprocess's
  stdin, reads one line back from its stdout, and parses it as JSON —
  printing the raw wire transcript (`--> SENT` / `<-- RECEIVED`) for every
  call, not just a summary of the result.
- **A full discover-then-call run, with real output:** `list_tools()`
  returned all three tools (real transcript,
  `{"jsonrpc": "2.0", "id": 1, "method": "list_tools", "params": {}}` sent,
  a `result` with all three tool specs received); `call_tool("add", {"a":
  17, "b": 25})` returned `{"result": {"content": [...], "value": 42}}`;
  `call_tool("word_count", {"text": "the quick brown fox jumps over the
  lazy dog"})` returned `value: 9`; `call_tool("reverse_string", {"text":
  "model context protocol"})` returned `value: "locotorp txetnoc ledom"`.
  The subprocess was shut down cleanly at the end (`proc.stdin.close();
  proc.wait()`), exiting with code `0`.

**Honest disclosure — real subprocess, not a fallback.** This topic's
binding constraint required trying a real subprocess-over-stdio approach
first, falling back to an in-process JSON-RPC-shaped simulation only if the
subprocess approach proved unreliable. The subprocess approach was tried
first here and worked cleanly on the first attempt (see the notebook's real
transcripts) — **no fallback was needed.** The client and server in this
notebook are two genuinely separate OS processes exchanging real JSON-RPC
2.0 messages over real stdio pipes, which is the actual MCP wire protocol
and transport, not merely a same-process simulation of its message shapes.

**Honest substitution statement — tool *selection* is not implemented.**
This topic's server and client demonstrate discovery and invocation
faithfully, but the *choice* of which tool to call, with which arguments,
for a given natural-language task, is not made here by any model — the
notebook picks the tool and arguments explicitly in each cell (e.g. "now
call `add` with `a=17, b=25`"). In a real agent system, an LLM reads the
`list_tools` response (names, descriptions, JSON schemas) and decides which
tool to call and how to fill in `arguments`, the same substitution
`01-agent-skills`' notes.md made explicit for skill *selection*. This
section's no-live-LLM-call constraint makes that piece unavailable here;
what's demonstrated instead is the *protocol mechanics* a real
model-driven client would sit on top of — discovery, schema-validated
invocation, and error handling — faithfully and with real output.

## Practical implementation

There is no separate "swap in a production library" step for this topic in
the way earlier from-scratch-then-practical topics have one, for the same
reason `01-agent-skills` didn't need one either: the from-scratch
implementation above *is* a real, if minimal, instance of the actual
production wire format (JSON-RPC 2.0) and the actual production transport
(stdio) that Anthropic's real MCP SDKs (`@modelcontextprotocol/sdk` for
TypeScript, `mcp` for Python) implement at production scale. The mapping is
direct:

| From-scratch (this topic) | Real system (Anthropic's MCP SDKs) |
|---|---|
| `mcp_server.py`'s `TOOLS` dict + `handle_request()` | An `mcp.server.Server` (or `McpServer` in TS) registering tools via decorators/`server.tool(...)` |
| Hand-rolled `validate_arguments()` (required + type check) | Full JSON-Schema validation, typically via `pydantic` (Python SDK) or `zod` (TS SDK) |
| Newline-delimited JSON over a `subprocess.Popen` pipe | The real stdio `Transport` class in the official SDKs (same framing) — or, for remote servers, an HTTP+SSE transport instead of stdio |
| `send_request()` in the notebook | An `mcp.client.session.ClientSession`, wrapping request/response correlation, timeouts, and typed results |
| This topic's `list_tools`/`call_tool` only | The full primitive set: `list_tools`/`call_tool`, `list_resources`/`read_resource`, `list_prompts`/`get_prompt`, plus an `initialize` handshake this topic omits for simplicity |

The one piece a production client adds that this topic's client doesn't: an
`initialize` handshake at connection start (capability negotiation and
protocol-version exchange — see Failure modes, "protocol version mismatch",
below) before any `list_tools`/`call_tool` traffic.

## Experiment

**Hypothesis:** the server's `call_tool` handler validates a call's
`arguments` against the target tool's JSON-Schema `inputSchema` — checking
required-field presence and type — *before* it ever dispatches to the
underlying Python function. A well-formed call should be accepted and
executed; a call missing a required argument, or with a wrong-typed
argument, should be rejected with a JSON-RPC error, and the underlying
function should never run for either rejected case.

**Setup:** three real `call_tool` requests against the running subprocess
server, all targeting `add(a, b)` (`inputSchema` requires both `a` and `b`
to be numbers):
1. `{"a": 4, "b": 5}` — well-formed.
2. `{"a": 4}` — missing the required `b` argument.
3. `{"a": "four", "b": 5}` — `a` present but wrong type (`str`, not
   `number`).

**Actual result (real notebook output):**

| Case | Arguments | Outcome | Detail |
|---|---|---|---|
| 1 (valid) | `{"a": 4, "b": 5}` | **Accepted** | `result.value = 9` |
| 2 (missing arg) | `{"a": 4}` | **Rejected** | `error.code = -32602`, message `"invalid params for tool 'add': missing required argument: 'b'"` |
| 3 (wrong type) | `{"a": "four", "b": 5}` | **Rejected** | `error.code = -32602`, message `"invalid params for tool 'add': argument 'a' expected type 'number', got 'str'"` |

Reading `mcp_server.py`'s `handle_request()` confirms *why* this holds
structurally, not just empirically: the `return make_error(...)` for a
validation failure is reached and returned before the line `value =
TOOL_FUNCTIONS[tool_name](arguments)` is ever executed — the rejected calls
never touched `call_add()` at all.

**Interpretation:** the hypothesis held exactly as stated. Schema validation
is a real gate sitting between "a client asked to call a tool" and "the
tool's actual code runs" — not cosmetic, and not something a well-formed
call ever pays a cost for (Case 1 executed and returned normally).

**Limitations:** the validator implemented here (`validate_arguments()`) is
deliberately basic — required-field presence and a single-level type check
against `{"type": "number"/"string"/"boolean"/"object"/"array"}` only. It
does not validate nested object/array schemas, `enum` constraints, numeric
`minimum`/`maximum`, string `pattern`/`format`, or reject unknown
properties (`additionalProperties: false` is not enforced — an unexpected
extra key is silently ignored, not rejected). A production MCP server
would typically delegate to a real JSON-Schema validation library (Python's
`jsonschema`, or a `pydantic`-model-derived schema) rather than this
hand-rolled check, which would catch strictly more malformed calls (e.g. a
negative number where only positive integers are allowed) than this
minimal version does.

## Failure modes

**1. Missing/malformed required argument (demonstrated above).** Exactly
the Experiment's Cases 2 and 3: a client omits a required field, or sends
one with the wrong JSON type. In a real agent system, this happens when the
LLM constructing the `call_tool` request misreads or hallucinates the
schema — e.g. inferring that a `date` argument is optional when the schema
marks it `required`, or passing a string like `"5"` where the schema
demands a JSON number `5`. JSON-Schema validation at the server is exactly
the mitigation demonstrated in the Experiment: it converts a would-be
silent failure or crash inside the tool function into an explicit, typed
JSON-RPC error the client (and the LLM reasoning about what to do next) can
see and react to.

**2. Ambiguous tool description — a concrete, plausible example.** Suppose
a server registers two tools: `search(query)`, described as *"Search for
information"*, and `lookup(query)`, described as *"Find information about a
topic."* Both descriptions are vague and nearly synonymous — neither says
what corpus is searched (the web? a local knowledge base? a specific
database?), what `query` should look like (a keyword? a natural-language
question?), or how the two tools actually differ in scope or behavior. A
real agent selecting between them from `list_tools`'s output alone (name +
description + schema, no access to the implementation) has no reliable
signal to prefer one over the other for a given task — this is structurally
the same failure `01-agent-skills`' notes.md documented for two
overlapping skill descriptions (`sql-query-builder` vs.
`sql-schema-reviewer`), now at the tool-selection layer instead of the
skill-selection layer. The fix is the same in both cases: a tool's
`description` needs to state, concretely, what it does and when to prefer
it over a similarly-named alternative — a vague one-line description is a
design defect in the server, not something the calling agent can reliably
correct for after the fact.

**3. Protocol version mismatch (discussed, not demonstrated).** MCP, like
most evolving protocols, has a version identifier exchanged during
connection setup (the `initialize` handshake mentioned under "Practical
implementation," omitted from this topic's minimal client/server for
simplicity). If a client built against protocol version $X$ connects to a
server that only understands version $Y$ ($X \neq Y$), the two sides may
disagree about which methods exist, what fields a given message type must
carry, or how errors are shaped — a client might send a `call_tool` request
shaped for a newer schema convention the server predates, and get a
confusing error (or, worse, a response the client can't parse at all)
rather than a clean "version mismatch" rejection. This is not demonstrated
in this topic's implementation, because the from-scratch server and client
here are always the same version of the same script talking to itself — but
it is a real, documented failure mode of any versioned RPC protocol, and
part of why a real `initialize` handshake exists: to let both sides agree
explicitly on a protocol version *before* any tool traffic, so a mismatch
surfaces immediately and legibly rather than as a downstream parsing
failure.

## Real-world usage

- **Anthropic's Model Context Protocol** — the real, named specification
  this topic's server/client are a minimal, faithful instance of: a
  client-server protocol over JSON-RPC 2.0, with tools/resources/prompts as
  its three core primitives, stdio and HTTP+SSE as its two standard
  transports. Anthropic publishes official SDKs (`@modelcontextprotocol/sdk`
  for TypeScript, `mcp` for Python) implementing exactly this contract at
  production scale — real servers exist today for GitHub, Google Drive,
  Slack, Postgres, and many other systems, all speaking this one protocol.
- **This very session's own tooling** — a real, live instance of exactly
  this protocol is running in the environment this topic was written in.
  This session's deferred-tool listing includes names like
  `mcp__claude_ai_Notion__notion-search`,
  `mcp__claude_ai_Google_Calendar__list_events`, and
  `mcp__claude_ai_Google_Drive__authenticate` — the `mcp__<server>__<tool>`
  naming convention itself is Claude Code's own evidence that these are
  tools discovered from real, connected MCP servers (Notion, Google
  Calendar, Google Drive, Figma, Gmail, and others), each one presumably
  exposing its own `list_tools` response the harness folded into this
  session's available tool set. This observation is made by name only —
  citing that these deferred tools exist and are named this way — without
  calling any of them, per this section's binding no-live-external-call
  constraint.
- **Any plugin ecosystem connecting one model/agent runtime to many
  third-party systems** (IDE extensions that expose project context to an
  AI assistant, chat platforms wiring bots to external services) faces the
  identical $N \times M \to N + M$ shape this topic's "Why simpler
  approaches fail" section argues from first principles — MCP is one
  concrete, now widely adopted, standardization of that same underlying
  need for the LLM-agent-and-tools case specifically.

## Mental model

**A tool is a labeled, contract-bearing electrical socket, not a bespoke
wire someone soldered in.** `list_tools` is reading the socket's label
before you plug anything in — what voltage it expects (`inputSchema`), what
it's for (`description`). `call_tool` is plugging in and drawing power —
and the socket itself refuses a mismatched plug (schema validation) before
any current flows, rather than letting a bad connection damage whatever's
on the other end.

## Questions to think about

1. This topic's client picks which tool to call and what arguments to pass
   explicitly, in each notebook cell — the "Honest substitution statement"
   above says a real system has an LLM do this instead, reading
   `list_tools`'s output. What specifically would that LLM need from a
   tool's `description` and `inputSchema` to make a *correct* choice, that
   this topic's plain Python dict lookup didn't need at all?
2. The Experiment demonstrated that a missing or wrong-typed argument is
   caught by `validate_arguments()` before `call_add()`/etc. ever runs. Its
   "Limitations" note that `additionalProperties: false` is not enforced —
   an unexpected extra argument key is silently ignored. Construct a
   concrete scenario (a tool, a schema, a malicious or merely buggy extra
   argument) where silently ignoring an unexpected key, rather than
   rejecting it, causes an incorrect result even though every *required*
   field was present and correctly typed.
3. "Why simpler approaches fail" argues bespoke integration is an
   $N \times M$ problem that a standard protocol turns into $N + M$. Under
   what condition (in terms of $N$ and $M$) does adopting a shared protocol
   *not* pay off compared to just writing the bespoke integrations directly
   — i.e., when might $N + M$ (plus the fixed cost of implementing protocol
   compliance itself on both sides) actually exceed a small $N \times M$?
4. Failure mode 2 (ambiguous tool descriptions, `search` vs. `lookup`) was
   called structurally identical to `01-agent-skills`' overlapping-skill-
   description failure. Both are instances of "the disclosed-but-not-loaded
   summary tier is where the calling side actually has to make its
   decision." Given that parallel, propose one concrete addition to MCP's
   `inputSchema`/`description` convention (not a new selection algorithm)
   that would reduce this class of ambiguity for tool descriptions
   specifically, that wouldn't obviously transfer back to fixing skill
   descriptions the same way.
5. The "Protocol version mismatch" failure mode was discussed but not
   demonstrated. Design a minimal, concrete experiment (what you'd add to
   `mcp_server.py` and the notebook's client, what request/response you'd
   send, what output you'd expect) that *would* actually demonstrate a
   version mismatch being caught during an `initialize` handshake, using
   only this topic's existing subprocess-over-stdio infrastructure.
