# 02 – Model Context Protocol (MCP)

Detailed notes (MCP's real client-server architecture formalized precisely — the
real JSON-RPC 2.0 wire format, the three core primitives (tools, resources,
prompts), the discovery-before-call flow — explicitly tied back to
`14-multi-agent-systems/01-communication-protocols`'s FIPA-ACL-performative point
as the same underlying "standard message shape" need applied to tool use):
[notes.md](notes.md)

Real, actually-executed, from-scratch MCP-shaped server
([mcp_server.py](mcp_server.py)) and client, running over a genuine subprocess
boundary via stdio (no network, no external host) — three toy tools with real
JSON-Schema parameter definitions, real JSON-RPC 2.0 request/response transcripts
for `list_tools` and `call_tool`, and a real schema-validation experiment (one
accepted call, two rejected with proper JSON-RPC errors):
[001_json_rpc_server_client.ipynb](001_json_rpc_server_client.ipynb)

## What you'll learn

Why an agent needs a standardized way to actually *call external tools* and *read
external resources*, not just load instructions — and why bespoke,
per-agent-per-tool integration code doesn't scale ($N \times M$ pieces of glue vs.
$N + M$ under a shared protocol). Then: MCP's real architecture (client-server
over JSON-RPC 2.0, the tools/resources/prompts primitives, discovery before
invocation), a from-scratch server and client that actually run over a real
subprocess and speak real JSON-RPC, and a measured demonstration that
JSON-Schema validation rejects a malformed tool call before it ever reaches the
tool's own code.

## Why it matters

Topic 1 solved *knowledge* disclosure (an agent learning a capability exists and
loading its instructions only when needed). This topic solves the companion
problem for *action*: an agent standing in front of an external system it needs
to actually operate. Every later topic in `15-agent-skills-and-mcp` that touches
real external tool use builds on this topic's discovery-then-invoke vocabulary
and its JSON-RPC message shapes. It is also, like Topic 1, the first topic in
this pair to sit next to a real, currently running instance of the pattern it
teaches: this very session's own tool set includes deferred tools named
`mcp__claude_ai_Notion__*`, `mcp__claude_ai_Google_Calendar__*`, and others —
real, live MCP server integrations, observed by name (not called, per this
section's binding constraint) while writing this topic.

## Prerequisites

- `15-agent-skills-and-mcp/01-agent-skills` — for the progressive-disclosure
  vocabulary (always-visible summary tier vs. an expensive tier loaded only on
  selection) this topic's discovery-before-call flow directly parallels: a
  tool's `name`+`description`+`inputSchema` from `list_tools` is the
  always-visible tier; actually calling it is the expensive, gated step.
- `14-multi-agent-systems/01-communication-protocols` — for the FIPA-ACL
  performative argument (a standard message shape lets a receiver dispatch on
  intent, not guess from payload shape) this topic cites directly and applies to
  JSON-RPC's `method` field.
- Comfort with plain Python, JSON, and basic `subprocess`/stdio I/O — no ML
  framework knowledge is required for this topic.

## What you'll build

- [mcp_server.py](mcp_server.py) — a standalone MCP-shaped server script,
  registering three toy tools with real JSON-Schema `inputSchema` definitions:
  `add(a, b)`, `word_count(text)`, `reverse_string(text)`. Handles real
  JSON-RPC 2.0 `list_tools` and `call_tool` requests over newline-delimited
  JSON on stdin/stdout, including a hand-rolled JSON-Schema validator
  (required-field presence + type checking) gating every `call_tool` before
  the underlying Python function runs.
- A minimal JSON-RPC 2.0 client, built in the notebook, that spawns
  `mcp_server.py` as a **real subprocess** (`subprocess.Popen`, its own PID, its
  own stdin/stdout pipes — no network) and exchanges real JSON-RPC messages
  with it, printing the raw wire transcript for every call.
- A full discover-then-call run with real output: `list_tools()` returning all
  three tools' schemas, then `call_tool("add", {"a": 17, "b": 25})` →
  `value: 42`, `call_tool("word_count", ...)` → `value: 9`,
  `call_tool("reverse_string", ...)` → `value: "locotorp txetnoc ledom"`.
- A real schema-validation experiment: a well-formed `add` call accepted and
  executed (`value: 9`), a call missing the required `b` argument rejected with
  JSON-RPC error `-32602`, and a call with `a` as a string instead of a number
  rejected with the same error code — both rejections happening, verifiably,
  before the underlying `add()` function is ever invoked.

## Where it appears in real systems

Anthropic's **Model Context Protocol** is the real, named specification this
topic's server/client are a minimal, honest instance of — real JSON-RPC 2.0 over
stdio or HTTP+SSE, the same tools/resources/prompts primitive split, the same
discovery-before-invocation flow, implemented at production scale by official
SDKs (`@modelcontextprotocol/sdk`, `mcp`) and by real servers for systems like
GitHub, Slack, Postgres, Google Drive, and Notion. This repository's own tooling
runs on a real instance of it right now: the Claude Code session that wrote this
topic has deferred tools named `mcp__claude_ai_Notion__*` and
`mcp__claude_ai_Google_Calendar__*` in its own tool listing — live, connected
MCP servers this session could call, observed by name only, never invoked, per
this section's no-live-external-call constraint (see notes.md's "Real-world
usage").

## What's next

Later `15-agent-skills-and-mcp` topics build on this topic's JSON-RPC
request/response vocabulary and its discovery-then-invoke pattern — the same way
Topic 1's `Skill`/`SkillRegistry` primitives underpin knowledge disclosure, this
topic's server/client shape underpins everything that follows involving an agent
actually *acting* on an external system rather than only reasoning with loaded
instructions.
