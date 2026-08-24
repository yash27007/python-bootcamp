# 15 – Agent Skills and MCP

Knowledge disclosure → action standardization → the loop that ties them together: how an
agent is handed a large, growing library of capabilities without every one of them
permanently occupying context (Agent Skills / progressive disclosure), how it gets a
standardized way to actually call external tools and read external resources (Model
Context Protocol), and how a single working agent connects the two. This is the final
section of the curriculum's multi-agent/agentic track, closing the loop
`14-multi-agent-systems` opened. No live LLM call and no live external service call is
made anywhere in this section — no API keys are available or authorized in this
environment; agent decisions are deterministic Python heuristics, stated honestly.

| # | Topic | Status | Description |
|---|-------|--------|--------------|
| 01 | [Agent Skills and Progressive Disclosure](./01-agent-skills/) | ✅ Complete | A real `Skill`/`SkillRegistry` built from scratch, deterministic keyword-overlap selection, a real measured context-savings experiment (70.7%→90.2% at 5→50 skills), two reproduced selection failures |
| 02 | [Model Context Protocol (MCP)](./02-model-context-protocol/) | ✅ Complete | A real MCP-shaped JSON-RPC 2.0 server/client over a genuine local subprocess-stdio boundary, real tool discovery and invocation, a real schema-validation experiment |
| 03 | [Skills+MCP Agent Loop](./03-skills-and-mcp-agent-loop/) | ✅ Complete | Topics 01 and 02 reused verbatim and tied together in a real 4-stage agent loop, a real measured skill-narrowed vs. consider-everything experiment (0% vs. up to 80% wrong-tool rate), two reproduced integration failures — the curriculum's capstone connection |

## Prerequisites

- `14-multi-agent-systems/01-communication-protocols` — Topic 01's "Conceptual foundation"
  convention and Topic 02's FIPA-ACL-performative citation both build on this directly.
- `14-multi-agent-systems/02-orchestration-patterns` — Topic 03 names its loop as a direct,
  single-agent ($n=1$) instance of that topic's decompose→delegate→aggregate pattern.
- Comfort with plain Python classes, regex/string parsing, JSON, and basic
  `subprocess`/stdio I/O — no ML framework knowledge is required anywhere in this section.

## Environment note

No live external LLM API or MCP server call is made anywhere in this section — no API
keys are available or authorized in this environment. Skill selection and tool-argument
extraction are deterministic keyword/regex heuristics, an honest, explicitly stated
stand-in for what a real LLM-driven agent would do. Topic 02's server/client DOES run over
a genuine local subprocess-stdio boundary (real JSON-RPC 2.0 wire messages, real PIDs, no
network) — the one piece of this section that is not simulated, just kept entirely local.

## What's next

This is the final planned section of the curriculum (Phases 0–13). Together with
`14-multi-agent-systems`, it covers the full arc from a single from-scratch model
(`13-llms-from-scratch`) to multiple coordinating agents (`14-multi-agent-systems`) to an
agent equipped with a capability library and standardized tool access
(`15-agent-skills-and-mcp`) — the same three concerns any production agent framework has
to solve, built and run at toy scale throughout.
