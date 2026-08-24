# 03 – Putting It Together: A Toy Skills+MCP Agent Loop

Detailed notes (the minimal 4-stage agent loop formalized precisely — select a
skill, narrow to a tool via that skill's own pointers intersected with what the
live MCP server actually discovered, extract real arguments deterministically,
invoke — explicitly named as the $n=1$, single-agent instance of
`14-multi-agent-systems/02-orchestration-patterns`'s decompose→delegate→aggregate
pattern):
[notes.md](notes.md)

Real, actually-executed, from-scratch agent loop
([001_skills_mcp_agent_loop.ipynb](001_skills_mcp_agent_loop.ipynb)) — Topic 1's
`Skill`/`SkillRegistry` and Topic 2's JSON-RPC MCP client/server
([mcp_server.py](mcp_server.py), a local copy for self-containedness) reused
verbatim, tied together by one small hand-authored `SKILL_TO_TOOLS` map, run
end-to-end on three toy tasks with real output at every stage
(`add(17, 25) -> 42`, `word_count(...) -> 4`, `reverse_string(...) ->
'locotorp txetnoc ledom :txet siht'`), a real measured experiment comparing
skill-narrowed vs. consider-everything tool selection as a tool registry grows
(0% vs. 40%/80%/80% wrong-tool-invocation rate at $N=5/15/30$ tools), and two
concrete, reproduced failure modes (a skill pointing at a nonexistent tool,
caught by discovery before invocation; a regex argument extractor failing on
an unexpected phrasing).

## What you'll learn

Why skill selection (Topic 1) and tool discovery/invocation (Topic 2), each
solid on its own, still don't add up to a working agent without something
explicitly connecting them — and what that connecting piece looks like at its
simplest: a 4-stage loop (select → narrow+choose tool → extract arguments →
invoke) run on a real task string end to end. Then: a measured demonstration
that reusing the selected skill to narrow which tools are even considered
meaningfully reduces wrong-tool invocations as a tool registry grows, and two
concrete failure modes that this deterministic, non-LLM version of the loop
cannot avoid — instructive precisely because they show where the LLM-driven
real version buys real robustness.

## Why it matters

This is the last topic in `15-agent-skills-and-mcp`, and it's where the
section's two standalone pieces stop being standalone. Every real agent
framework that both selects among specialized capabilities *and* calls
external tools — including the Claude Code session that wrote this topic —
runs some version of exactly this loop, just with a language model in place of
the keyword heuristics used here. This topic is also the curriculum's own
explicit link back to `14-multi-agent-systems/02-orchestration-patterns`: the
loop built here is that topic's decompose→delegate→aggregate pattern, run by a
single agent against itself at $n=1$, named directly rather than left as an
implicit resemblance.

## Prerequisites

- `15-agent-skills-and-mcp/01-agent-skills` — the `Skill`/`SkillRegistry`
  classes and the deterministic keyword-overlap `select()` heuristic are
  reused verbatim, not rebuilt.
- `15-agent-skills-and-mcp/02-model-context-protocol` — the JSON-RPC 2.0
  subprocess client/server (`mcp_server.py`, `list_tools`/`call_tool`) is
  reused verbatim; this topic's `mcp_server.py` is a local copy of that
  topic's script for self-containedness.
- `14-multi-agent-systems/02-orchestration-patterns` — for the
  decompose→delegate→aggregate formalization this topic's loop is shown to be
  a direct, named instance of, at $n=1$.

## What you'll build

- Two on-topic toy skills (`arithmetic-calculation`, `text-analysis`)
  registered alongside three distractor skills carried over unchanged from
  Topic 1, so skill selection is a real 5-way choice.
- `SKILL_TO_TOOLS` — the small, explicit, hand-authored map from skill name to
  candidate MCP tool names that was **missing** from Topics 1 and 2
  individually; this is the piece that actually connects them.
- `agent_loop(task)` — the full 4-stage function (select skill → narrow +
  choose tool from the live `list_tools` result → extract arguments via regex
  → `call_tool` over a real subprocess), run on three toy tasks with real,
  printed, stage-by-stage output and real returned results (`42`, `4`, a
  reversed string).
- A real measured experiment: two tool-selection variants
  (skill-narrowed vs. consider-everything) run against a growing MCP tool
  registry ($N = 5, 15, 30$, extended with 27 programmatically-synthesized
  toy tools) — skill-narrowed holds a flat 0% wrong-tool-invocation rate at
  every $N$; consider-everything rises from 40% to 80%.
- Two concrete, reproduced failure modes: a skill (`code-review-checklist`)
  pointing at a tool (`lint_code`) that doesn't exist on this server, caught
  by the discovery step before any invocation is attempted; and a regex
  argument extractor failing outright on a task phrased with spelled-out
  numbers instead of digits.

## Where it appears in real systems

This loop, with a real LLM in place of the keyword heuristics and real
external MCP servers in place of the toy in-process one, is architecturally
what Claude Code itself, Anthropic's Agent SDK, and "tool-using LLM agent"
systems generally do — this is `15-agent-skills-and-mcp`'s own capstone
connection, stated directly in notes.md's "Real-world usage" rather than left
implicit. No live LLM call and no live external service call was made
anywhere in this topic, per this section's binding constraint; the same
deferred `mcp__claude_ai_*` tools named (never called) in Topic 2's notes.md
are named again here as the real, currently-running instance of the pattern
this loop is a toy version of.

## What's next

This is the final topic in `15-agent-skills-and-mcp`. Together, the three
topics in this section cover knowledge disclosure (Topic 1), action
standardization (Topic 2), and the loop that ties them into one working agent
(this topic) — the same three concerns any production agent framework has to
solve, at toy scale and without any live model or network call anywhere in the
section.
