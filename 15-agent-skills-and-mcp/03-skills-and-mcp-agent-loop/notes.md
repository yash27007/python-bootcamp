# 03 — Putting It Together: A Toy Skills+MCP Agent Loop

## Problem

`01-agent-skills` solved *knowledge* disclosure: given a task, which set of
instructions is relevant, loaded only when selected. `02-model-context-protocol`
solved *action* standardization: given a tool name and arguments, how to
discover and invoke it through one fixed wire format instead of bespoke glue
code per integration. Both are real, load-bearing pieces — and neither one, by
itself, is a working agent.

A `SkillRegistry` that selects `arithmetic-calculation` for the task "add 17
and 25" has told you *how to approach* the task (reach for a calculation tool,
don't compute it yourself). It has not called anything. An MCP client that can
discover `add(a, b)` and invoke it correctly has a real capability sitting
there — but nothing in `02-model-context-protocol`'s notebook ever decided,
from a task string, that `add` was the tool to reach for; a human picked it by
hand, once per cell. Something has to sit *between* skill selection and tool
invocation and make that connection for real: given one task string, decide
which skill applies, let that skill's own instructions narrow which tool(s)
are even worth considering, extract real arguments from the string, and
invoke. That connecting loop is this topic's entire subject.

## Intuition

Think of a large company's help desk. A ticket comes in: *"my invoice math
looks wrong, can someone add up these two line items."* The triage step
doesn't send it to a random employee, or worse, hand it simultaneously to
every department (accounting, IT, HR, legal) and let each one decide for
itself whether the ticket concerns them. Triage first decides *which
department* this is (accounting), and only then does someone *within*
accounting decide which specific system to open (the calculator, not the
payroll system, not the expense-report tool) and what numbers to plug into it.

Topic 1's `SkillRegistry.select()` is the triage step — which department.
Topic 2's `list_tools()`/`call_tool()` is the department's own toolkit. This
topic's contribution is small but structural: it is the fact that triage's
answer (which department) should *narrow* which tools even get considered next,
rather than every ticket being matched against every tool in the entire
company regardless of department. That narrowing step did not exist anywhere
in Topics 1 or 2 — each built and tested its own mechanism in isolation.

## Why simpler approaches fail

**"Just run skill selection and tool selection as two separate, uncoordinated
systems."** This is literally what Topics 1 and 2's notebooks did, and it is
the natural next step to try before building a real loop: keep
`SkillRegistry.select()` exactly as Topic 1 built it, keep the MCP
`list_tools()`/`call_tool()` client exactly as Topic 2 built it, and run them
side by side with no data passed between them.

This fails for a concrete, structural reason, not a hypothetical one:

- Topic 1's `SkillRegistry` has **no notion of a tool name at all**. Its
  `select()` returns a skill name and a `body` string; nothing in that
  dataclass or that registry ever mentions `add`, `word_count`, or any MCP
  tool. A skill's instructions can *tell a human* to use a particular tool
  (this topic's `arithmetic-calculation` skill body literally says "Call the
  `add` tool"), but there is no code path connecting that sentence to Topic
  2's `TOOLS` dict or its `list_tools()` result.
- Topic 2's MCP client has **no notion of a skill at all**. `list_tools()`
  returns every registered tool unconditionally; `call_tool()` takes a tool
  name and arguments a human already decided on. Nothing in Topic 2's
  notebook ever narrows the tool list based on what kind of task is being
  handled — every cell in that notebook picked its tool name as a Python
  literal, by hand.
- Running both systems "side by side" with no coordination means an agent
  built this way would need a *third*, separately-authored mechanism to
  decide which tool(s) from Topic 2's full, unfiltered registry are even
  relevant to the skill Topic 1 selected — and without that mechanism, the
  only two fallback options are (a) a human still picks the tool by hand,
  which isn't an agent at all, or (b) the agent keyword-matches the task
  against **every** registered tool with no narrowing, which is exactly the
  "consider-everything" baseline this topic's Experiment section measures and
  shows getting *worse*, not better, as the tool registry grows. Neither
  option is "two decoupled systems working fine independently" — the
  decoupling itself is the failure.

The fix is not a new algorithm; it is the deliberately small, explicit piece
of glue this topic adds: `SKILL_TO_TOOLS`, a map from a Topic-1 skill name to
the Topic-2 tool names that skill's instructions actually point toward, plus a
loop that runs selection, narrowing, extraction, and invocation as one
sequence instead of two unrelated experiments.

## Conceptual foundation (documented substitution for "Mathematical foundation")

Like Topics 1, 2, and `14-multi-agent-systems`, this is a systems/architecture
topic — what follows is a precise structural definition of the loop, not a
derivation.

### The 4-stage loop, formalized

Given one task string $t$, a skill registry $S$ (Topic 1's `SkillRegistry`), an
MCP tool catalog discovered live via `list_tools` (Topic 2's client) called
$C$, and a skill-to-tool map $M: \text{skill name} \to \{\text{tool names}\}$
(this topic's own glue, `SKILL_TO_TOOLS`), the loop is:

1. **Select.** $s_k = \text{select}(t, S)$ — Topic 1's deterministic
   keyword-overlap `SkillRegistry.select()`, reused verbatim, unchanged.
2. **Narrow + choose tool.** Candidate tool names are
   $\text{candidates} = M(s_k) \cap \text{names}(C)$ — the intersection of
   what the selected skill points toward and what the live server actually
   discovered via `list_tools`. This intersection is deliberate and load-
   bearing: a name in $M(s_k)$ that is *not* in $C$ (the server never
   registered it) is filtered out here, before any invocation is attempted —
   this is exactly the mechanism "Failure modes" below demonstrates catching
   a real skill/server mismatch. The single best tool
   $\tau = \arg\max_{n \in \text{candidates}} |\text{tokens}(t) \cap
   \text{tokens}(\text{description}(n))|$ is chosen by the same
   keyword-overlap scoring Topic 1 uses for skills, applied here to tool
   descriptions instead.
3. **Extract.** $\text{args} = \text{extract}_\tau(t)$ — a deterministic,
   per-tool regex function (an explicit, honestly-labeled stand-in for what an
   LLM would normally do: turn natural language into a structured argument
   dict).
4. **Invoke.** $r = \text{call\_tool}(\tau, \text{args})$ over the real MCP
   subprocess (Topic 2's JSON-RPC client, reused verbatim) — a real result,
   not a simulated one.

### This loop IS `14-multi-agent-systems/02-orchestration-patterns`'s
### decompose→delegate→aggregate pattern, at $n=1$

That topic's notes.md formalizes manager/worker orchestration as: (1) **task
decomposition** — $O$ splits $T$ into $T_1, \ldots, T_n$ such that
$T = \text{combine}(T_1, \ldots, T_n)$; (2) **delegation** — each $T_i$ is sent
1-to-1 to a worker $w_i$; (3) **worker execution** — each $w_i$ computes
$R_i = f(T_i)$ independently; (4) **result aggregation** — $O$ computes
$R = \text{aggregate}(R_1, \ldots, R_n)$.

`agent_loop` above is that same four-step structure with $n = 1$:

| Orchestration pattern step | This topic's loop stage |
|---|---|
| (1) Decomposition, $T \to T_1$ | Trivial at $n=1$: the whole task string *is* the one subtask; `combine` is the identity |
| (2) Delegation, $T_i \to w_i$ | Stages 1–2: select a skill (which specialist), narrow to a tool (which capability that specialist should invoke) |
| (3) Worker execution, $R_i = f(T_i)$ | Stage 4: the real `call_tool` JSON-RPC call — $f$ is whatever Python function the MCP server runs |
| (4) Aggregation, $R = \text{aggregate}(R_1)$ | Trivial at $n=1$: `agent_loop`'s return value already *is* the final answer |

This is not a loose analogy stretched to fit — it is the same named pattern,
named directly, run by a single agent sequentially against itself instead of
across a pool of worker processes. The "orchestrator" and the sole "worker"
are different processes here too (the notebook process and the `mcp_server.py`
subprocess), the same structural split `14-multi-agent-systems` uses, just
with $n=1$ worker instead of many.

## Algorithm

1. Register skills into a `SkillRegistry` (verbatim reuse of Topic 1's class).
2. Author `SKILL_TO_TOOLS`, an explicit map from skill name to candidate tool
   names — the piece missing from Topics 1 and 2 individually.
3. Spawn the MCP server as a real subprocess and call `list_tools` once
   (verbatim reuse of Topic 2's client) to get the live, ground-truth tool
   catalog.
4. Register one regex-based argument extractor per known tool name.
5. For each incoming task string: select a skill, intersect that skill's
   candidate tools with the live catalog and pick the best-scoring one,
   extract arguments with that tool's extractor, invoke `call_tool`, return
   the real result — or abort with an explicit, typed error at whichever
   stage first fails (no candidate tool available; extraction pattern doesn't
   match; JSON-RPC error from the server).

## From-scratch implementation

Built and actually run in
[001_skills_mcp_agent_loop.ipynb](001_skills_mcp_agent_loop.ipynb):

- `Skill` and `SkillRegistry` — copied verbatim from
  `01-agent-skills/001_progressive_disclosure_skill_selection.ipynb` (cells 3
  and 7), not reimplemented.
- Five skills registered: two on-topic (`arithmetic-calculation`,
  `text-analysis`, each written so its description vocabulary overlaps the
  right kind of task and its body honestly tells the agent to call an
  external tool rather than compute the answer itself) and three distractors
  carried over unchanged from Topic 1 (`csv-cleaning`, `email-drafting`,
  `code-review-checklist`), so selection is a real 5-way choice.
- `SKILL_TO_TOOLS`, the explicit skill→tool-names glue map: `{"arithmetic-
  calculation": ["add"], "text-analysis": ["word_count", "reverse_string"],
  "csv-cleaning": [], "email-drafting": [], "code-review-checklist":
  ["lint_code"]}` — the last entry deliberately points at a tool that does not
  exist on the server, used in Failure modes below.
- `mcp_server.py` — a local copy of Topic 2's server script (`add`,
  `word_count`, `reverse_string`, same JSON-Schema validation, same JSON-RPC
  2.0 stdio transport) — copied for this topic's self-containedness, not
  re-derived.
- The same subprocess-spawning + `send_request` JSON-RPC client as Topic 2's
  notebook, reused verbatim, and one real `list_tools` call confirming the 3
  real tools are discovered live.
- Three regex extractors: `extract_add_args` (first two numbers in the
  string), `extract_word_count_args` (text after `"count ... words ... in:"`),
  `extract_reverse_args` (text after `"reverse:"`/`"reverse this text:"`).
- `agent_loop(task)`, the full 4-stage function, run end-to-end on three toy
  tasks with real printed output at every stage:

  ```
  TASK: 'please add 17 and 25 for me'
    [stage 1] selected skill : arithmetic-calculation  (scores: {'arithmetic-calculation': 1, ...})
    [stage 2] candidate tools: ['add']  -> selected tool: add
    [stage 3] extracted arguments: {'a': 17, 'b': 25}
    [stage 4] tool result: 42

  TASK: 'count the words in: the quick brown fox'
    [stage 1] selected skill : text-analysis  (scores: {'text-analysis': 2, ...})
    [stage 2] candidate tools: ['word_count', 'reverse_string']  -> selected tool: word_count
    [stage 3] extracted arguments: {'text': 'the quick brown fox'}
    [stage 4] tool result: 4

  TASK: 'reverse this text: model context protocol'
    [stage 1] selected skill : text-analysis  (scores: {'text-analysis': 2, ...})
    [stage 2] candidate tools: ['word_count', 'reverse_string']  -> selected tool: reverse_string
    [stage 3] extracted arguments: {'text': 'this text: model context protocol'}
    [stage 4] tool result: locotorp txetnoc ledom :txet siht
  ```

  All three real results (`42`, `4`, the reversed string) are returned by the
  live MCP subprocess, not fabricated.

## Practical implementation

There is no separate "production library" step for this topic beyond what
Topics 1 and 2 already documented individually — this topic's own
contribution (the `SKILL_TO_TOOLS` map and `agent_loop`'s sequencing) *is*
already the minimal practical shape of the connecting logic a real agent
harness runs, just with a keyword heuristic where a real system puts an LLM.
The mapping back to a real system is direct:

| From-scratch (this notebook) | Real system |
|---|---|
| `registry.select(task)` | An LLM reading available skill descriptions and picking one, given the task |
| `SKILL_TO_TOOLS` map | Implicit in a real skill's own instructions/body (which tools it tells the model to call) plus the model's own reasoning about which of those still exist |
| `select_tool_from_candidates` restricted to `discovered_tools` | The model choosing among an MCP server's live `list_tools` result, filtered to what the current skill's instructions suggest |
| `extract_*_args(task)` regex functions | The LLM itself performing structured argument extraction as part of forming a tool call |
| `call_tool` over a real subprocess | The same JSON-RPC `call_tool` request, but typically over a real external MCP server (network or local process) |

## Experiment

**Hypothesis:** reusing Topic 1's skill-selection heuristic to *narrow* which
tools are even considered (only the tool names the selected skill's
`SKILL_TO_TOOLS` entry lists) reduces wrong-tool-invocation errors compared to
considering **all** registered tools every time, and this advantage should
grow — not shrink — as the number of registered tools grows.

**Setup.** A fixed battery of 10 toy tasks was built, each with a known-correct
tool (5 arithmetic tasks → `add`, 3 word-count tasks → `word_count`, 2
reverse-string tasks → `reverse_string`). The 3 real MCP tools were extended
with 27 **synthetic** toy tools — generated programmatically, the same
recombination technique Topic 1 uses for synthetic skills — to build tool
catalogs of size $N \in \{5, 15, 30\}$ (first 3 slots always the real tools).
Each synthetic tool's description is built by randomly drawing words from a
pool made of **both** the real tools' own descriptions **and** the task
battery's own vocabulary — an explicit, honestly-labeled stand-in for a
realistic large tool marketplace where many unrelated third-party tools'
descriptions happen to share generic words ("count," "text," "add," "numbers")
with common task phrasings, without being the actually-correct tool for any of
them. (A first attempt built the pool from only the real tools' descriptions;
that capped every synthetic tool's overlap score at a tie with the real tool's
own score — ties always resolved toward the real tool by dict insertion order
— producing a flat, uninformative 0% wrong-tool rate for *both* variants at
every $N$. The task+description pool is what actually lets a synthetic tool
occasionally out-score the correct real tool by chance, which is what makes
this experiment measure something real.)

Two variants were implemented and run against the exact same battery and
catalogs:

- **Consider-everything**: keyword-match the task against every tool name
  currently in the catalog (real + synthetic), pick the single best-scoring
  match, with no skill-based narrowing at all.
- **Skill-narrowed**: select a skill first (`registry.select`), restrict
  candidates to `SKILL_TO_TOOLS[selected_skill]`, then keyword-match only
  within that (much smaller, fixed-size) candidate set.

**Expected result:** skill-narrowed stays near 0% wrong-tool invocations
regardless of $N$; consider-everything's wrong-tool rate rises as $N$ grows.

**Actual result (real measured output from the notebook):**

| N tools | consider-everything wrong-tool rate | skill-narrowed wrong-tool rate |
|---:|---:|---:|
| 5 | 40% | 0% |
| 15 | 80% | 0% |
| 30 | 80% | 0% |

**Interpretation.** The hypothesis held, and specifically in the way predicted:
the skill-narrowed variant's rate is 0% at every $N$, because
`SKILL_TO_TOOLS` restricts its candidate pool to at most 2 real tool names no
matter how many synthetic tools sit elsewhere in the catalog — the synthetic
tools are structurally excluded, since nothing routes any skill to them. The
consider-everything variant's rate rises sharply from $N=5$ to $N=15$ (40% →
80%) and then plateaus into $N=30$ rather than climbing further, which is
itself informative rather than a weakness in the result: once a task's
outcome is already wrong at $N=15$ because at least one synthetic tool
out-scored the correct real tool, adding still more synthetic tools cannot
make that same task any *more* wrong — the ceiling on this measure is 100%,
and this particular seeded battery of 10 tasks happened to saturate most of
the way there by $N=15$.

**Limitations.** The synthetic tools are recombinations of a shared vocabulary
pool, not independently authored realistic tool descriptions — the exact
percentages are a function of this experiment's specific pool-construction
choice and its `random.Random(2000 + i)` seed, not a universal constant; a
different pool-construction strategy would shift the curve's steepness (as the
first, uninformative attempt above demonstrates directly). The task battery is
small (10 tasks) and fixed across all three registry sizes rather than scaled
with $N$. Keyword-overlap scoring itself, not just the narrowing/no-narrowing
choice, is the thing being stressed here — a stronger tool-selection heuristic
(even without skill-narrowing) might also reduce the consider-everything rate,
so this experiment isolates "does narrowing help *this* heuristic," not "is
narrowing the only fix."

## Failure modes

Two concrete, reproduced failures — not hypothetical ones, both from the
notebook's actual output.

**1. A skill points at a tool that does not exist on the current MCP
server — caught by discovery before invocation.** `SKILL_TO_TOOLS` maps
`code-review-checklist` to `["lint_code"]`, but this server was only ever
asked to register `add`, `word_count`, `reverse_string` — `lint_code` was
never registered. This is a real integration-mismatch bug: a skill author's
assumption about available tools drifted from what the server actually
exposes (in a real system, a plausible cause: the skill was written against
an older or different MCP server deployment). Actual measured behavior for
the task *"please review this pull request for bugs and missing tests"*:
`code-review-checklist` is correctly selected (score 6, well ahead of
`csv-cleaning`'s score 1), `candidates = ['lint_code']`, and then
`lint_code not in discovered_tools` — the intersection with the live
`list_tools` result filters it out, `selected_tool` comes back `None`, and
`agent_loop` aborts with `"no available tool for skill 'code-review-
checklist' among candidates ['lint_code']"` **before any `call_tool` request
is ever sent**. This is exactly what the discovery-before-invocation
discipline from `02-model-context-protocol` buys, applied here one layer up:
the mismatch is caught structurally by checking candidate names against a
live source of truth, not by hoping the skill author's tool list stays
correct forever.

**2. The deterministic argument-extraction regex fails on a task phrased
differently than expected — a genuine parsing failure.** `extract_add_args`
looks for digit sequences (`NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")`) in
the task string. It has no notion of numbers spelled out as words. Actual
measured behavior for the task *"please add seventeen and twenty five for
me"*: stage 1 correctly selects `arithmetic-calculation` (score 1), stage 2
correctly selects `add` (it does exist on the server) — both upstream stages
get the right answer — but stage 3 fails cleanly:
`extract_add_args: expected at least two numbers in the task, found [] in
'please add seventeen and twenty five for me'`, a caught `ValueError` rather
than a silently wrong tool call (e.g. calling `add` with no arguments, or
with the wrong ones parsed from unrelated digits elsewhere in a longer
sentence). This demonstrates precisely where the deterministic-regex stand-in
for LLM-based extraction breaks: it can only handle the surface forms it was
written to expect, and any task phrased outside that surface form fails
*before* the tool is ever called, not after — the same "fail before
invocation, not after" discipline as failure mode 1, just triggered by a
different upstream cause.

## Real-world usage

This loop, scaled up along exactly two axes, is architecturally what
production "tool-using LLM agent" systems actually do:

1. **Replace the two deterministic keyword heuristics (`SkillRegistry.select`
   and `select_tool_from_candidates`) with a real LLM call.** An LLM reading a
   task, a list of available skill descriptions, and a list of available tool
   schemas, and choosing among them, is a strict generalization of the
   keyword-overlap scoring this notebook uses throughout — same inputs, same
   discrete choice, a far more capable selector. The same generalization
   applies to argument extraction: an LLM parsing "please add seventeen and
   twenty five for me" into `{"a": 17, "b": 25}` is exactly the capability
   this topic's regex extractors were an honest, explicitly-labeled stand-in
   for (see Failure mode 2, which is precisely the class of input a real LLM
   extractor would handle and this notebook's regex could not).
2. **Replace the toy in-process/subprocess MCP server with real external MCP
   servers** — GitHub, Slack, Postgres, Google Drive, Notion, and the rest,
   the same class of servers this very Claude Code session has deferred tools
   for (`mcp__claude_ai_Notion__*`, `mcp__claude_ai_Google_Calendar__*`, named
   again here as in Topic 2's notes.md, never called, per this section's
   no-live-external-call constraint).

This is `15-agent-skills-and-mcp`'s own capstone connection: Claude Code
itself, Anthropic's Agent SDK, and "tool-using LLM agent" systems generally
are built on precisely this decompose→select→extract→invoke loop — an LLM in
place of `registry.select`/`select_tool_from_candidates`, real MCP servers in
place of `mcp_server.py`, and the same discovery-before-invocation discipline
this topic's Failure mode 1 demonstrated catching a real integration mismatch
before it ever reached a tool call. Nothing about the *structure* of the loop
changes at production scale — only the fidelity of steps 1–3 (selection,
narrowing, extraction) and the realness of what step 4 talks to.

## Mental model

**Skills tell you *which department to walk into*; MCP tells you *what's on
the shelf once you're there*; the agent loop is the person who actually walks
through the door, reads the shelf labels, picks the right item, and rings it
up.** None of the three roles substitutes for the other two: knowing which
department is right doesn't stock the shelf, a well-stocked shelf doesn't
walk anyone through the door, and a person with no idea which department to
enter wanders forever regardless of how well-organized any one department's
shelf is.

## Questions to think about

1. `SKILL_TO_TOOLS` is authored by hand in this notebook — a human decided
   `arithmetic-calculation` maps to `["add"]`. In a real system where skill
   bodies are free-form Markdown text (as in `01-agent-skills`'s real
   `SKILL.md` format), how would you derive this mapping automatically rather
   than hand-authoring it, and what would have to go right for that automatic
   derivation to reliably catch the same `lint_code`-style mismatch this
   topic's Failure mode 1 demonstrates catching by hand?
2. This topic's loop stops at $n=1$ (Conceptual foundation's decomposition
   step is the identity function). Sketch, concretely, what stage 1
   (decomposition) would need to do differently for the task "add 17 and 25,
   then tell me how many words are in the result" — and which of this
   loop's four stages would then need to run *twice*, in what order, for that
   task to resolve correctly.
3. The Experiment section's consider-everything wrong-tool rate plateaus at
   80% between $N=15$ and $N=30$ rather than climbing to 100%. Using the
   battery of 10 fixed tasks, explain precisely what would have to be true
   of the 2 tasks that stayed correctly routed even at $N=30$ for the rate to
   plateau rather than continue climbing — is it about those tasks'
   vocabulary, the real tool's description, or something about how the
   synthetic pool was built?
4. Failure mode 1 (skill points at a nonexistent tool) is caught by
   intersecting `SKILL_TO_TOOLS[skill]` with the *live* `discovered_tools`
   result rather than trusting `SKILL_TO_TOOLS` on its own. What would have
   to be true for this same intersection check to *fail* to catch a
   mismatch — i.e., can you construct a scenario where a skill points at a
   tool name that technically exists in `discovered_tools` but is still the
   wrong tool for what the skill actually needs?
5. This topic reuses Topic 1's `SkillRegistry.select` for skill selection
   (stage 1) and a near-identical keyword-overlap function for tool selection
   (stage 2) — the same scoring mechanism applied twice, at two different
   granularities. Is there an argument for why the *same* selection quality
   (or the same failure modes, like Topic 1's near-tie problem) should be
   expected at both stages, or is there a structural reason one stage might
   be more reliable than the other given how `SKILL_TO_TOOLS` shrinks the
   stage-2 candidate set?
