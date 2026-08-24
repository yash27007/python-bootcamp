# 01 — Agent Communication Protocols

## Problem

A single agent — one process, one context window, one line of reasoning at a time — can
only do one thing at once. Real tasks often decompose naturally into pieces that are best
handled by *different* specialists working from *different* state: a "buyer" that knows a
budget and a "seller" that knows a cost structure are not the same reasoning process, and
forcing one agent to simulate both roles internally throws away the reason to split them
in the first place (separate goals, separate information, sometimes separate owners or
separate machines). Once a task is split across more than one agent, those agents need a
way to actually exchange information with each other — that exchange mechanism, not the
agents themselves, is what this topic is about.

## Intuition

Picture three people trying to buy and sell a used graphics card over text messages
instead of a phone call. One buyer texts "does anyone have a GPU for sale?" to a group
chat. Two sellers each reply privately with a price. The buyer picks the cheaper one and
texts back "deal" to the winner and "no thanks" to the other. Nobody ran anybody else's
brain — each person decided what to say based only on their own goals and the messages
they personally received. That's the whole shape of an agent communication protocol:
**who can send what to whom, and what shape does each message have.**

Concretely, in this topic's actual toy negotiation (transcript captured for real in the
notebook):

```
[00]    buyer -> ALL      request  {'item': 'gpu'}
[01] seller_a -> buyer    propose  {'item': 'gpu', 'price': 72}
[02] seller_b -> buyer    propose  {'item': 'gpu', 'price': 65}
[03]    buyer -> seller_a reject   {'item': 'gpu'}
[04]    buyer -> seller_b accept   {'item': 'gpu'}
```

Five messages, three independent agents, zero shared internal state, one price war
resolved. Nobody's code ever calls another agent's function directly — every interaction
happened as a message with an explicit sender, receiver, and intent.

## Why simpler approaches fail

It is tempting to write:

```python
buyer_budget = 100
price_a = seller_a_quote(item="gpu")
price_b = seller_b_quote(item="gpu")
winner = "a" if price_a < price_b else "b"
```

This runs, and for a two-seller toy example it even produces the same *answer* as the
message-passing version above. But it is not multi-agent communication — it's one process
calling two functions in sequence. Three things silently disappear when you do this:

1. **No separation of state.** `seller_a_quote()` is a plain function; it has no
   persistent internal state of its own (a cost basis, a memory of past deals, a goal it's
   optimizing). The moment a seller needs to remember something across calls, or decide
   *whether* to answer at all, "function call" stops being an adequate model and "agent
   with its own state" becomes necessary.
2. **No enforced interface.** Direct function calls let the caller reach into the callee's
   internals in whatever shape is convenient *right now*. A message format is a contract:
   every participant agrees in advance on `(sender, receiver, performative, content)`, so a
   new seller can be added later without the buyer's code changing at all — it just needs
   to speak the same message shape. Section "Algorithm" below shows this holds for the
   toy implementation: `BuyerAgent.accept_best()` never once imports `SellerAgent`.
3. **No explicit synchronization model.** "Call function A, then call function B" *is* a
   synchronization decision (strict sequential order), but it's an invisible one, baked
   into control flow instead of stated as a protocol. Real agents may run as separate
   processes or separate machines; whether they talk synchronously (block for a reply) or
   asynchronously (fire a message and continue), and what happens if a reply never comes,
   has to be a **deliberate design choice**, not an accident of how the code happens to be
   laid out. This topic's `MessageBus` is intentionally synchronous and single-threaded —
   see "Failure modes" for exactly what that assumption is hiding.

The point of building a message/bus abstraction at all — even at toy scale, even with
scripted (non-LLM) agents — is to make these three things explicit instead of implicit.

## Conceptual foundation

*(This is a systems/architecture topic without calculus-style mathematics behind it, per
AGENTS.md's documented substitution — the same category as `08-mlops-deployment/01-docker`
and `08-mlops-deployment/02-git`. The real conceptual depth here is message structure and
communication topology, not an equation.)*

### The message as a formal object

Every message in this topic (and in the real standard it descends from) is modeled as a
4-tuple:

$$
m = (\text{sender}, \text{receiver}, \text{performative}, \text{content})
$$

- **sender** — the agent that produced the message.
- **receiver** — the intended addressee, or `None` for a message with no single addressee
  (broadcast or blackboard post).
- **performative** — the *communicative intent* of the message: not what the message
  contains, but what speech-act it is performing. Asking for something is a different
  performative than informing someone of a fact, even if the payload data looks similar.
- **content** — the actual payload (here, a plain dict — in a real system, structured data
  or natural language).

This 4-tuple is a direct simplification of **FIPA-ACL** (the Foundation for Intelligent
Physical Agents' Agent Communication Language), a real, standardized message format from
multi-agent systems research that defines a fixed vocabulary of **performatives** —
standard communicative-intent labels every compliant agent is supposed to understand the
same way. This topic borrows four of the most common ones by name, used with their
standard meaning:

- **`inform`** — tell the receiver a fact ("the price is 65").
- **`request`** — ask the receiver to do something or provide something ("does anyone have
  a GPU?").
- **`propose`** — offer specific terms in response to a request ("I'll sell it for 65").
- **`accept` / `reject`** — respond to a proposal with a binding decision.

The value of a fixed performative vocabulary is exactly the value of a fixed HTTP verb
vocabulary (`GET`/`POST`/`PUT`/`DELETE`): a receiver can dispatch on *intent* without
having to parse or guess what the sender meant from the payload's shape alone. This
topic's from-scratch `SellerAgent.receive()` does precisely this — it branches on
`msg.performative`, not on inspecting `msg.content`.

### Three communication topologies

The same 4-tuple message can be delivered under (at least) three different topologies —
*who receives a message once it's sent*. This topic implements all three.

**1. Direct point-to-point (request-response).** The sender addresses exactly one
receiver; the bus (or network) delivers to that one agent only. This is the most
information-preserving topology (nobody uninvolved sees the message) but requires the
sender to already know *who* to talk to, and its cost grows with the number of
one-to-one conversations needed — sending the same request to $n$ different receivers
takes $n$ separate direct messages.

**2. Broadcast.** The sender addresses no one in particular; the message is delivered to
*everyone* currently registered. This trades away addressing precision (every listener
sees every broadcast message, whether it's relevant to them or not) for a single
send-time operation regardless of how many listeners there are. The tradeoff is explored
concretely in "Experiment" below — broadcast collapses the *sender's* cost from $n$
messages to $1$, but does not reduce the total number of deliveries.

**3. Blackboard.** The sender writes a message to a shared, passively-readable store; it
is not delivered (pushed) to anyone. Other agents choose, on their own schedule, to poll
the store and read what's relevant to them. This is the loosest coupling of the three —
the sender doesn't even need to know how many readers exist, or whether they exist yet —
at the cost of losing any delivery guarantee or timing guarantee at all: a reader that
never polls never sees the message.

| Topology | Sender must know receiver? | Cost to reach n listeners | Delivery guarantee |
|---|---|---|---|
| Direct | yes | n messages | yes (or an explicit failure) |
| Broadcast | no | 1 send, n deliveries | yes, to everyone currently registered |
| Blackboard | no | 1 write | no — depends on readers polling |

## Algorithm

The toy negotiation this topic implements, end to end:

1. **Setup.** A `BuyerAgent` and two `SellerAgent`s register with a shared `MessageBus`.
2. **Request (broadcast).** The buyer constructs one `Message(sender="buyer",
   receiver=None, performative="request", content={"item": "gpu"})` and calls
   `bus.broadcast(msg)`. Every other registered agent's `receive()` fires.
3. **Propose (direct, x2).** Each seller's `receive()` sees `performative == "request"`,
   checks whether it stocks that item, and if so replies with a direct `send()` back to
   `msg.sender` — a `propose` message carrying its price.
4. **Collect.** The buyer's own `receive()` appends every incoming `propose` to
   `self.offers`.
5. **Decide.** `accept_best()` picks the offer with the minimum price
   (`min(self.offers, key=lambda m: m.content["price"])`), then sends one direct `accept`
   to the winner and one direct `reject` to every other offerer.
6. **Terminate.** Each seller's `receive()` updates its own `self.won` flag based on
   whether it got `accept` or `reject`.

No agent ever calls another agent's methods directly, inspects another agent's internal
attributes, or shares mutable state — every cross-agent effect happens through a `Message`
passed through the bus. That is the property the "Why simpler approaches fail" section
argued a plain sequential function-call chain does not have.

## From-scratch implementation

Implemented in `001_message_bus_negotiation.ipynb`, actually executed, real output:

- **`Message`** — a `@dataclass` with `sender`, `receiver` (`Optional[str]`),
  `performative`, `content`, and an auto-incrementing `msg_id` for transcript ordering.
- **`MessageBus`** — plain Python, no network, no threads:
  - `send(msg)` — direct delivery; raises `KeyError` on an unknown receiver.
  - `broadcast(msg, exclude_sender=True)` — fans out to every other registered agent.
  - `publish(msg)` / `read_blackboard(performative=None)` — blackboard write/read, with no
    push at all.
  - `self.log` — every message that ever passed through the bus, in send order, used for
    the transcript and for the message-count experiment below.
- **`BuyerAgent`, `SellerAgent`** — the scripted negotiation logic described in
  "Algorithm." Both are deterministic: given the same inputs, they always produce the same
  outgoing messages. There is no LLM call, sampling, or randomness anywhere in either
  class.
- **`BlackboardBuyer`, `BlackboardSeller`** — a second pair of agents demonstrating the
  third topology on the same scenario: the buyer `publish`es its request instead of
  broadcasting it, and each seller calls `check_and_quote()` to actively poll the
  blackboard for outstanding requests, rather than being pushed a message.

The real, captured transcript from running the buyer/2-seller negotiation:

```
[00]    buyer -> ALL      request  {'item': 'gpu'}
[01] seller_a -> buyer    propose  {'item': 'gpu', 'price': 72}
[02] seller_b -> buyer    propose  {'item': 'gpu', 'price': 65}
[03]    buyer -> seller_a reject   {'item': 'gpu'}
[04]    buyer -> seller_b accept   {'item': 'gpu'}

winner: seller_b at price {'item': 'gpu', 'price': 65}
seller_a.won = False  seller_b.won = True
```

The buyer correctly identified and accepted the cheaper of the two real proposals it
received, purely by exchanging messages — it never read `seller_a.price_for_item` or
`seller_b.price_for_item` directly.

## Practical implementation

This topic deliberately has **no separate "practical/library" implementation step** the
way, say, `08-mlops-deployment/01-docker`'s from-scratch cache-key simulator is followed by
real `docker build`. There is no lightweight, offline, no-API-key multi-agent-messaging
library that would add insight beyond what the from-scratch `MessageBus` already shows at
this scale — the *real* practical frameworks in this space (AutoGen, CrewAI, LangGraph)
are all designed around orchestrating actual LLM API calls, which this section is
explicitly barred from making (see the top of notes.md and "Real-world usage" below for
why, and for how those frameworks' message-passing designs map back to this notebook's
`Message`/`MessageBus` abstractions without running them here).

## Experiment

**Hypothesis.** For the same one-buyer/$n$-seller negotiation, the **direct** topology
should cost the sender $3n$ logged bus operations ($n$ requests + $n$ proposals + $n$
accept/reject replies), while **broadcast** should cost only $2n+1$ ($1$ broadcast + $n$
proposals + $n$ accept/reject replies) — because broadcast collapses the $n$ individual
request-sends into a single bus call. The gap should grow linearly with $n$.

**Setup.** `run_direct_topology(n)` and `run_broadcast_topology(n)` each build a fresh
`MessageBus`, one `BuyerAgent`, and `n` `SellerAgent`s with distinct prices, run the full
negotiation to completion, and return `len(bus.log)`. Tested at $n \in \{3, 6, 10\}$.

**Actual measured result:**

| n (sellers) | direct topology log entries | broadcast topology log entries | predicted $3n$ | predicted $2n+1$ |
|---:|---:|---:|---:|---:|
| 3  | 9  | 7  | 9  | 7  |
| 6  | 18 | 13 | 18 | 13 |
| 10 | 30 | 21 | 30 | 21 |

Both formulas matched exactly for every tested $n$ (asserted in the notebook, not just
eyeballed).

**Interpretation.** Broadcast strictly reduces the number of bus operations the *sender*
has to construct and address as $n$ grows (the gap is $n - 1$ messages saved, e.g. 9 fewer
at $n=10$) — that is a real, measured, linearly-growing advantage. But actual message
**deliveries** — how many times some agent's `receive()` fires — are $3n$ under *both*
topologies; broadcast doesn't reduce total network traffic, it only reduces how many times
the sender had to explicitly name a recipient. This distinction is exactly what "Failure
modes" below relies on for the broadcast-storm discussion: a broadcast is cheap to *send*
and identically expensive to *receive*.

**Limitations.** This measures message count only — not latency, bandwidth, or the cost of
each agent actually processing a message. It assumes every seller always replies (no
seller ever silently declines), a fully synchronous single-threaded bus with no message
loss, and a fixed negotiation shape (one request round, one accept/reject round). None of
those assumptions survive contact with a real distributed multi-agent deployment — see
"Failure modes" for what breaks first.

## Failure modes

**1. Message ordering / race conditions.** This implementation is fully synchronous:
`bus.send()` and `bus.broadcast()` call the receiving agent's `receive()` immediately and
block until it returns, so message order is exactly the order `send`/`broadcast`/`publish`
were called in, with no possibility of interleaving. A real deployment running agents as
separate processes or over a network has no such guarantee — two sellers might both reply
to a broadcast `request` at nearly the same wall-clock time, and the order the buyer's
process actually observes them in depends on network latency and scheduling, not on any
property of the negotiation itself. If `accept_best()` were written to accept the *first*
offer received instead of the cheapest, a race in delivery order — not price — would
decide the winner. Concurrency doesn't just make code slower or trickier to write here; it
can silently change *which agent wins*.

**2. Protocol mismatch — a performative the receiver doesn't understand.** Demonstrated
concretely in the notebook: `SellerAgent.receive()` only branches on `request`, `accept`,
and `reject`. Sending it a `cancel` message (a performative it was never coded to handle)
does not raise an exception, does not log a warning, and does not change `seller.won` —
the message is delivered into the seller's `inbox` list and then simply has no effect. In
a two-line toy example this is easy to spot by reading the code; in a system with many
agents built by different people (or generated at different times) an unhandled
performative is a **silent** failure — nothing signals that the sender's intent was never
acted on. A more robust protocol would require every agent to reply with something like
`Message(performative="not-understood", ...)` whenever it receives a performative outside
its known vocabulary, so the sender at least learns the message was dropped.

**3. Broadcast storms.** The "Experiment" section measured that broadcast and direct
topologies deliver the *same* total number of messages ($3n$) even though broadcast is
cheaper to *send*. As agent count grows, that $3n$ delivery count is what actually matters
for the receivers: every agent registered on the bus receives (and must at least inspect)
every broadcast message, whether it is relevant to that agent's role or not. At $n=10$
sellers, ten `receive()` calls fire off a single `broadcast()` invocation; at $n=1000$,
one broadcast fires a thousand `receive()` calls. This is the concrete mechanism behind
"broadcast storms" in real distributed multi-agent systems — a topology that looks cheap
from the sender's local view (`O(1)` sends) is actually `O(n)` total work distributed
across the network, and it is easy to under-provision for that because the sender-side
code doesn't show it.

## Real-world usage

- **AutoGen** (Microsoft) models multi-agent interaction as *conversable agents* that
  exchange chat-style messages through a controllable conversation loop — structurally the
  same `(sender, receiver-or-group, content)` shape this topic's `Message` formalizes,
  except AutoGen's `content` is typically full natural-language text destined for a real
  LLM call, and its "receiver" is often a whole group-chat topology closer to this topic's
  broadcast than to direct point-to-point.
- **CrewAI** frames multi-agent coordination around explicit *task delegation*: a manager
  or orchestrator agent assigns discrete tasks to specialist agents and collects their
  results — closer to this topic's direct request-response topology (the buyer's
  `request` → sellers' `propose`) than to broadcast or blackboard, with the task itself
  standing in for this topic's `content` payload.
- **MCP (Model Context Protocol)** — the not-yet-built `15-agent-skills-and-mcp` section
  covers this in depth — standardizes a *request/response* pattern between a model-driven
  client and a tool-providing server, conceptually similar to this topic's direct `send()`
  topology (one addressed request, one addressed response) but specified as a real
  JSON-RPC-based network protocol rather than an in-process Python call.

None of these three real systems are called, imported, or executed anywhere in this
section — they are named here only to connect this topic's toy abstractions to real,
production message-passing designs by name, as AGENTS.md's "practical implementation"
step normally requires; the substitution taken here (naming the real systems instead of
running them) is the same one documented under "Practical implementation" above.

## Mental model

An agent communication protocol is not "calling a function on another object" — it's
**mail with a fixed envelope format and no guaranteed reader**. The envelope is
`(sender, receiver, performative, content)`; whether a message actually gets read,
understood, and acted on correctly is entirely the receiver's own responsibility, and the
sender has no way to know that failed unless the protocol *also* defines an explicit
"I didn't understand you" reply. Three topologies — direct, broadcast, blackboard — are
three different answers to one question: *does the sender need to know who's listening
before it speaks?*

## Questions to think about

1. In the direct-vs-broadcast experiment, broadcast reduced *logged bus operations* from
   $3n$ to $2n+1$ but not *deliveries* (both stayed $3n$). Design a fourth topology (or a
   variant of blackboard) where the number of actual deliveries could be reduced below
   $3n$ for the same negotiation. What has to be true about the receivers for that to be
   safe?
2. The blackboard topology in this notebook requires each `BlackboardSeller` to explicitly
   call `check_and_quote()` — nothing runs automatically. What would have to change about
   the `MessageBus` (not the agents) to make blackboard reads "close enough" to real-time,
   and what does that change cost in terms of coupling or complexity compared to plain
   broadcast?
3. `SellerAgent.receive()` silently ignores an unrecognized performative like `cancel`.
   Sketch, in words or pseudocode, the minimal protocol addition (a new performative? a
   required reply? a schema check at the bus level?) that would turn that from a silent
   failure into a *visible* one — and consider whether making it visible could itself
   create a new failure mode (e.g., what happens if *every* agent starts replying
   `not-understood` to every message it's confused by, including replies to
   `not-understood` messages)?
4. This topic's `MessageBus` is synchronous and single-process. If two `SellerAgent`s ran
   as genuinely separate OS processes reading from a real queue, what specific line of
   `BuyerAgent.accept_best()` would need to change to stay correct, and why does the
   in-process version get away without that change today?
5. AutoGen's conversable agents and CrewAI's task delegation were both named as running
   over real LLM calls in production, where "content" is natural-language text instead of
   a structured dict. What could go wrong with dispatching on `performative` (this topic's
   fixed, closed vocabulary) if `content` were instead a free-form LLM-generated string
   that had to be *parsed* to recover the sender's true intent?
