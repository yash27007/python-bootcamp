# Streamlit

## Problem

A model or analysis is done. Someone who isn't comfortable running a notebook or reading raw
`.predict()` output — a stakeholder, a teammate on a different team, a non-technical reviewer —
still needs to interact with it: adjust an input, pick a file, see a chart update. Building that
interactivity the traditional way means a browser frontend (HTML/CSS/JS, a build toolchain, a
client-server API to wire the two together) — a substantial second project on top of the analysis
itself. This topic answers: **how do you give a piece of Python analysis an interactive UI without
writing any of that frontend layer by hand?**

## Intuition

`13-streamlit/main.py` and `widgets.py` (already in this folder) are plain Python scripts —
`st.title(...)`, `st.write(...)`, `st.slider(...)` — with no HTML, no JavaScript, no separate
frontend build step anywhere. Running `streamlit run main.py` turns that script into a live web page
a browser can open. The trick that makes this possible, and that governs everything else about how
Streamlit apps behave, is Streamlit's execution model: **every time a user interacts with any
widget on the page (moves a slider, types in a text box, clicks a button), Streamlit reruns the
*entire* Python script from top to bottom**, with the new widget value plugged in, and redraws
whatever the script produces. There is no partial re-render, no separate "update this one element"
code path to write — the mental model is "the whole script is a function that gets called again on
every interaction," which is a genuinely different way of thinking about a UI than a traditional
event-handler-per-widget frontend.

## Why simpler approaches fail

**"Just print results to the terminal / a notebook cell."** Works for the person who wrote the
analysis, in the environment they wrote it in. It gives nobody else a way to change an input and see
the result update, and it isn't shareable as a link the way a running web app is.

**"Just write a real HTML/CSS/JS frontend, or a Flask app with templates."** This is the *general*
solution (`12-flask`'s topic, in fact) — and it is strictly more powerful and more customizable. But
it requires writing and maintaining frontend code as a genuinely separate concern from the Python
analysis: HTML markup, CSS layout, JavaScript event handlers wired to fetch calls against a backend
API. For the specific, common case of "I have a Python script that computes something and I want a
person to interactively explore it," that entire layer is exactly what Streamlit exists to remove —
see From-scratch below for why this topic doesn't build that layer by hand.

## Conceptual foundation

*(Substituting for "Mathematical foundation" — this topic's foundation is a UI execution model, not
a numeric derivation.)*

**Rerun-the-whole-script.** Every widget function (`st.slider`, `st.text_input`, `st.selectbox`,
...) returns the widget's *current* value on every single script run, not just when it changes.
Reading `widgets.py`: `age = st.slider("Select your age: ", 0, 100, 25)` doesn't register an
event-handler callback the way a traditional GUI framework would — it simply returns whatever the
slider's position currently is, every time the script executes top to bottom. Moving the slider
triggers Streamlit to rerun the script; this time `st.slider(...)` returns the new value, and every
line after it (including `st.write(f"Your age is {age}")`) recomputes using that new value. This is
the single idea that explains both why Streamlit code looks so simple (linear, top-to-bottom, no
callback wiring) and why it has the specific performance/state failure modes covered below.

**Widgets as declarative, not imperative.** In a traditional frontend, you construct a widget once
and later mutate it or attach handlers to it. In Streamlit, `st.slider(...)` is called fresh on
every rerun and simply *declares* "there is a slider here, with this current value" — there's no
persistent widget object your code holds onto across runs, which is why anything that needs to
persist *across* reruns (a running counter, an accumulated list) needs `st.session_state`
explicitly (see Failure modes) rather than a plain Python variable, which gets reset to its initial
value on every rerun since the whole script — including variable initializations — reruns from
scratch.

## Algorithm

The Streamlit execution model, as concrete steps:

1. `streamlit run app.py` starts a local web server and runs `app.py` top to bottom once, rendering
   whatever `st.*` calls produced, in order, on the page.
2. The user interacts with a widget (moves a slider, types text, clicks a button, uploads a file).
3. Streamlit reruns `app.py` from the top, this time with that widget's function call returning the
   new value (and every other widget's function call returning its own current, possibly-unchanged
   value).
4. Every line of the script re-executes in order — including any computation, any `st.write`/
   `st.line_chart` calls, any `pd.DataFrame` construction — and the page is redrawn to match.
5. Repeat from step 2 for every subsequent interaction, for the lifetime of the browser session.

## From-scratch implementation

**N/A / minimal — a deliberate judgment call, documented here rather than skipped silently.**
Every other topic in this section builds the underlying mechanism by hand before showing the
framework that automates it (`12-flask`'s raw `http.server` router before `@app.route`,
`11-memory-management`'s manual refcount/cycle demo before `gc`). Streamlit doesn't have an
equivalent "mechanism underneath the framework" worth hand-building for this section's purpose: the
genuine from-scratch equivalent of a Streamlit app is a raw HTML page with CSS layout and JavaScript
`addEventListener` calls wired to `fetch()` requests against a backend — real frontend engineering,
not a Python concept this section is trying to teach. Building a minimal HTML/JS form here would
not illuminate anything about *Streamlit's specific idea* (rerun-the-whole-script) the way, say,
`raw_http_server.py`'s hand-rolled router directly illuminates what `@app.route` automates — it
would just be a different, unrelated technology. **Streamlit itself is the practical layer for this
topic**; the "from-scratch" step this section's template asks for is better spent, for this one
topic, explaining *why* it's skipped (this section) than building an unrelated JS demo whose only
lesson would be "frontend code exists and it's harder to write than `st.slider(...)`," which the
Problem/Why-simpler-approaches-fail sections above already establish directly.

## Practical implementation

The existing Streamlit apps in this folder — described, since Streamlit apps aren't something that
executes meaningfully inside a notebook-execution or `nbconvert`-style check (they need a live
browser session and `streamlit run`, not a one-shot script run); what `streamlit run` would show is
described here rather than screenshotted or faked:

- [`main.py`](main.py) — `st.title`, `st.write` on plain text, `st.write` on a `pd.DataFrame`
  (renders as an interactive, sortable table), and `st.line_chart` on a random 15x3 DataFrame
  (renders as a live line chart with a legend for columns `a`/`b`/`c`). Running
  `streamlit run main.py` would open a browser tab showing a page titled "Yashwanth," a paragraph of
  text, a rendered data table, and a line chart below it — no interaction needed, since none of
  these calls read a widget value.
- [`widgets.py`](widgets.py) — interactive input widgets: `st.text_input` (a name field; the page
  conditionally shows `f"Hello {name}"` once something is typed — a direct example of "rerun the
  script, `name` now has the typed value, the `if name:` branch now executes"), `st.slider` (age,
  0-100, default 25), `st.selectbox` (a language dropdown), and `st.file_uploader` (accepts a CSV,
  and if one is uploaded, reads it with `pd.read_csv` and displays it via `st.write`). Running
  `streamlit run widgets.py` and typing a name would visibly demonstrate the rerun model: the page
  redraws with the greeting line present, without a page reload, the instant typing stops (or on
  each keystroke, depending on Streamlit's input-debouncing behavior for that widget type) — because
  the *entire script* reran with `st.text_input(...)` now returning the typed string. The same file
  also writes [`sample-data.csv`](sample-data.csv) to disk on every run (`df.to_csv(...)`) — worth
  noting as a Failure-modes-adjacent detail: this happens on *every single rerun*, including reruns
  triggered by moving the unrelated age slider, since the whole script reruns regardless of which
  widget changed (see Failure modes).

## Experiment

A live-browser `st.session_state` performance/rerun experiment cannot be executed headlessly in
this environment the way a notebook cell can (no browser session, no widget interaction to script
against `nbconvert`-style) — this is stated honestly rather than fabricating browser interaction
output. In place of a runnable experiment, the concrete, checkable prediction from the Conceptual
foundation section above is: **loading `widgets.py` and moving only the age slider would still
re-execute the `df.to_csv("sample-data.csv")` line**, even though that line has nothing to do with
age — because Streamlit has no way to know a given line's output doesn't depend on the widget that
changed; it only knows to rerun everything. This is directly checkable by inspecting the file's
modification time before and after moving the slider in a live session, without needing to fabricate
any Streamlit-rendered output.

## Failure modes

- **Rerunning the whole script on every interaction causes real performance problems.** Any
  expensive operation placed directly in the script body (loading a large file, training a model,
  querying a slow database) reruns on *every single* interaction, even ones unrelated to that
  operation — moving a slider that has nothing to do with the loaded file still re-triggers the
  file load, exactly as `widgets.py`'s `df.to_csv(...)` call demonstrated above re-triggering on an
  unrelated slider move. Streamlit's real fix for this is `@st.cache_data` (for data-loading
  functions) / `@st.cache_resource` (for models/connections) — decorating a function so its result
  is memoized across reruns and only recomputed when its actual arguments change, instead of on
  every rerun regardless of relevance. Neither script in this folder uses caching yet, which is
  worth flagging explicitly as a real, present gap rather than glossing over it — a natural first
  extension if either script's operations grew more expensive.
- **Losing state across reruns without `st.session_state`.** Because the entire script — including
  every plain variable's initialization — reruns from scratch on every interaction, a plain Python
  variable used as a counter or accumulator resets to its initial value every single rerun; it can
  never "remember" anything from the previous run. `st.session_state` (a dict-like object Streamlit
  persists across reruns *within the same browser session*, not reinitialized each time) is the
  actual mechanism for state that must survive a rerun — e.g. a running tally of button clicks, a
  multi-step wizard's current step, or a list a user is incrementally building up across several
  interactions. Neither `main.py` nor `widgets.py` currently needs this (nothing in them accumulates
  state across reruns), but any extension that did — a click counter, a running list of uploaded
  files — would silently reset on every interaction without it.

## Real-world usage

- Streamlit is a common choice for internal ML/data tooling — a quick model-exploration dashboard, a
  labeling tool, an exploratory-data-analysis app a data scientist can share as a link without
  writing a frontend, especially before (or instead of) building the kind of production API this
  section's `12-flask` topic covers.
- The rerun-the-whole-script model is precisely why Streamlit apps that wrap a real ML model almost
  always use `@st.cache_resource` to load the model once per session rather than on every
  interaction — the exact same "load once, not per-request" principle
  `08-mlops-deployment/06-bentoml`'s serving pipeline uses (`MODEL = joblib.load(...)` at
  module/`__init__` level, never inside the per-request handler), applied to Streamlit's rerun model
  instead of an HTTP request handler.
- `st.file_uploader` (used in `widgets.py`) is the standard Streamlit pattern for letting a
  non-technical user bring their own data into an analysis interactively, without needing to edit
  any code or know a file path.

## Mental model

**A Streamlit script is not a persistent, stateful UI program the way a traditional GUI or web
frontend is — it's a function that gets fully re-called, top to bottom, every time a widget
changes, with each widget call simply returning its current value on that particular call.**
Anything that must survive between calls needs `st.session_state`; anything expensive needs
`@st.cache_data`/`@st.cache_resource` to avoid recomputing on every unrelated interaction.

## Questions to think about

1. `widgets.py` calls `df.to_csv("sample-data.csv")` unconditionally, near the top of the script,
   before any widget-dependent branching. Using the rerun model, explain exactly when this line
   executes relative to a user moving the age slider — and why wrapping it in
   `if not os.path.exists("sample-data.csv"):` would (or wouldn't) fully solve the "recomputes
   unnecessarily" problem `@st.cache_data` is built for.
2. `main.py`'s `st.line_chart(chart_data)` uses `np.random.rand(15, 3)` generated fresh in the
   script body, not behind any widget. Predict what happens to the displayed chart every time any
   widget elsewhere on a combined page (hypothetically merging `main.py` and `widgets.py`) is
   interacted with, and explain why, referencing the rerun model directly.
3. Why can't the from-scratch equivalent of Streamlit be "a manually built HTML page with `<input>`
   tags" in the same useful sense that `raw_http_server.py` is a genuine from-scratch equivalent of
   Flask's routing? What's structurally different about what each framework automates?
4. A teammate wants to add a "total uploads so far" counter to `widgets.py`, incremented every time
   `st.file_uploader` receives a new file. Using a plain module-level Python integer, would this
   counter correctly persist and increase across multiple uploads in the same browser session?
   Justify the answer using the rerun model, and name the specific Streamlit mechanism that would
   fix it if not.
5. `st.selectbox` in `widgets.py` returns the currently selected option on every rerun, the same way
   `st.slider` returns the current position. If a user selects "Python" and then, on a later
   interaction, moves the age slider, what does `choice` evaluate to on that later rerun — and what
   does that reveal about which widgets' values persist across reruns versus which script-level
   variables do?
