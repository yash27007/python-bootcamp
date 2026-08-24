# Streamlit

## What you'll learn

How Streamlit turns a plain Python script into an interactive web UI with no HTML/CSS/JS — and the
execution model that makes this possible: the entire script reruns top to bottom on every widget
interaction, with each widget call simply returning its current value on that run.

## Why it matters

A finished analysis or model is only useful to someone who can run Python code, until it has a UI
someone else can interact with. Building that UI the traditional way means a full separate frontend
project; Streamlit removes that layer for the common case of "a Python script that computes
something, with a few inputs a user should be able to change."

## Prerequisites

- `12-flask` — the general-purpose alternative (a real HTTP API/frontend); this topic explains why
  Streamlit is a narrower, faster path to an interactive UI for the specific case of exploring a
  Python analysis, not a replacement for a real API.

## What you'll build

Nothing new is executed here in a notebook-execution sense — Streamlit apps need a live browser
session (`streamlit run`), which this write-up describes explicitly rather than fakes. What's
covered:

- Why this topic has no from-scratch section (a real, deliberate judgment call, documented in
  `notes.md` rather than silently skipped) — the genuine from-scratch equivalent (raw HTML/CSS/JS)
  isn't the pedagogical point of this topic.
- A close read of the existing `main.py`/`widgets.py`, tied directly to the rerun-the-whole-script
  execution model, including a concrete, checkable prediction about when `widgets.py`'s
  `df.to_csv(...)` line actually re-executes.

See [`notes.md`](notes.md) for the full write-up, and [`main.py`](main.py)/[`widgets.py`](widgets.py)
for the existing apps.

## Where it shows up in real systems

Internal ML/data tooling — quick model-exploration dashboards, labeling tools, exploratory-data-
analysis apps shared as a link without a frontend build step, often as a faster first step before
(or instead of) building the kind of production HTTP API `12-flask` covers.

## What's next

This is the last topic in `01-python-foundation`. `02-statistics` builds the mathematical grounding
the rest of the curriculum (`03-data-analysis` onward) depends on.
