# Monitoring & drift detection

## Problem

Every topic before this one in `08-mlops-deployment` ends at the moment a
model goes live: it's containerized, tested, versioned, tracked, and
served behind an API. None of that says anything about what happens
*after*. A model's accuracy at deploy time was measured against a
snapshot of the world — the training data's feature distribution, the
relationships between features and the label at that point in time. The
real world doesn't hold still: user behavior changes, an upstream data
source changes units or gets a new default value, a sensor degrades, a
new product category appears that didn't exist in training. The model
itself doesn't change — it keeps computing the exact same function it was
trained to compute — but the *inputs it now receives* can drift away from
what it was trained on, silently degrading its real-world accuracy while
every system around it reports "the model is running fine." **How do you
detect that the live data a deployed model sees is no longer what it was
trained on, without waiting for someone downstream to notice bad
predictions?**

## Intuition

Imagine a thermostat calibrated in a room at sea level, where "72°F"
means a specific, correct thing. Move it to a location at high altitude
with different air pressure and it keeps reporting numbers exactly as
confidently as before — it hasn't broken, it hasn't crashed, it just
isn't calibrated for the air it's actually measuring anymore. Nothing
about the thermostat's internal mechanism signals "something changed
about my environment"; you'd only find out by comparing today's readings
against a reference you trust.

A deployed ML model is the same. It was fit to a particular distribution
of inputs — say, average customer order size clustered around $40 with a
particular spread. If a new marketing channel starts sending customers
who order in bulk, the model doesn't throw an exception; it keeps making
predictions on inputs that look statistically different from anything it
saw during training, with no built-in signal that this happened. The fix
is the same as the thermostat's: keep a reference (the training-time
distribution) and periodically compare a live window of production data
against it, using a real statistic, not a feeling.

## Why simpler approaches fail

The naive approach is: "someone periodically checks accuracy by hand." A
few concrete reasons this doesn't scale:

1. **It requires ground-truth labels the system usually doesn't have
   promptly.** Computing "accuracy" needs the true label for each
   prediction. In production, the true label often arrives late (did the
   loan actually default? did the customer actually churn?) or never
   arrives at all in a form anyone collects. Waiting for labels to check
   accuracy means waiting exactly as long as the label takes to
   materialize — which can be weeks, by which point a lot of bad
   predictions have already shipped.
2. **It doesn't run continuously.** "Occasionally" means someone has to
   remember to look, and — just like the CI/CD checklist problem in
   `07-cicd` — that discipline erodes under normal workload. There's no
   automatic, always-on signal; there's a task on someone's to-do list
   that competes with everything else they're doing.
3. **The first real signal ends up being a downstream complaint.** By the
   time a support ticket says "this recommendation makes no sense" or a
   business stakeholder notices a KPI moving the wrong way, the model has
   likely been silently degraded for some time — the problem is detected
   *after* it has already caused damage, not before.

The structural fix doesn't require waiting for labels at all: instead of
asking "are the predictions still accurate" (which needs labels), ask
"does the live input data still look like the data the model was trained
on" (which only needs the inputs, available immediately, no label
required). That's what distribution-shift ("drift") detection measures,
and it's the earliest possible warning signal — often available well
before enough labels exist to measure accuracy degradation directly.

## Mathematical foundation

The chosen drift statistic here is the **Population Stability Index
(PSI)** — a way to quantify how much a *reference* distribution and a
*live-window* distribution of the same feature have diverged, using only
their binned proportions (no labels required).

**Starting point: relative entropy (KL divergence).** For two discrete
probability distributions $P$ and $Q$ over the same $n$ categories (here:
histogram bins), the Kullback-Leibler divergence of $Q$ from $P$ is

$$D_{KL}(P \| Q) = \sum_{i=1}^{n} P_i \ln\frac{P_i}{Q_i}$$

This measures how surprised you'd be, on average, seeing data generated
by $P$ if you'd assumed it came from $Q$. By Gibbs' inequality,
$D_{KL}(P\|Q) \ge 0$, with equality only when $P = Q$ bin-for-bin — so it
is a valid (if asymmetric) measure of "distance" from $Q$ to $P$.

**The asymmetry problem.** $D_{KL}(P\|Q) \ne D_{KL}(Q\|P)$ in general —
divergence "from training toward live" is a different number than
"from live toward training." For a drift *alert*, this asymmetry is an
unwanted complication: we want one number describing how different the
two distributions are, not two numbers depending on which one we call
the reference.

**Symmetrizing: the Jeffreys divergence.** Add both directions together:

$$D_J(P, Q) = D_{KL}(P\|Q) + D_{KL}(Q\|P) = \sum_{i=1}^n P_i \ln\frac{P_i}{Q_i} + \sum_{i=1}^n Q_i \ln\frac{Q_i}{P_i}$$

The second sum can be rewritten using $\ln(Q_i/P_i) = -\ln(P_i/Q_i)$:

$$\sum_{i=1}^n Q_i \ln\frac{Q_i}{P_i} = -\sum_{i=1}^n Q_i \ln\frac{P_i}{Q_i}$$

Substituting back and combining both sums term-by-term:

$$D_J(P, Q) = \sum_{i=1}^n \left(P_i - Q_i\right) \ln\frac{P_i}{Q_i}$$

This is exactly the **PSI formula**:

$$\mathrm{PSI}(P, Q) = \sum_{i=1}^{n} (P_i - Q_i)\, \ln\!\left(\frac{P_i}{Q_i}\right)$$

where:
- $P_i$ = proportion of the **reference** window's observations falling
  in bin $i$,
- $Q_i$ = proportion of the **live** window's observations falling in
  the *same* bin $i$ (bin edges are fixed from the reference, not
  recomputed from the live data — otherwise this wouldn't be comparing
  against a fixed baseline),
- $n$ = number of bins (10 is conventional).

**Why it's ≥ 0, and why it's 0 only when there's no shift.** $D_J = D_{KL}(P\|Q) + D_{KL}(Q\|P)$
is a sum of two non-negative terms (Gibbs' inequality applies to each),
so $D_J \ge 0$; it is exactly $0$ only when $P_i = Q_i$ for every bin —
i.e. the live window's binned distribution exactly matches the
reference's. Each individual term $(P_i - Q_i)\ln(P_i/Q_i)$ is also
non-negative on its own: $P_i - Q_i$ and $\ln(P_i/Q_i)$ always share the
same sign (if $P_i > Q_i$ then $P_i/Q_i > 1$ so its log is positive; if
$P_i < Q_i$ the reverse), so every term in the sum is a product of two
same-signed quantities — non-negative — and larger per-bin
discrepancies contribute more, which is exactly the "how much have these
distributions diverged" behavior wanted from a drift statistic.

**Conventional thresholds** (empirically established, not derived from
the math above, but standard in industry usage — e.g. credit-risk
scorecards, where PSI originates, and ML monitoring tooling that adopted
it): PSI $< 0.10$ → no significant shift; $0.10 \le$ PSI $< 0.25$ →
moderate shift, worth investigating; PSI $\ge 0.25$ → significant shift,
should raise an alert.

## Algorithm

1. Collect a **reference window**: the feature's distribution at
   training/deploy time (or any trusted baseline period).
2. Choose $n$ bins (10 is conventional) using the reference window's
   **quantile** cut points, so each reference bin holds ≈$1/n$ of the
   reference mass by construction — this makes the binning adapt to the
   feature's actual scale/skew instead of using naive equal-width bins
   that could leave most bins empty for a skewed feature.
3. Collect a **live window**: the same feature over a recent production
   period (e.g. "the last day" or "the last 1,000 requests").
4. Bin the live window using the **same, fixed** edges from step 2 (not
   new quantiles computed on the live data — that would make the metric
   compare two independently-binned distributions instead of measuring
   drift against a fixed baseline).
5. Compute proportions $P_i$, $Q_i$ per bin; floor both away from exactly
   0 (a small epsilon) so an empty bin can't produce a division by zero
   or $\ln(0)$.
6. Compute $\mathrm{PSI} = \sum_i (P_i - Q_i)\ln(P_i/Q_i)$.
7. Compare against the thresholds above; alert if PSI $\ge 0.25$ (or
   $\ge 0.10$ for an earlier, more sensitive warning).

## From-scratch implementation

`drift_detection.py` (this directory):

```python
def compute_psi(reference: np.ndarray, live: np.ndarray, n_bins: int = 10) -> float:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(reference, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf  # capture out-of-range live values

    ref_counts, _ = np.histogram(reference, bins=edges)
    live_counts, _ = np.histogram(live, bins=edges)

    p = np.clip(ref_counts / ref_counts.sum(), EPSILON, None)
    q = np.clip(live_counts / live.shape[0], EPSILON, None)

    return float(((p - q) * np.log(p / q)).sum())
```

Directly implements the derived formula: quantile-bin the reference,
re-bin the live window on the *same* edges, floor both proportions, sum
$(P_i-Q_i)\ln(P_i/Q_i)$ over bins.

## Practical implementation

The from-scratch function above *is* a real, usable implementation of
PSI — production monitoring tools don't use a fundamentally different
formula, they wrap the same statistic (or an equivalent one, like a
two-sample Kolmogorov-Smirnov test comparing empirical CDFs) in
infrastructure this toy script doesn't have:

- **Scheduling and windowing** — running the comparison automatically
  every hour/day against a rolling live window, rather than by hand.
- **Per-feature dashboards and thresholds** — computing PSI (or a
  suite of drift statistics) for every input feature and the model's
  output distribution, not just one feature at a time.
- **Alerting integration** — pushing a Slack/PagerDuty/email alert when a
  threshold is crossed, instead of printing to a terminal.
- **Automatic retraining triggers** — in a mature pipeline, a
  significant-drift alert can kick off the exact CI/CD pipeline from
  `07-cicd`, but with "retrain and re-validate the model" as the pipeline
  instead of "test and deploy the code."

Two real, named tools that implement this category of monitoring:
**Evidently** (open-source; computes PSI, KS-tests, and many other
drift/data-quality metrics between a reference and current dataset, and
renders them as reports/dashboards) and **WhyLabs** (a hosted monitoring
platform built around a compact statistical summary format — "profiles"
— computed continuously on production traffic, with alerting on drift
and other data-quality signals). Neither is installed or run here —
referenced by name because knowing they exist and roughly what problem
they solve is the practical takeaway, not their specific APIs.

## Experiment

**Hypothesis:** computing PSI between a fixed reference distribution and
an *unshifted* live window (drawn from the same distribution) will stay
below the no-significant-shift threshold (0.10). Computing PSI between
the same reference and a *shifted* live window (mean shifted, spread
widened) will exceed the significant-shift threshold (0.25).

**Setup:** `np.random.default_rng(42)` (fixed seed). Reference:
2,000 samples from $N(50, 10)$. Unshifted live window: 500 independent
samples from the same $N(50, 10)$. Shifted live window: 500 samples from
$N(58, 13)$ — mean shifted by 0.8 reference-standard-deviations, spread
widened by 30%, modeling a realistic "the upstream data source changed"
scenario. `n_bins=10` (quantile bins on the reference).

**Actual result** (`.venv/bin/python 08-mlops-deployment/08-monitoring/drift_detection.py`):

```
Hypothesis:
  PSI on the UNSHIFTED live window should stay below 0.1 (no significant shift).
  PSI on the SHIFTED live window (mean +0.8 std, wider spread) should exceed 0.25 (significant shift).

Actual result:
  PSI(reference, unshifted live) = 0.0187  -> no significant shift
  PSI(reference, shifted live)   = 0.6506  -> significant shift

Both assertions held: the detector stayed quiet on the unshifted
window and correctly flagged the shifted one.
```

(Runtime: ~0.1s — pure NumPy on 2,500 synthetic floats, no training
involved.)

**Interpretation:** exactly as hypothesized. The unshifted window's PSI
(0.0187) sits comfortably below the 0.10 no-shift threshold — some
non-zero value is expected even with no true shift, since a live window
of only 500 samples has sampling noise relative to the 2,000-sample
reference, and PSI correctly reflects that as "small" rather than "zero."
The shifted window's PSI (0.6506) is well over twice the 0.25
significant-shift threshold, correctly and strongly flagging the change.
The gap between the two (0.0187 vs 0.6506, roughly 35x) shows the
statistic has good separation, not just a lucky threshold crossing.

**Limitations:** one seed, one feature, one distribution family (Gaussian)
— this demonstrates the mechanism works on a clean synthetic case, not
that PSI=0.25 is a universally correct threshold for every feature/domain
(real deployments often tune thresholds per-feature based on the cost of
false alarms vs. missed drift, see Failure modes below). It also only
tests *input* drift on one numeric feature; a real system runs this per
feature and separately monitors the model's *output* distribution too
(see Failure modes).

## Failure modes

- **Alert fatigue from over-sensitive thresholds.** Setting the alert
  threshold too low (e.g. treating any PSI > 0.05 as urgent) means normal
  sampling noise — exactly the ~0.02 seen on the unshifted window above,
  which will vary run to run — regularly crosses it. Once a team has seen
  enough false alarms, they start ignoring drift alerts by default,
  which means the one alert that mattered gets ignored too. The fix is
  choosing thresholds empirically (e.g. by observing PSI's natural
  variance on known-stable historical windows, the way the experiment
  above shows 0.0187 as a "normal noise" baseline) rather than by
  intuition.
- **Monitoring input drift without checking output/label drift.** PSI on
  input features only tells you the *data* looks different — it doesn't
  directly tell you the *predictions* got worse. A feature can drift in a
  way the model is fully robust to (e.g. a feature it barely weights),
  producing an alert with no real accuracy impact; conversely, accuracy
  can degrade from a subtle relationship change (the same inputs, a
  changed relationship to the true label — "concept drift") that input
  monitoring alone won't catch. A complete monitoring setup tracks input
  drift, the model's *output* (prediction) distribution, and — once
  labels eventually arrive — realized accuracy, together.
- **No monitoring of the monitoring system itself.** The drift monitor is
  itself a running piece of software with its own failure modes: a
  scheduled job that silently stops running, a reference distribution
  computed once and never refreshed even as "normal" legitimately
  evolves, a live-data feed that goes stale so the monitor keeps
  comparing against old cached data and reports "no drift" for the wrong
  reason. A monitoring system that can fail silently is worse than
  useless — it produces false confidence ("no alerts fired" being
  misread as "everything is fine" when it actually means "the check
  didn't run"). Production setups need a heartbeat/liveness check *on*
  the monitoring job itself.

## Real-world usage

Drift monitoring is the standard second half of a production ML system,
paired with the deployment mechanics from `01-docker` through
`07-cicd`: once a model is served, a monitoring job runs continuously (or
on a schedule) comparing live traffic against a stored reference,
surfacing dashboards, and firing alerts. In mature setups, a significant
sustained drift alert is wired to *automatically* trigger a retraining
pipeline — itself just another CI/CD DAG (per `07-cicd`'s conceptual
foundation), except the trigger event is "drift alert crossed threshold"
instead of "git push," and the pipeline's stages are "pull fresh labeled
data → retrain → validate the new model beats the old one → deploy" gated
the same way "test → build → deploy" was gated in `07-cicd`. This is the
loop that closes the full lifecycle this section has built up to: ship a
model, watch it, and automatically respond when it stops matching the
world it was trained on.

## Mental model

**A deployed model doesn't break when the world changes underneath it —
it keeps confidently answering questions about a world that no longer
exists. Drift detection is a standing comparison between "what the model
was trained on" and "what it's actually seeing now," computed
continuously and requiring no ground-truth labels, so degradation is
caught from the input side, early, instead of discovered later from a
downstream complaint.**

## Questions to think about

1. The PSI formula treats a feature whose mean shifted slightly across
   *every* bin the same as a feature whose distribution is identical
   except one bin's proportion moved sharply. Construct two toy
   `(reference, live)` proportion vectors that produce a similar PSI value
   despite looking very different when plotted as histograms. What does
   this tell you about relying on PSI as a single number vs. also
   inspecting the per-bin histogram?
2. This topic's experiment uses `n_bins=10`. What would happen to the
   unshifted window's PSI value (roughly) if `n_bins` were increased to
   100 while keeping the live window at only 500 samples — and why, in
   terms of how many live samples land in each individual bin?
3. Why does the algorithm fix the bin edges from the *reference* window
   only, rather than recomputing quantile edges separately for the live
   window each time PSI is calculated? Construct a scenario where using
   independently-computed edges for each window would hide a real
   shift.
4. Suppose accuracy on labeled data (once labels arrive) is trending down
   but PSI on every individual input feature stays near zero the whole
   time. Given this topic's Failure-modes section, what kind of drift is
   most likely happening, and what would you need to monitor in addition
   to input-feature PSI to catch it?
