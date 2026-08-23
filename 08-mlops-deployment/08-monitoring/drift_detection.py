"""
From-scratch Population Stability Index (PSI) drift detector.

PSI compares a *reference* distribution (e.g. the training-time feature
distribution) against a *live-window* distribution (e.g. this week's
production traffic for the same feature) and returns a single number
summarizing how much they differ. See notes.md "Mathematical foundation"
for the derivation (PSI is the discretized, binned Jeffreys divergence —
the symmetrized KL divergence — between the two distributions).

Run directly to execute the toy experiment described in notes.md's
"Experiment" section:

    .venv/bin/python 08-mlops-deployment/08-monitoring/drift_detection.py
"""

from __future__ import annotations

import numpy as np

# Conventional PSI thresholds (widely used in industry, e.g. credit-risk
# and ML-monitoring literature): below 0.1 the two distributions are
# considered practically the same; 0.1-0.25 is a moderate shift worth a
# look; above 0.25 is a significant shift that should raise an alert.
PSI_NO_SHIFT = 0.10
PSI_SIGNIFICANT_SHIFT = 0.25

# A floor for any bin proportion before it enters a division or a log, so
# a bin that happens to be empty in one window doesn't produce inf/NaN.
EPSILON = 1e-4


def compute_psi(reference: np.ndarray, live: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between `reference` and `live`, for one
    continuous feature.

    Binning strategy: bin edges are the reference distribution's
    equal-frequency (quantile) cut points, so by construction each bin
    holds ~1/n_bins of the reference mass. The live window is then binned
    using those *same, fixed* edges — this is what makes PSI a comparison
    against a fixed baseline rather than a comparison of two arbitrary
    binnings.
    """
    reference = np.asarray(reference, dtype=np.float64)
    live = np.asarray(live, dtype=np.float64)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(reference, quantiles)
    # Guarantee the true min/max are captured even for out-of-range live
    # values (np.histogram would otherwise silently drop them).
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    live_counts, _ = np.histogram(live, bins=edges)

    p = ref_counts / ref_counts.sum()  # reference proportions per bin
    q = live_counts / live.shape[0]  # live proportions per bin

    # Floor both so an empty bin in either window can't divide by zero or
    # take log(0) — see notes.md "Failure modes" for why this matters.
    p = np.clip(p, EPSILON, None)
    q = np.clip(q, EPSILON, None)

    psi_per_bin = (p - q) * np.log(p / q)
    return float(psi_per_bin.sum())


def classify(psi: float) -> str:
    if psi < PSI_NO_SHIFT:
        return "no significant shift"
    if psi < PSI_SIGNIFICANT_SHIFT:
        return "moderate shift"
    return "significant shift"


def _run_demo() -> None:
    print("Hypothesis:")
    print(
        "  PSI on the UNSHIFTED live window should stay below "
        f"{PSI_NO_SHIFT} (no significant shift)."
    )
    print(
        "  PSI on the SHIFTED live window (mean +0.8 std, wider spread) "
        f"should exceed {PSI_SIGNIFICANT_SHIFT} (significant shift)."
    )
    print()

    rng = np.random.default_rng(42)

    # Reference distribution: the "training-time" feature distribution.
    reference = rng.normal(loc=50.0, scale=10.0, size=2000)

    # Unshifted live window: same distribution, independent draw, smaller
    # sample (a realistic "one week of production traffic" size).
    live_unshifted = rng.normal(loc=50.0, scale=10.0, size=500)

    # Shifted live window: mean shifted by 0.8 reference-std and spread
    # widened — a realistic "upstream data source changed" scenario.
    live_shifted = rng.normal(loc=58.0, scale=13.0, size=500)

    psi_unshifted = compute_psi(reference, live_unshifted, n_bins=10)
    psi_shifted = compute_psi(reference, live_shifted, n_bins=10)

    print("Actual result:")
    print(
        f"  PSI(reference, unshifted live) = {psi_unshifted:.4f}"
        f"  -> {classify(psi_unshifted)}"
    )
    print(
        f"  PSI(reference, shifted live)   = {psi_shifted:.4f}"
        f"  -> {classify(psi_shifted)}"
    )
    print()

    assert psi_unshifted < PSI_NO_SHIFT, "unshifted window unexpectedly flagged as shifted"
    assert psi_shifted > PSI_SIGNIFICANT_SHIFT, "shifted window was NOT flagged as shifted"
    print("Both assertions held: the detector stayed quiet on the unshifted")
    print("window and correctly flagged the shifted one.")


if __name__ == "__main__":
    _run_demo()
