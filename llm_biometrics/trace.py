"""
Sec. 4 -- coarse-grained lineage discrimination via spectral energy.

Given the layer-wise trace fingerprints ``tau_c(theta)`` produced by
:func:`llm_biometrics.extract.extract_trace`, the similarity between two models
is

    tilde-tau_c(theta)[j] = Interp(tau_c(theta), (j-1)/(L_max-1))     (Eq. 4)
    S_c(a, b)             = Corr(tilde-tau_c(a), tilde-tau_c(b))      (Eq. 5)
    S(a, b)               = mean_c S_c(a, b)

Models of different depth are aligned by piecewise-linear interpolation over the
normalized layer index, so the metric applies across scales (Scenario 2) as well
as within a family.  Each curve is z-scored before correlating, which removes
the absolute magnitude difference between model sizes and leaves the *shape* of
the layer-wise energy profile as the signal.

Scores lie in ``[-1, 1]``; the paper reports them rescaled to ``[0, 1]``
where noted.  Higher means more closely related.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

Fingerprint = Dict[str, np.ndarray]


def interpolate_to(sequence: np.ndarray, length: int) -> np.ndarray:
    """Piecewise-linear resampling over the normalized layer index (Eq. 4)."""
    sequence = np.asarray(sequence, dtype=np.float64)
    if len(sequence) == length:
        return sequence
    if len(sequence) < 2:
        return np.repeat(sequence, length)
    source = np.linspace(0.0, 1.0, len(sequence))
    target = np.linspace(0.0, 1.0, length)
    return interp1d(source, sequence, kind="linear")(target)


def zscore(sequence: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization; all-zeros for a constant curve."""
    sequence = np.asarray(sequence, dtype=np.float64)
    sd = np.std(sequence)
    if sd < 1e-10:
        return np.zeros_like(sequence)
    return (sequence - np.mean(sequence)) / sd


def trace_from_subspace(bases: Dict[str, List]) -> Fingerprint:
    """
    Recover a trace fingerprint from cached subspace bases (or bare singular
    values), without re-reading the weights.

    This uses ``sqrt(sum_{i<=k} sigma_i^2)`` over the *retained* ``k``
    directions rather than the full spectrum.  Since the metric compares the
    shape of the z-scored layer curve, the truncated energy tracks the full
    Frobenius curve closely -- but for exact Eq. (2) values, extract from
    weights with :func:`llm_biometrics.extract.extract_trace`.
    """
    fingerprint: Fingerprint = {}
    for component, per_layer in bases.items():
        values = []
        for item in per_layer:
            sigma = np.asarray(item[1] if isinstance(item, tuple) else item, dtype=np.float64)
            values.append(float(np.sqrt(np.sum(sigma ** 2))))
        fingerprint[component] = np.asarray(values, dtype=np.float64)
    return fingerprint


def trace_similarity(fingerprint_a: Fingerprint,
                     fingerprint_b: Fingerprint,
                     components: Optional[Iterable[str]] = None,
                     ) -> Tuple[float, Dict[str, float]]:
    """
    Trace similarity ``S(theta_a, theta_b)`` (Eq. 5).

    Args:
        fingerprint_a, fingerprint_b: ``{component: per-layer trace array}``.
        components: Restrict the comparison; defaults to every component the
            two models share.

    Returns:
        ``(overall, per_component)``.  ``overall`` is the mean of the
        per-component Pearson correlations, or ``nan`` if nothing was
        comparable.
    """
    shared = set(fingerprint_a) & set(fingerprint_b)
    if components is not None:
        requested = set(components)
        missing = requested - shared
        if missing:
            print(f"Warning: components not present in both models: {sorted(missing)}")
        shared &= requested

    if not shared:
        raise ValueError("The two fingerprints share no comparable components.")

    per_component: Dict[str, float] = {}
    for component in sorted(shared):
        curve_a = np.asarray(fingerprint_a[component], dtype=np.float64)
        curve_b = np.asarray(fingerprint_b[component], dtype=np.float64)

        if len(curve_a) < 2 or len(curve_b) < 2:
            print(f"Warning: skipping {component} -- fewer than two layers")
            continue
        if not (np.isfinite(curve_a).all() and np.isfinite(curve_b).all()):
            print(f"Warning: skipping {component} -- non-finite trace values")
            continue

        length = max(len(curve_a), len(curve_b))                   # L_max
        curve_a = zscore(interpolate_to(curve_a, length))
        curve_b = zscore(interpolate_to(curve_b, length))

        if np.std(curve_a) < 1e-10 or np.std(curve_b) < 1e-10:
            print(f"Warning: skipping {component} -- zero variance across layers")
            continue

        per_component[component] = float(stats.pearsonr(curve_a, curve_b)[0])

    overall = float(np.mean(list(per_component.values()))) if per_component else float("nan")
    return overall, per_component


def interpret(score: float) -> str:
    """Rough verbal reading of a trace score, for CLI output only."""
    if not np.isfinite(score):
        return "no comparable components"
    if score > 0.9:
        return "very high -- shared base or near-identical weights"
    if score > 0.7:
        return "high -- same series is likely"
    if score > 0.5:
        return "moderate -- some structural relatedness"
    return "low -- consistent with independent origin"
