"""
Sec. 5 -- fine-grained lineage discrimination via subspace alignment.

For each layer ``l`` and component ``c``, take the top-``k`` left singular
vectors of the two models and form the cross-subspace matrix

    C_c^(l) = U_{c,k}^(l)(theta_a)^T  U_{c,k}^(l)(theta_b)             (Eq. 7)

Its singular values ``sigma_i(C) in [0, 1]`` are the cosines of the principal
angles between the two subspaces.  The *worst-aligned* directions carry the
fine-grained signal, so the layer score averages the ``J`` smallest:

    S_c^(l)(a, b) = (1/J) sum_{i in I_J} sigma_i(C_c^(l))              (Eq. 8)

Aggregation is hierarchical (Alg. A2): per component, average the ``K_layer``
least-aligned layers; then average over components.

    S_c = mean of the K_layer smallest S_c^(l)
    S   = mean_c S_c

``S = 1`` means the two subspaces coincide, ``S = 0`` that they are orthogonal.
Because the comparison is index-aligned per layer, this metric applies to
models of matching architecture and depth -- the shared-base regime (S3).
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .defaults import DEFAULT_J, DEFAULT_N_BOTTOM

SubspaceBases = Dict[str, List[Tuple[np.ndarray, np.ndarray]]]


def principal_cosines(U_a: np.ndarray, U_b: np.ndarray) -> np.ndarray:
    """
    Singular values of ``C = U_a^T U_b`` (Eq. 7), ascending.

    These are ``cos(theta_i)`` for the principal angles between the two
    subspaces, clipped to ``[0, 1]`` for numerical safety.  If the two bases
    differ in shape (different hidden size or retained ``k``) both are
    truncated to the common leading block.
    """
    U_a = np.asarray(U_a)
    U_b = np.asarray(U_b)
    if U_a.size == 0 or U_b.size == 0:
        return np.zeros(0, dtype=np.float64)

    rows = min(U_a.shape[0], U_b.shape[0])
    cols = min(U_a.shape[1], U_b.shape[1])
    if rows == 0 or cols == 0:
        return np.zeros(0, dtype=np.float64)

    C = U_a[:rows, :cols].T @ U_b[:rows, :cols]
    sigma = np.linalg.svd(C, compute_uv=False)
    return np.sort(np.clip(sigma, 0.0, 1.0))


def layer_alignment(U_a: np.ndarray, U_b: np.ndarray, J: int = DEFAULT_J) -> float:
    """``S_c^(l)`` -- mean of the ``J`` smallest principal cosines (Eq. 8)."""
    sigma = principal_cosines(U_a, U_b)
    if sigma.size == 0:
        return float("nan")
    return float(np.mean(sigma[:min(J, sigma.size)]))


def per_layer_alignment(bases_a: SubspaceBases,
                        bases_b: SubspaceBases,
                        component: str,
                        J: int = DEFAULT_J) -> np.ndarray:
    """
    ``[S_c^(1), ..., S_c^(L)]`` for one component.

    Layers are matched by index, so this is meaningful only for models of the
    same depth; if the depths differ the common prefix is used and a warning is
    printed by :func:`subspace_similarity`.
    """
    layers_a, layers_b = bases_a[component], bases_b[component]
    depth = min(len(layers_a), len(layers_b))
    return np.array([layer_alignment(layers_a[i][0], layers_b[i][0], J) for i in range(depth)])


def subspace_similarity(bases_a: SubspaceBases,
                        bases_b: SubspaceBases,
                        components: Optional[Iterable[str]] = None,
                        J: int = DEFAULT_J,
                        n_bottom: int = DEFAULT_N_BOTTOM,
                        ) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Subspace-alignment similarity ``S(theta_a, theta_b)`` (Alg. A2).

    Args:
        bases_a, bases_b: ``{component: [(U_k, sigma_k) per layer]}``.
        components: Restrict the comparison; defaults to the shared components.
        J: Least-aligned singular directions averaged per layer (Eq. 8).
        n_bottom: Least-aligned layers averaged per component (``K_layer``).

    Returns:
        ``(overall, per_component, per_layer)`` where ``overall in [0, 1]``,
        ``per_component`` maps each component to ``S_c``, and ``per_layer``
        keeps the full ``S_c^(l)`` curves for inspection and plotting.
    """
    shared = set(bases_a) & set(bases_b)
    if components is not None:
        requested = set(components)
        missing = requested - shared
        if missing:
            print(f"Warning: components not present in both models: {sorted(missing)}")
        shared &= requested

    if not shared:
        raise ValueError("The two models share no comparable components.")

    per_component: Dict[str, float] = {}
    per_layer: Dict[str, np.ndarray] = {}

    for component in sorted(shared):
        depth_a, depth_b = len(bases_a[component]), len(bases_b[component])
        if depth_a != depth_b:
            print(f"Warning: {component} depth mismatch ({depth_a} vs {depth_b}); "
                  f"comparing the first {min(depth_a, depth_b)} layers by index")

        curve = per_layer_alignment(bases_a, bases_b, component, J)
        curve = curve[np.isfinite(curve)]
        if curve.size == 0:
            continue

        per_layer[component] = curve
        # S_c = mean of the n_bottom least-aligned layers
        per_component[component] = float(np.mean(np.sort(curve)[:min(n_bottom, curve.size)]))

    overall = float(np.mean(list(per_component.values()))) if per_component else float("nan")
    return overall, per_component, per_layer


def interpret(score: float) -> str:
    """Rough verbal reading of a subspace score, for CLI output only."""
    if not np.isfinite(score):
        return "no comparable components"
    if score > 0.95:
        return "near-identical subspaces -- light or no post-training"
    if score > 0.8:
        return "well aligned -- shared base with moderate post-training"
    if score > 0.5:
        return "partially rotated -- substantial post-training divergence"
    return "strongly rotated -- heavy post-training, or not a shared base"
