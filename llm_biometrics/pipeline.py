"""
End-to-end pair comparison: extract (or load from cache) then score.

This is the layer the CLI sits on, and the one to import when scripting a
custom study:

    from llm_biometrics import pipeline
    result = pipeline.compare("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct",
                              method="both", cache_dir="features/")
"""

import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from . import cache, extract, subspace, trace
from .architectures import DEFAULT_COMPONENTS


def get_trace_fingerprint(model_name: str,
                          components: Sequence[str],
                          cache_dir: Optional[str] = None,
                          device: str = "cuda",
                          multi_gpu: bool = False,
                          noise_std: float = 0.0,
                          noise_seed: int = 42,
                          verbose: bool = True) -> Dict[str, np.ndarray]:
    """Load the trace fingerprint from cache, or extract and cache it."""
    path = cache.trace_path(model_name, cache_dir, noise_std) if cache_dir else None

    if path and os.path.exists(path):
        if verbose:
            print(f"  cache hit: {path}")
        return cache.load_trace(path)

    fingerprint = extract.extract_trace(
        model_name, components=components, device=device, multi_gpu=multi_gpu,
        noise_std=noise_std, noise_seed=noise_seed, verbose=verbose)

    if path:
        cache.save_trace(fingerprint, path)
        if verbose:
            print(f"  cached: {path}")
    return fingerprint


def get_subspace_bases(model_name: str,
                       components: Sequence[str],
                       top_k: int = extract.DEFAULT_TOP_K,
                       cache_dir: Optional[str] = None,
                       device: str = "cuda",
                       multi_gpu: bool = False,
                       noise_std: float = 0.0,
                       noise_seed: int = 42,
                       verbose: bool = True) -> Dict[str, List]:
    """Load subspace bases from cache, or extract and cache them."""
    path = cache.subspace_path(model_name, cache_dir, noise_std) if cache_dir else None

    if path and os.path.exists(path):
        if verbose:
            print(f"  cache hit: {path}")
        return cache.load_subspace(path)

    bases = extract.extract_subspace(
        model_name, components=components, top_k=top_k, device=device,
        multi_gpu=multi_gpu, noise_std=noise_std, noise_seed=noise_seed, verbose=verbose)

    if path:
        cache.save_subspace(bases, path)
        if verbose:
            print(f"  cached: {path}")
    return bases


def compare(model_a: str,
            model_b: str,
            method: str = "both",
            components: Iterable[str] = DEFAULT_COMPONENTS,
            top_k: int = extract.DEFAULT_TOP_K,
            J: int = subspace.DEFAULT_J,
            n_bottom: int = subspace.DEFAULT_N_BOTTOM,
            cache_dir: Optional[str] = None,
            device: str = "cuda",
            multi_gpu: bool = False,
            noise_std: float = 0.0,
            noise_seed: int = 42,
            verbose: bool = True) -> Dict:
    """
    Compare two models and return a JSON-serializable result record.

    Args:
        method: ``"trace"`` (Sec. 4), ``"subspace"`` (Sec. 5), or ``"both"``.
        components: Components to compare; ``"all"`` (seven) reproduces the paper.
        top_k: Singular directions retained per matrix (subspace only).
        J: Least-aligned directions averaged per layer, Eq. (8).
        n_bottom: Least-aligned layers averaged per component.
        cache_dir: Where fingerprints are cached; ``None`` disables caching.
        noise_std: Gaussian weight perturbation applied before extraction.

    Returns:
        ``{"model_a", "model_b", "trace": {...}, "subspace": {...}, ...}``.
        Model order does not affect either score -- both metrics are symmetric.
    """
    components = list(components)
    if method not in ("trace", "subspace", "both"):
        raise ValueError(f"Unknown method '{method}'; expected trace, subspace or both.")

    record: Dict = {
        "model_a": model_a,
        "model_b": model_b,
        "components": components,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if noise_std:
        record["noise_std"] = noise_std
        record["noise_seed"] = noise_seed

    if method in ("trace", "both"):
        if verbose:
            print(f"\n[trace] {model_a}")
        fingerprint_a = get_trace_fingerprint(model_a, components, cache_dir, device,
                                              multi_gpu, noise_std, noise_seed, verbose)
        if verbose:
            print(f"[trace] {model_b}")
        fingerprint_b = get_trace_fingerprint(model_b, components, cache_dir, device,
                                              multi_gpu, noise_std, noise_seed, verbose)

        overall, per_component = trace.trace_similarity(fingerprint_a, fingerprint_b, components)
        record["trace"] = {
            "score": overall,
            "per_component": per_component,
            "layers_a": {c: int(len(v)) for c, v in fingerprint_a.items()},
            "layers_b": {c: int(len(v)) for c, v in fingerprint_b.items()},
        }

    if method in ("subspace", "both"):
        if verbose:
            print(f"\n[subspace] {model_a}")
        bases_a = get_subspace_bases(model_a, components, top_k, cache_dir, device,
                                     multi_gpu, noise_std, noise_seed, verbose)
        if verbose:
            print(f"[subspace] {model_b}")
        bases_b = get_subspace_bases(model_b, components, top_k, cache_dir, device,
                                     multi_gpu, noise_std, noise_seed, verbose)

        overall, per_component, per_layer = subspace.subspace_similarity(
            bases_a, bases_b, components, J=J, n_bottom=n_bottom)
        record["subspace"] = {
            "score": overall,
            "per_component": per_component,
            "top_k": top_k,
            "J": J,
            "n_bottom": n_bottom,
            "per_layer": {c: [float(x) for x in curve] for c, curve in per_layer.items()},
        }

    return record


def format_result(record: Dict) -> str:
    """Human-readable rendering of a :func:`compare` record."""
    lines = ["=" * 72,
             f"{record['model_a']}",
             f"{record['model_b']}",
             "=" * 72]

    if "trace" in record:
        block = record["trace"]
        lines.append("")
        lines.append("Trace  (Sec. 4, Eq. 5): spectral energy across layers")
        for component in sorted(block["per_component"]):
            lines.append(f"    {component:<10s} {block['per_component'][component]:+.4f}")
        lines.append(f"    {'OVERALL':<10s} {block['score']:+.4f}   {trace.interpret(block['score'])}")

    if "subspace" in record:
        block = record["subspace"]
        lines.append("")
        lines.append(f"Subspace  (Sec. 5, Eq. 8): k={block['top_k']}, J={block['J']}, "
                     f"bottom {block['n_bottom']} layers")
        for component in sorted(block["per_component"]):
            lines.append(f"    {component:<10s} {block['per_component'][component]:.4f}")
        lines.append(f"    {'OVERALL':<10s} {block['score']:.4f}   {subspace.interpret(block['score'])}")

    lines.append("=" * 72)
    return "\n".join(lines)


def append_result(record: Dict, path: str) -> None:
    """Append a record to a JSON array file, creating it if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    existing: List = []
    if os.path.exists(path):
        try:
            with open(path) as handle:
                loaded = json.load(handle)
            existing = loaded if isinstance(loaded, list) else [loaded]
        except (json.JSONDecodeError, OSError) as error:
            backup = path + ".corrupted"
            os.replace(path, backup)
            print(f"Warning: could not parse {path} ({error}); moved to {backup}")

    existing.append(record)
    with open(path, "w") as handle:
        json.dump(existing, handle, indent=2)
