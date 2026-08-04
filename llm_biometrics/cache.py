"""
On-disk fingerprint cache.

Extraction is the expensive step (a full SVD of every weight matrix), so both
fingerprints are cached per model and reused across every pair that mentions
that model.  File names follow the conventions used for the paper runs, so
caches produced by the original scripts remain readable:

    <safe-name>_trace_fingerprint.npz                  trace fingerprint
    <safe-name>_noise_<std>_trace_fingerprint.npz      Gaussian-perturbed
    <safe-name>_svd_bases.pkl                          subspace bases
    <safe-name>_noise_<std>_svd_bases.pkl              Gaussian-perturbed

where ``<safe-name>`` is the model id with ``/`` replaced by ``_``.

Size note: a trace fingerprint is a few kB; subspace bases are large (~2-3 GB
per 7B model at ``k = 512``, since every retained ``U`` is stored). Point
``--cache-dir`` at scratch space, not at the repository.
"""

import os
import pickle
from typing import Dict, List, Optional

import numpy as np

#: Where fingerprints go when the caller does not say.  Deliberately outside
#: the repository -- subspace bases are multi-GB and must never be committed.
#: Override per run with ``--cache-dir``, or globally with $LLM_BIOMETRICS_CACHE.
DEFAULT_CACHE_DIR = os.environ.get(
    "LLM_BIOMETRICS_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "llm-biometrics"))


def safe_name(model_name: str) -> str:
    """Filesystem-safe form of a HuggingFace id or local path."""
    return model_name.replace("/", "_")


def _suffix(noise_std: Optional[float]) -> str:
    if not noise_std:
        return ""
    return "_noise_" + f"{noise_std:.5f}".rstrip("0").rstrip(".")


def trace_path(model_name: str, cache_dir: str, noise_std: Optional[float] = None) -> str:
    """Canonical cache path for a trace fingerprint."""
    return os.path.join(cache_dir, f"{safe_name(model_name)}{_suffix(noise_std)}_trace_fingerprint.npz")


def subspace_path(model_name: str, cache_dir: str, noise_std: Optional[float] = None) -> str:
    """Canonical cache path for subspace bases."""
    return os.path.join(cache_dir, f"{safe_name(model_name)}{_suffix(noise_std)}_svd_bases.pkl")


def save_trace(fingerprint: Dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **fingerprint)


def load_trace(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def save_subspace(bases: Dict[str, List], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(bases, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_subspace(path: str) -> Dict[str, List]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def singular_values_only(bases: Dict[str, List]) -> Dict[str, List[np.ndarray]]:
    """
    Strip the ``U`` matrices, keeping only ``{component: [sigma per layer]}``.

    Subspace caches are dominated by ``U``; the singular values alone are a few
    hundred kB and are all the trace metric needs, which makes bulk analyses
    over many models cheap to re-run.
    """
    return {component: [np.asarray(item[1] if isinstance(item, tuple) else item)
                        for item in per_layer]
            for component, per_layer in bases.items()}
