"""
LLM Biometrics -- weight-space lineage fingerprinting for open-weight LLMs.

Reference implementation for *Who Built This Model? Tracing LLM Lineage via
Spectral Fingerprints in Weight Space* (COLM 2026).

Two complementary fingerprints are computed from model weights alone, with no
inference, prompts, or training data:

``trace``      Sec. 4 -- spectral energy ``sqrt(sum_i sigma_i^2)`` per layer,
               compared across models by Pearson correlation of the depth-
               aligned curve.  Separates independent-origin from same-series
               and shared-base models, and works across architectures and
               model scales.

``subspace``   Sec. 5 -- alignment of the top-``k`` left singular subspaces,
               scored by the least-aligned principal angles.  Resolves
               fine-grained differences inside the shared-base regime, where
               every magnitude-based metric saturates near 1.

Typical use::

    from llm_biometrics import pipeline
    record = pipeline.compare("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct",
                              method="both", cache_dir="features/")
    print(record["trace"]["score"], record["subspace"]["score"])

or from the shell::

    python -m llm_biometrics compare --model-a A --model-b B
"""

__version__ = "1.0.0"

from . import architectures, cache, extract, pipeline, subspace, trace
from .architectures import COMPONENTS, DEFAULT_COMPONENTS, parse_components
from .extract import DEFAULT_TOP_K, extract_subspace, extract_trace
from .subspace import DEFAULT_J, subspace_similarity
from .trace import trace_similarity

__all__ = [
    "architectures", "cache", "extract", "pipeline", "subspace", "trace",
    "COMPONENTS", "DEFAULT_COMPONENTS", "DEFAULT_J", "DEFAULT_TOP_K",
    "extract_subspace", "extract_trace", "parse_components",
    "subspace_similarity", "trace_similarity", "__version__",
]
