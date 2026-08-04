"""
Fingerprint extraction from model weights.

Two fingerprints are extracted here, matching the two halves of the paper:

* :func:`extract_trace` -- Sec. 4, spectral **trace** fingerprint.
  Per layer and component, ``t(W) = sqrt(tr(W^T W)) = sqrt(sum_i sigma_i^2)``
  (Eq. 2), stacked over layers into ``tau_c(theta)`` (Eq. 3).  Computed as the
  Frobenius norm, which is well defined for rectangular matrices and needs no
  SVD.

* :func:`extract_subspace` -- Sec. 5, **subspace alignment** fingerprint.
  Per layer and component, the top-``k`` left singular vectors ``U_{c,k}^(l)``
  of the compact SVD (Eq. 6), retained together with the singular values.

Both extractors walk the model once and never move a full weight matrix to CPU
before reducing it, so a 14B model fits comfortably on a single GPU.
"""

import gc
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel

from .architectures import (
    COMPONENTS,
    DEFAULT_COMPONENTS,
    get_transformer_layers,
    layer_weight_matrices,
)
from .defaults import DEFAULT_TOP_K


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(model_name: str,
               device: str = "cuda",
               multi_gpu: bool = False,
               noise_std: float = 0.0,
               noise_seed: int = 42):
    """
    Load a model's transformer trunk (``AutoModel``) for weight inspection.

    Args:
        model_name: HuggingFace hub id or local path.
        device: ``"cuda"``, ``"cpu"`` or ``"auto"``.
        multi_gpu: Shard across visible GPUs (``device_map="auto"``); required
            for models that do not fit on one card.
        noise_std: If > 0, add ``N(0, noise_std^2)`` to every weight matrix
            (skipping norms and biases) before extraction.  This reproduces the
            Gaussian-perturbation robustness study of Appendix B/D.
        noise_seed: Seed for the perturbation.
    """
    if multi_gpu or device == "auto":
        device_map, dtype = "auto", torch.float16
    elif device == "cuda":
        device_map, dtype = None, torch.float16
    else:
        device_map, dtype = None, torch.float32

    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if device_map is not None:
        kwargs["device_map"] = device_map

    try:
        model = AutoModel.from_pretrained(model_name, **kwargs)
    except TypeError:
        # transformers >= 4.56 renamed torch_dtype -> dtype
        kwargs["dtype"] = kwargs.pop("torch_dtype")
        model = AutoModel.from_pretrained(model_name, **kwargs)

    if device_map is None and device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    if noise_std > 0:
        _perturb_weights_(model, noise_std, noise_seed)

    return model


def _perturb_weights_(model, noise_std: float, seed: int) -> None:
    """Add Gaussian noise in place, one parameter at a time to cap peak memory."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    for name, param in model.named_parameters():
        is_weight_matrix = ("weight" in name and "norm" not in name.lower()) or "mlp" in name.lower()
        if not is_weight_matrix:
            continue
        if param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            continue
        noise = (torch.randn(param.shape, dtype=param.dtype, device="cpu") * noise_std)
        param.data.add_(noise.to(param.device))
        del noise

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_model(model) -> None:
    """Drop a model and reclaim GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# Generic layer walk
# --------------------------------------------------------------------------- #

def _walk(model,
          components: Sequence[str],
          reduce_fn: Callable[[torch.Tensor], object],
          verbose: bool = True) -> Dict[str, List]:
    """
    Apply ``reduce_fn`` to every requested ``W_c^(l)`` and collect the results
    per component, in layer order.
    """
    layers = get_transformer_layers(model)
    collected: Dict[str, List] = {c: [] for c in components}

    for index, layer in enumerate(layers):
        if verbose and index % 10 == 0:
            print(f"  layer {index}/{len(layers)}")
        for component, weight in layer_weight_matrices(layer, components).items():
            collected[component].append(reduce_fn(weight))

    found = {c: v for c, v in collected.items() if v}
    missing = set(components) - set(found)
    if missing and verbose:
        print(f"  note: architecture does not expose {sorted(missing)}")
    return found


# --------------------------------------------------------------------------- #
# Sec. 4 -- spectral trace fingerprint
# --------------------------------------------------------------------------- #

def spectral_trace(weight: torch.Tensor) -> float:
    """
    ``t(W) = sqrt(tr(W^T W)) = sqrt(sum_i sigma_i^2)``  (Eq. 2).

    Equal to the Frobenius norm, so it is defined for rectangular matrices and
    requires no SVD.
    """
    if weight.ndim != 2:
        weight = weight.reshape(weight.shape[0], -1)
    return float(torch.linalg.norm(weight.float(), ord="fro").item())


def extract_trace(model_name: str,
                  components: Iterable[str] = DEFAULT_COMPONENTS,
                  device: str = "cuda",
                  multi_gpu: bool = False,
                  noise_std: float = 0.0,
                  noise_seed: int = 42,
                  verbose: bool = True,
                  model=None) -> Dict[str, np.ndarray]:
    """
    Extract the layer-wise spectral trace fingerprint ``tau_c(theta)`` (Eq. 3).

    Returns:
        ``{component: array of shape (L,)}`` -- one trace value per layer.

    Pass an already-loaded ``model`` to skip loading (the caller then owns it).
    """
    components = list(components)
    owns_model = model is None
    if owns_model:
        if verbose:
            print(f"Loading {model_name}")
        model = load_model(model_name, device=device, multi_gpu=multi_gpu,
                           noise_std=noise_std, noise_seed=noise_seed)

    if verbose:
        print(f"Extracting spectral traces for {components}")

    with torch.no_grad():
        collected = _walk(model, components, spectral_trace, verbose=verbose)

    if owns_model:
        release_model(model)

    return {c: np.asarray(v, dtype=np.float32) for c, v in collected.items()}


# --------------------------------------------------------------------------- #
# Sec. 5 -- subspace (SVD basis) fingerprint
# --------------------------------------------------------------------------- #

def top_k_svd(weight: torch.Tensor, k: int = DEFAULT_TOP_K) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compact SVD of ``W`` (Eq. 6), truncated to the leading ``k`` directions.

    Returns ``(U_k, sigma_k)`` as float32 numpy arrays of shape ``(m, k)`` and
    ``(k,)``.  ``k`` is clipped to ``min(m, n)``.
    """
    if weight.ndim != 2:
        weight = weight.reshape(weight.shape[0], -1)

    effective_k = min(k, min(weight.shape))
    if effective_k < 1:
        return np.zeros((weight.shape[0], 0), dtype=np.float32), np.zeros(0, dtype=np.float32)

    U, s, _ = torch.linalg.svd(weight.float(), full_matrices=False)
    return (U[:, :effective_k].cpu().numpy().astype(np.float32),
            s[:effective_k].cpu().numpy().astype(np.float32))


def extract_subspace(model_name: str,
                     components: Iterable[str] = DEFAULT_COMPONENTS,
                     top_k: int = DEFAULT_TOP_K,
                     device: str = "cuda",
                     multi_gpu: bool = False,
                     noise_std: float = 0.0,
                     noise_seed: int = 42,
                     verbose: bool = True,
                     model=None) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Extract per-layer top-``k`` singular subspaces ``[U_{c,k}^(1), ..., U_{c,k}^(L)]``.

    Returns:
        ``{component: [(U_k, sigma_k) per layer]}``.

    The singular values are kept alongside ``U`` so a cached subspace file can
    also serve the Sec. 4 trace metric without re-reading the weights (see
    :func:`trace_from_subspace`).
    """
    components = list(components)
    owns_model = model is None
    if owns_model:
        if verbose:
            print(f"Loading {model_name}")
        model = load_model(model_name, device=device, multi_gpu=multi_gpu,
                           noise_std=noise_std, noise_seed=noise_seed)

    if verbose:
        print(f"Extracting top-{top_k} singular subspaces for {components}")

    with torch.no_grad():
        collected = _walk(model, components, lambda w: top_k_svd(w, top_k), verbose=verbose)

    if owns_model:
        release_model(model)

    return collected
