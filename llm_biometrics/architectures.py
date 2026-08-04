"""
Architecture-agnostic access to transformer weight matrices.

This is the single place in the codebase that knows how different HuggingFace
model families name their attention / FFN projections.  Everything else works
with the canonical component names below:

    Q, K, V, O                    -- attention projections
    FFN_GATE, FFN_UP, FFN_DOWN    -- feed-forward projections

which correspond to the component index ``c`` in Eq. (1) of the paper:

    theta = { W_c^(l) | l in [L], c in {Q,K,V,O,FFN_UP,FFN_DOWN,FFN_GATE} }

Fused-projection architectures (Falcon, Pythia/GPT-NeoX, Phi-3) do not expose
separate Q/K/V matrices.  For those we expose the fused matrix under ``Q`` and
the output projection under ``O``; the remaining components are simply absent,
and downstream code compares only the components two models have in common.
"""

from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn


ATTENTION_COMPONENTS = ("Q", "K", "V", "O")
FFN_COMPONENTS = ("FFN_GATE", "FFN_UP", "FFN_DOWN")
COMPONENTS = ATTENTION_COMPONENTS + FFN_COMPONENTS

#: Components used when the caller does not ask for anything specific.
#: The paper's headline numbers use ``--components all`` (all seven).
DEFAULT_COMPONENTS = ("Q", "K", "V", "O")


# Each scheme is (probe_attribute, {component: (candidate attribute names...)}).
# A scheme is selected when its probe attribute is present on the module; every
# component whose candidates are all missing is skipped.
_ATTENTION_SCHEMES: Sequence = (
    # LLaMA / Qwen / Mistral / Gemma / Yi / DeepSeek
    ("q_proj", {"Q": ("q_proj",), "K": ("k_proj",), "V": ("v_proj",), "O": ("o_proj",)}),
    # BERT-style encoders
    ("query", {"Q": ("query",), "K": ("key",), "V": ("value",), "O": ("dense",)}),
    # Short-form naming
    ("q", {"Q": ("q",), "K": ("k",), "V": ("v",), "O": ("o", "dense")}),
    # Fused QKV (Falcon, Phi-3): the fused matrix stands in for Q
    ("qkv", {"Q": ("qkv",), "O": ("out_proj", "dense", "o_proj")}),
    # GPT-NeoX / Pythia
    ("query_key_value", {"Q": ("query_key_value",), "O": ("dense",)}),
)

_FFN_SCHEMES: Sequence = (
    # LLaMA-style gated FFN
    ("gate_proj", {"FFN_GATE": ("gate_proj",), "FFN_UP": ("up_proj",), "FFN_DOWN": ("down_proj",)}),
    # Vanilla transformer FFN
    ("fc1", {"FFN_UP": ("fc1",), "FFN_DOWN": ("fc2",)}),
    # Megatron-style naming
    ("w1", {"FFN_UP": ("w1",), "FFN_DOWN": ("w2",), "FFN_GATE": ("w3",)}),
    # GPT-NeoX / Falcon
    ("dense_h_to_4h", {"FFN_UP": ("dense_h_to_4h",), "FFN_DOWN": ("dense_4h_to_h",)}),
)


class ArchitectureError(RuntimeError):
    """Raised when the transformer layer stack cannot be located."""


def get_base_model(model: nn.Module) -> nn.Module:
    """Unwrap a ``*ForCausalLM`` wrapper down to the bare transformer."""
    return model.model if hasattr(model, "model") else model


def get_transformer_layers(model: nn.Module) -> List[nn.Module]:
    """
    Return the list of transformer blocks, handling the common layouts.

    Accepts either a wrapped model (``AutoModelForCausalLM``) or a bare
    transformer (``AutoModel``).
    """
    base = get_base_model(model)

    if hasattr(base, "layers"):                                        # LLaMA-family
        return list(base.layers)
    if hasattr(base, "transformer") and hasattr(base.transformer, "h"):  # GPT-2 style
        return list(base.transformer.h)
    if hasattr(base, "h"):                                             # Falcon / GPT-NeoX
        return list(base.h)
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):    # BERT style
        return list(base.encoder.layer)
    if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):   # OPT
        return list(base.decoder.layers)

    visible = [a for a in dir(base) if not a.startswith("_")]
    raise ArchitectureError(
        f"Could not locate transformer layers on {type(base).__name__}. "
        f"Available attributes: {visible[:30]}"
    )


def _resolve(module: nn.Module, schemes: Sequence, wanted: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Match ``module`` against the first applicable naming scheme."""
    wanted = set(wanted)
    for probe, mapping in schemes:
        if not hasattr(module, probe):
            continue
        found = {}
        for component, candidates in mapping.items():
            if component not in wanted:
                continue
            for attr in candidates:
                sub = getattr(module, attr, None)
                if sub is not None and hasattr(sub, "weight"):
                    found[component] = sub.weight.detach()
                    break
        return found
    return {}


def _attention_module(layer: nn.Module):
    for attr in ("self_attn", "attention", "attn"):
        mod = getattr(layer, attr, None)
        if mod is not None:
            return mod
    return None


def _ffn_module(layer: nn.Module):
    for attr in ("mlp", "feed_forward", "ffn"):
        mod = getattr(layer, attr, None)
        if mod is not None:
            return mod
    return None


def layer_weight_matrices(layer: nn.Module,
                          components: Iterable[str] = COMPONENTS) -> Dict[str, torch.Tensor]:
    """
    Return ``{component: W_c^(l)}`` for one transformer block.

    Components the architecture does not expose are simply absent from the
    result -- callers must not assume every requested component is present.
    """
    components = set(components)
    weights: Dict[str, torch.Tensor] = {}

    if components & set(ATTENTION_COMPONENTS):
        attn = _attention_module(layer)
        if attn is not None:
            weights.update(_resolve(attn, _ATTENTION_SCHEMES, components))

    if components & set(FFN_COMPONENTS):
        ffn = _ffn_module(layer)
        if ffn is not None:
            weights.update(_resolve(ffn, _FFN_SCHEMES, components))

    return weights


def parse_components(spec: str) -> List[str]:
    """
    Parse a ``--components`` string: ``"all"``, or a comma-separated list such
    as ``"Q,K,V,O"``.  Order follows :data:`COMPONENTS`.
    """
    if spec is None:
        return list(DEFAULT_COMPONENTS)
    if spec.strip().lower() == "all":
        return list(COMPONENTS)

    requested = {token.strip().upper() for token in spec.split(",") if token.strip()}
    invalid = requested - set(COMPONENTS)
    if invalid:
        raise ValueError(
            f"Unknown components {sorted(invalid)}. Available: {list(COMPONENTS)} (or 'all')."
        )
    return [c for c in COMPONENTS if c in requested]
