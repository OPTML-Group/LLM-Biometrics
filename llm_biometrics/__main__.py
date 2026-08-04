"""
Command-line interface.

    python -m llm_biometrics compare   --model-a A --model-b B [--method both]
    python -m llm_biometrics extract   --model  M  --method subspace
    python -m llm_biometrics run-pairs --pairs configs/pairs/s1_independent_origin.txt

Run any subcommand with ``--help`` for the full option list.
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np

from . import cache, extract, pipeline, subspace
from .architectures import COMPONENTS, parse_components


# --------------------------------------------------------------------------- #
# Shared arguments
# --------------------------------------------------------------------------- #

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=("trace", "subspace", "both"), default="both",
                        help="Trace = Sec. 4 (any pair); subspace = Sec. 5 "
                             "(same architecture and depth). Default: both.")
    parser.add_argument("--components", default="all",
                        help=f"'all' or a comma-separated subset of {list(COMPONENTS)}. "
                             "Default: all (matches the paper).")
    parser.add_argument("--top-k", type=int, default=extract.DEFAULT_TOP_K,
                        help=f"Singular directions retained per weight matrix "
                             f"(default: {extract.DEFAULT_TOP_K}).")
    parser.add_argument("--cache-dir", default=cache.DEFAULT_CACHE_DIR,
                        help="Directory for cached fingerprints. Subspace caches are "
                             "multi-GB per model, so this defaults outside the repo; "
                             "set $LLM_BIOMETRICS_CACHE to change it globally.")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"),
                        default="cuda" if _cuda_available() else "cpu")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Shard the model across visible GPUs (device_map='auto').")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Add N(0, sigma^2) to every weight matrix before extraction "
                             "(Appendix D robustness study used 1e-4).")
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")


def _add_scoring(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-J", "--num-directions", type=int, default=subspace.DEFAULT_J,
                        dest="J",
                        help=f"J in Eq. 8: least-aligned singular directions averaged "
                             f"per layer (default: {subspace.DEFAULT_J}).")
    parser.add_argument("--n-bottom", type=int, default=subspace.DEFAULT_N_BOTTOM,
                        help=f"Least-aligned layers averaged per component "
                             f"(default: {subspace.DEFAULT_N_BOTTOM}).")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #

def cmd_compare(args) -> int:
    components = parse_components(args.components)
    record = pipeline.compare(
        args.model_a, args.model_b,
        method=args.method, components=components, top_k=args.top_k,
        J=args.J, n_bottom=args.n_bottom, cache_dir=args.cache_dir,
        device=args.device, multi_gpu=args.multi_gpu,
        noise_std=args.noise_std, noise_seed=args.noise_seed,
        verbose=not args.quiet)

    print()
    print(pipeline.format_result(record))

    if args.results:
        pipeline.append_result(record, args.results)
        print(f"\nAppended to {args.results}")
    return 0


def cmd_extract(args) -> int:
    components = parse_components(args.components)
    os.makedirs(args.cache_dir, exist_ok=True)
    verbose = not args.quiet

    if args.method in ("trace", "both"):
        fingerprint = pipeline.get_trace_fingerprint(
            args.model, components, args.cache_dir, args.device,
            args.multi_gpu, args.noise_std, args.noise_seed, verbose)
        path = cache.trace_path(args.model, args.cache_dir, args.noise_std)
        print(f"trace     -> {path}")
        for component in sorted(fingerprint):
            print(f"    {component:<10s} {len(fingerprint[component])} layers")

    if args.method in ("subspace", "both"):
        bases = pipeline.get_subspace_bases(
            args.model, components, args.top_k, args.cache_dir, args.device,
            args.multi_gpu, args.noise_std, args.noise_seed, verbose)
        path = cache.subspace_path(args.model, args.cache_dir, args.noise_std)
        print(f"subspace  -> {path}")
        for component in sorted(bases):
            U, _ = bases[component][0]
            print(f"    {component:<10s} {len(bases[component])} layers, U shape {U.shape}")

    return 0


def read_pairs(path: str) -> List[Tuple[str, str]]:
    """
    Read a pair list: one ``model_a  model_b`` per line, ``#`` starts a comment.
    """
    pairs = []
    with open(path) as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{path}:{lineno}: expected two model ids, got {len(fields)}")
            pairs.append((fields[0], fields[1]))
    return pairs


def cmd_run_pairs(args) -> int:
    components = parse_components(args.components)
    pairs = read_pairs(args.pairs)
    print(f"{len(pairs)} pairs from {args.pairs}\n")

    scores = []
    for index, (model_a, model_b) in enumerate(pairs, 1):
        print(f"[{index}/{len(pairs)}] {model_a}  vs  {model_b}")
        try:
            record = pipeline.compare(
                model_a, model_b,
                method=args.method, components=components, top_k=args.top_k,
                J=args.J, n_bottom=args.n_bottom, cache_dir=args.cache_dir,
                device=args.device, multi_gpu=args.multi_gpu,
                noise_std=args.noise_std, noise_seed=args.noise_seed,
                verbose=not args.quiet)
        except Exception as error:                       # keep the sweep going
            print(f"    FAILED: {type(error).__name__}: {error}\n")
            if args.results:
                pipeline.append_result(
                    {"model_a": model_a, "model_b": model_b, "error": str(error)},
                    args.results)
            continue

        print(pipeline.format_result(record))
        print()
        scores.append(record)
        if args.results:
            pipeline.append_result(record, args.results)

    _summarize(scores, args.method)
    if args.results:
        print(f"\nResults written to {args.results}")
    return 0


def _summarize(records: List[dict], method: str) -> None:
    if not records:
        print("No pairs completed.")
        return

    print("=" * 72)
    print(f"SUMMARY over {len(records)} pairs")
    print("=" * 72)
    for key, label in (("trace", "Trace"), ("subspace", "Subspace")):
        values = [r[key]["score"] for r in records if key in r and np.isfinite(r[key]["score"])]
        if values:
            print(f"  {label:<9s} mean {np.mean(values):.4f}   "
                  f"std {np.std(values):.4f}   "
                  f"min {np.min(values):.4f}   max {np.max(values):.4f}")


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m llm_biometrics",
        description="Weight-space lineage fingerprinting for open-weight LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare", help="Score one model pair.",
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    compare.add_argument("--model-a", "--model1", dest="model_a", required=True)
    compare.add_argument("--model-b", "--model2", dest="model_b", required=True)
    compare.add_argument("--results", default=None, help="Append the record to this JSON file.")
    _add_common(compare)
    _add_scoring(compare)
    compare.set_defaults(func=cmd_compare)

    extract_cmd = sub.add_parser("extract", help="Extract and cache one model's fingerprints.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    extract_cmd.add_argument("--model", required=True)
    _add_common(extract_cmd)
    extract_cmd.set_defaults(func=cmd_extract)

    run_pairs = sub.add_parser("run-pairs", help="Score every pair in a pair-list file.",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run_pairs.add_argument("--pairs", required=True, help="Path to a pair list (see configs/pairs/).")
    run_pairs.add_argument("--results", default=None, help="Append records to this JSON file.")
    _add_common(run_pairs)
    _add_scoring(run_pairs)
    run_pairs.set_defaults(func=cmd_run_pairs)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
