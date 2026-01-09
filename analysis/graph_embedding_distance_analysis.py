#!/usr/bin/env python3
"""
Analyze graph embedding distances used by pretraining.py.

Loads entity_embedding.npy and relation_embedding.npy from a directory and
reports distance distributions, min/max distances, and cosine similarities.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np

DEFAULT_NUM_PAIRS = 200_000
DEFAULT_BATCH_SIZE = 20_000
DEFAULT_EXACT_MAX_N = 6_000
DEFAULT_CHUNK_SIZE = 1_024
DEFAULT_NN_SAMPLES = 512


def _load_embeddings(embeddings_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    entity_path = os.path.join(embeddings_dir, "entity_embedding.npy")
    relation_path = os.path.join(embeddings_dir, "relation_embedding.npy")
    if not os.path.exists(entity_path) or not os.path.exists(relation_path):
        raise FileNotFoundError(
            f"Expected {entity_path} and {relation_path} in embeddings dir"
        )
    entity = np.load(entity_path)
    relation = np.load(relation_path)
    if entity.ndim != 2 or relation.ndim != 2:
        raise ValueError("Expected 2D arrays for embeddings")
    if entity.shape[1] != relation.shape[1]:
        raise ValueError("Entity and relation embedding dims do not match")
    return entity, relation


def _basic_stats(x: np.ndarray) -> Dict[str, float]:
    norms = np.linalg.norm(x, axis=1)
    return {
        "count": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "centroid_norm": float(np.linalg.norm(np.mean(x, axis=0))),
    }


def _sample_pairs(n: int, count: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    idx_a = np.empty(count, dtype=np.int64)
    idx_b = np.empty(count, dtype=np.int64)
    filled = 0
    while filled < count:
        remaining = count - filled
        draw = int(math.ceil(remaining * 1.1))
        a = rng.integers(0, n, size=draw, dtype=np.int64)
        b = rng.integers(0, n, size=draw, dtype=np.int64)
        mask = a != b
        a = a[mask]
        b = b[mask]
        take = min(remaining, a.shape[0])
        idx_a[filled : filled + take] = a[:take]
        idx_b[filled : filled + take] = b[:take]
        filled += take
    return idx_a, idx_b


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    qs = np.percentile(values, [5, 25, 50, 75, 95, 99])
    return {
        "p05": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p95": float(qs[4]),
        "p99": float(qs[5]),
    }


def _sample_distance_stats(
    x: np.ndarray,
    num_pairs: int,
    rng: np.random.Generator,
    batch_size: int,
) -> Dict[str, float]:
    n = x.shape[0]
    if n < 2:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    num_pairs = min(num_pairs, n * (n - 1) // 2)
    distances = []
    remaining = num_pairs
    while remaining > 0:
        cur = min(batch_size, remaining)
        idx_a, idx_b = _sample_pairs(n, cur, rng)
        diff = x[idx_a] - x[idx_b]
        d = np.linalg.norm(diff, axis=1)
        distances.append(d)
        remaining -= cur
    values = np.concatenate(distances, axis=0)
    stats = {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
    stats.update(_quantiles(values))
    return stats


def _sample_cosine_stats(
    x: np.ndarray,
    num_pairs: int,
    rng: np.random.Generator,
    batch_size: int,
) -> Dict[str, float]:
    n = x.shape[0]
    if n < 2:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    num_pairs = min(num_pairs, n * (n - 1) // 2)
    norms = np.linalg.norm(x, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    values = []
    remaining = num_pairs
    while remaining > 0:
        cur = min(batch_size, remaining)
        idx_a, idx_b = _sample_pairs(n, cur, rng)
        dot = np.sum(x[idx_a] * x[idx_b], axis=1)
        cosine = dot / (norms[idx_a] * norms[idx_b])
        values.append(cosine)
        remaining -= cur
    vals = np.concatenate(values, axis=0)
    stats = {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }
    stats.update(_quantiles(vals))
    return stats


def _exact_min_max_distance(x: np.ndarray, chunk_size: int) -> Tuple[float, float]:
    n = x.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    norms = np.sum(x * x, axis=1)
    min_d2 = float("inf")
    max_d2 = float("-inf")
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = x[start:end]
        d2 = norms[start:end, None] + norms[None, :] - 2.0 * (chunk @ x.T)
        # Remove self-distances within the chunk
        for i in range(start, end):
            d2[i - start, i] = float("inf")
        min_d2 = min(min_d2, float(np.min(d2)))
        max_d2 = max(max_d2, float(np.max(d2)))
    return math.sqrt(min_d2), math.sqrt(max_d2)


def _approx_min_distance(
    x: np.ndarray,
    rng: np.random.Generator,
    nn_samples: int,
    chunk_size: int,
) -> float:
    n = x.shape[0]
    if n < 2:
        return float("nan")
    nn_samples = min(nn_samples, n)
    anchor_idx = rng.choice(n, size=nn_samples, replace=False)
    norms = np.sum(x * x, axis=1)
    best = float("inf")
    for idx in anchor_idx:
        anchor = x[idx : idx + 1]
        d2_min = float("inf")
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = x[start:end]
            d2 = norms[start:end] + norms[idx] - 2.0 * (chunk @ anchor.T).squeeze(1)
            if start <= idx < end:
                d2[idx - start] = float("inf")
            d2_min = min(d2_min, float(np.min(d2)))
        best = min(best, math.sqrt(d2_min))
    return best


def analyze_embeddings(
    name: str,
    x: np.ndarray,
    rng: np.random.Generator,
    num_pairs: int,
    batch_size: int,
    exact_max_n: int,
    chunk_size: int,
    nn_samples: int,
) -> Dict[str, object]:
    stats: Dict[str, object] = {"name": name}
    stats.update(_basic_stats(x))
    stats["pairwise_l2_samples"] = _sample_distance_stats(
        x, num_pairs=num_pairs, rng=rng, batch_size=batch_size
    )
    stats["pairwise_cosine_samples"] = _sample_cosine_stats(
        x, num_pairs=num_pairs, rng=rng, batch_size=batch_size
    )

    if x.shape[0] <= exact_max_n:
        min_d, max_d = _exact_min_max_distance(x, chunk_size=chunk_size)
        stats["min_distance"] = min_d
        stats["max_distance"] = max_d
        stats["min_max_exact"] = True
    else:
        stats["min_distance"] = _approx_min_distance(
            x, rng=rng, nn_samples=nn_samples, chunk_size=chunk_size
        )
        stats["max_distance"] = stats["pairwise_l2_samples"]["max"]
        stats["min_max_exact"] = False
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze graph embeddings used by pretraining.py"
    )
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        help="Directory containing entity_embedding.npy and relation_embedding.npy",
    )
    parser.add_argument(
        "--which",
        choices=["entity", "relation", "both"],
        default="both",
        help="Which embeddings to analyze",
    )
    parser.add_argument("--num-pairs", type=int, default=DEFAULT_NUM_PAIRS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--exact-max-n", type=int, default=DEFAULT_EXACT_MAX_N)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--nn-samples", type=int, default=DEFAULT_NN_SAMPLES)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    entity, relation = _load_embeddings(args.embeddings_dir)

    results = {"embeddings_dir": args.embeddings_dir, "timestamp": time.time()}

    if args.which in ("entity", "both"):
        results["entity"] = analyze_embeddings(
            "entity",
            entity,
            rng,
            num_pairs=args.num_pairs,
            batch_size=args.batch_size,
            exact_max_n=args.exact_max_n,
            chunk_size=args.chunk_size,
            nn_samples=args.nn_samples,
        )
    if args.which in ("relation", "both"):
        results["relation"] = analyze_embeddings(
            "relation",
            relation,
            rng,
            num_pairs=args.num_pairs,
            batch_size=args.batch_size,
            exact_max_n=args.exact_max_n,
            chunk_size=args.chunk_size,
            nn_samples=args.nn_samples,
        )

    if args.which == "both":
        combined = np.concatenate([entity, relation], axis=0)
        results["combined"] = analyze_embeddings(
            "combined",
            combined,
            rng,
            num_pairs=args.num_pairs,
            batch_size=args.batch_size,
            exact_max_n=args.exact_max_n,
            chunk_size=args.chunk_size,
            nn_samples=args.nn_samples,
        )

    print(json.dumps(results, indent=2, sort_keys=True))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
