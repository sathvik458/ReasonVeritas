"""
src/balanced_sampling.py

Domain-balanced WeightedRandomSampler for joint LIAR + CoAID training.

The merged corpus is heavily skewed:
  - ~96% LIAR rows, ~4% CoAID rows
  - Within CoAID: real-class is ~10x more frequent than fake-class

A standard shuffled DataLoader sees almost no CoAID examples per batch,
so the model never has to learn anything CoAID-specific. This sampler
gives every (dataset_source, binary_label) bucket equal expected weight
per epoch, so each batch contains a healthy mix of LIAR-real, LIAR-fake,
CoAID-real, and CoAID-fake.

Usage in a trainer:

    from balanced_sampling import build_balanced_sampler
    sampler = build_balanced_sampler(has_meta_arr, labels_arr)
    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=B, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=B, shuffle=True)
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_balanced_sampler(
    has_meta: np.ndarray | None,
    labels: np.ndarray,
) -> WeightedRandomSampler | None:
    """
    Build a per-row sampler that balances on (has_meta, label).

    has_meta : (N,) array in {0, 1}; 1 = LIAR (full metadata), 0 = CoAID
    labels   : (N,) array of int class labels (0 = fake, 1 = real)

    Returns None if balancing isn't applicable (no has_meta variation, or
    only one bucket present after intersecting with labels).
    """
    if has_meta is None:
        return None
    has_meta = np.asarray(has_meta).astype(int)
    labels   = np.asarray(labels).astype(int)
    assert has_meta.shape == labels.shape, "has_meta and labels must align"

    # Bucket index combining domain (LIAR/CoAID) and class (fake/real).
    # 4 possible buckets: (1,0), (1,1), (0,0), (0,1).
    keys = np.stack([has_meta, labels], axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    if len(unique) < 2:
        return None

    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    # Weight per row = 1 / count_of_its_bucket. Then per-bucket expected
    # number of draws per epoch is constant.
    per_bucket_w = 1.0 / counts
    weights = per_bucket_w[inverse]
    weights = torch.from_numpy(weights).double()

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),  # one full pass equivalent
        replacement=True,
    )


def describe_buckets(
    has_meta: np.ndarray | None,
    labels: np.ndarray,
) -> str:
    """Human-readable breakdown of (domain, class) bucket sizes."""
    if has_meta is None:
        return "no has_meta — balancing disabled"
    has_meta = np.asarray(has_meta).astype(int)
    labels   = np.asarray(labels).astype(int)
    parts = []
    for hm, name in ((1, "liar"), (0, "coaid")):
        for cl, cname in ((0, "fake"), (1, "real")):
            n = int(((has_meta == hm) & (labels == cl)).sum())
            parts.append(f"{name}/{cname}={n}")
    return "  ".join(parts)
