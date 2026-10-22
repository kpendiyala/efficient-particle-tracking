"""HEPTv2's encoder serialization, ported from PyTorch to NumPy.

Follows ``heptv2/utils/hash_utils.py`` (``get_regions``, ``E2LSH``),
``heptv2/data/preprocess.py`` (``build_region_indices``) and
``HEPTAttention.forward`` in ``heptv2/model/attention.py``
(github.com/Graph-COM/HEPTv2, commit of 2026-06-16).

Per layer HEPTv2 draws ``num_hashes * num_heads`` orders. For each order:

* AND step: hits get an eta region and a phi region by equal-count quantile
  bins, with per-order bin counts drawn by ``get_regions``.
* OR step: a random projection ``alpha . (eta, phi)``.

The sort key is ``projection + (eta_region + phi_region * (ceil(n_eta) + 1)) *
span(projection)``, so hits sort by (phi region, eta region) cell first and by
projection inside a cell. Given the same region table and projection
weights, this produces exactly HEPTv2's permutations; ``scripts/check_heptv2_parity.py``
checks that against their torch code. Only the random number stream differs
(NumPy instead of torch).
"""
import numpy as np


def get_regions(num_regions: int, num_or_hashes: int, num_heads: int, rng: np.random.Generator, num_and_hashes: int = 2) -> np.ndarray:
    """Per-order (eta, phi) region counts, shape (2, num_or_hashes * num_heads), ordered (c h)."""
    lb = 2
    ub = 2 * num_regions ** (1 / num_and_hashes) - lb
    regions = rng.uniform(size=(num_or_hashes * num_heads, num_and_hashes)) * (ub - lb) + lb
    regions = (num_regions / regions.prod(axis=1, keepdims=True)) ** (1 / num_and_hashes) * regions
    regions = np.round(regions * 3) / 3
    # torch: rearrange(regions, "(h c) a -> c a h"), then preprocess: "c a h -> a (c h)".
    regions = regions.reshape(num_heads, num_or_hashes, num_and_hashes).transpose(1, 2, 0)
    return regions.transpose(1, 0, 2).reshape(num_and_hashes, num_or_hashes * num_heads)


def quantile_partition(values: np.ndarray, num_regions: np.ndarray) -> np.ndarray:
    """1-based equal-count region index of each hit, one row per entry of ``num_regions``."""
    n = len(values)
    rank = np.empty(n, dtype=np.int64)
    rank[np.argsort(values, kind="stable")] = np.arange(n)
    region_size = np.ceil(n / num_regions)[:, None]
    return rank[None, :] // region_size + 1


def sort_keys(coords2: np.ndarray, regions_h: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """HEPTv2's per-order sort keys, shape (num_hashes * num_heads, n).

    ``coords2`` is (n, 2) as (eta, phi), ``regions_h`` is (2, c*h) from
    ``get_regions``, and ``alpha`` is the E2LSH projection, (num_heads, 2, num_hashes).
    """
    num_heads, _, num_hashes = alpha.shape
    region_eta = quantile_partition(coords2[:, 0], regions_h[0])
    region_phi = quantile_partition(coords2[:, 1], regions_h[1])
    hashed = np.einsum("nd,hdc->chn", coords2, alpha).reshape(num_hashes * num_heads, -1)
    hash_shift = hashed.max(axis=1, keepdims=True) - hashed.min(axis=1, keepdims=True)
    shifts = region_eta * hash_shift + region_phi * hash_shift * (np.ceil(regions_h[0])[:, None] + 1)
    return hashed + shifts


def heptv2_lsh(event, num_hashes: int = 3, num_heads: int = 8, num_regions: int = 100, seed: int = 0, **_) -> list[np.ndarray]:
    """One HEPTv2 encoder layer's orders: ``num_hashes * num_heads`` permutations.

    Defaults match HEPTv2's config (3 hashes x 8 heads = 24 orders per layer).
    """
    rng = np.random.default_rng(seed)
    coords2 = np.stack([event["eta"], event["phi"]], axis=1)
    regions_h = get_regions(num_regions, num_hashes, num_heads, rng)
    alpha = rng.normal(size=(num_heads, 2, num_hashes))
    return list(np.argsort(sort_keys(coords2, regions_h, alpha), axis=1, kind="stable"))
