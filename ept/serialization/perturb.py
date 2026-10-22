"""Perturbations of a base order (the original order-sensitivity RQ).

Each takes a permutation and returns a new one, so they compose with any
ordering in ``orderings.py``.
"""
import numpy as np


def reverse(perm: np.ndarray, **_) -> np.ndarray:
    return perm[::-1].copy()


def within_block_shuffle(perm: np.ndarray, block_size: int = 128, seed: int = 0, **_) -> np.ndarray:
    """Shuffle hits inside each block; block membership is unchanged."""
    rng = np.random.default_rng(seed)
    out = perm.copy()
    for start in range(0, len(out), block_size):
        rng.shuffle(out[start : start + block_size])
    return out


def block_shuffle(perm: np.ndarray, block_size: int = 128, seed: int = 0, **_) -> np.ndarray:
    """Shuffle the order of blocks; order inside each block is unchanged."""
    rng = np.random.default_rng(seed)
    blocks = [perm[i : i + block_size] for i in range(0, len(perm), block_size)]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


PERTURBATIONS = {
    "reverse": reverse,
    "within_block_shuffle": within_block_shuffle,
    "block_shuffle": block_shuffle,
}
