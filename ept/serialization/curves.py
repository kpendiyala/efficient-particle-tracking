"""Space-filling curve indices on a 2D integer grid (vectorized)."""
import numpy as np


def quantize(values: np.ndarray, bits: int, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Map values linearly onto integers in [0, 2**bits - 1]."""
    lo = values.min() if lo is None else lo
    hi = values.max() if hi is None else hi
    n = (1 << bits) - 1
    scaled = (values - lo) / max(hi - lo, 1e-12)
    return np.clip(np.floor(scaled * n + 0.5), 0, n).astype(np.int64)


def hilbert_index_2d(x: np.ndarray, y: np.ndarray, bits: int) -> np.ndarray:
    """Hilbert-curve distance for integer coordinates in [0, 2**bits)."""
    x = x.astype(np.int64).copy()
    y = y.astype(np.int64).copy()
    n = 1 << bits
    d = np.zeros_like(x)
    s = n >> 1
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx.astype(np.int64)) ^ ry.astype(np.int64))
        # Rotate the quadrant so the sub-curve keeps the right orientation.
        flip = (~ry) & rx
        x = np.where(flip, n - 1 - x, x)
        y = np.where(flip, n - 1 - y, y)
        swap = ~ry
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        s >>= 1
    return d


def zorder_index_2d(x: np.ndarray, y: np.ndarray, bits: int) -> np.ndarray:
    """Morton (Z-order) index: interleave the bits of x and y."""
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    d = np.zeros_like(x)
    for b in range(bits):
        d |= ((x >> b) & 1) << (2 * b)
        d |= ((y >> b) & 1) << (2 * b + 1)
    return d
