"""How well a serialization keeps each track's hits together.

Two different properties matter for two kinds of token mixer:

* **Adjacency** (recurrent mixers: Mamba, DeltaNet). A recurrence has to carry
  a hit's information to the next hit on the same track, so we measure the
  sequence distance between *physically consecutive* hits of a track (ordered
  by distance from the origin).
* **Co-blocking** (block-local attention: HEPT, HEPTv2). Inside a block every
  hit sees every other hit, so the order within the block is irrelevant; we
  measure how many blocks a track is spread over.

For multi-order schemes, ``best`` takes the best order per pair or track (what
a model with one order per head could in principle exploit) and ``single``
averages over orders (what one order gives on its own).
"""
import numpy as np
import pandas as pd

PAIR_WINDOWS = (1, 8, 64, 512)
BLOCK_SIZES = (128, 1024)
FRAGMENT_WINDOW = 16


def ranks_of(perm: np.ndarray) -> np.ndarray:
    """Inverse permutation: sequence position of each hit."""
    rank = np.empty_like(perm)
    rank[perm] = np.arange(len(perm))
    return rank


def track_pairs(pid: np.ndarray, path_key: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices (a, b) of physically consecutive hits on each selected track."""
    idx = np.flatnonzero(selected)
    order = idx[np.lexsort((path_key[idx], pid[idx]))]
    same = pid[order[1:]] == pid[order[:-1]]
    return order[:-1][same], order[1:][same]


def blocks_spanned(rank: np.ndarray, pid: np.ndarray, selected: np.ndarray, block_size: int) -> pd.Series:
    """Number of distinct blocks each selected track's hits fall into."""
    df = pd.DataFrame({"pid": pid[selected], "block": rank[selected] // block_size})
    return df.groupby("pid")["block"].nunique()


def fragments(rank: np.ndarray, pid: np.ndarray, selected: np.ndarray, window: int) -> pd.Series:
    """Runs a track splits into when consecutive sequence positions differ by > window."""
    idx = np.flatnonzero(selected)
    order = idx[np.lexsort((rank[idx], pid[idx]))]
    p, r = pid[order], rank[order]
    breaks = (p[1:] == p[:-1]) & (np.diff(r) > window)
    df = pd.DataFrame({"pid": p[1:], "brk": breaks})
    counts = df.groupby("pid")["brk"].sum()
    return counts.reindex(np.unique(p), fill_value=0) + 1


def evaluate(perms: list[np.ndarray], pid: np.ndarray, path_key: np.ndarray, selected: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-pair and per-track locality metrics for one serialization scheme.

    Returns ``(pairs, tracks)``. ``pairs`` has one row per consecutive hit pair
    with the best and mean sequence gap over orders. ``tracks`` is indexed by
    particle id with best/single blocks spanned and fragment counts.
    """
    a, b = track_pairs(pid, path_key, selected)
    ranks = [ranks_of(p) for p in perms]
    gaps = np.stack([np.abs(r[a] - r[b]) for r in ranks])
    pairs = pd.DataFrame({"pid": pid[a], "gap_best": gaps.min(0), "gap_mean": gaps.mean(0)})
    for w in PAIR_WINDOWS:
        pairs[f"within_{w}_single"] = (gaps <= w).mean(0)

    cols = {}
    for bs in BLOCK_SIZES:
        spans = pd.concat([blocks_spanned(r, pid, selected, bs) for r in ranks], axis=1)
        cols[f"blocks_B{bs}_best"] = spans.min(axis=1)
        cols[f"one_block_B{bs}_single"] = (spans == 1).mean(axis=1)
    frags = pd.concat([fragments(r, pid, selected, FRAGMENT_WINDOW) for r in ranks], axis=1)
    cols[f"fragments_w{FRAGMENT_WINDOW}_best"] = frags.min(axis=1)
    cols[f"fragments_w{FRAGMENT_WINDOW}_single"] = frags.mean(axis=1)
    tracks = pd.DataFrame(cols)
    tracks.index.name = "pid"
    return pairs, tracks


def summarize(pairs: pd.DataFrame, tracks: pd.DataFrame) -> dict[str, float]:
    """Collapse per-pair and per-track tables into one row of headline numbers."""
    out = {
        "n_tracks": float(len(tracks)),
        "n_pairs": float(len(pairs)),
        "pair_gap_median_best": float(pairs["gap_best"].median()),
        "pair_gap_median_single": float(pairs["gap_mean"].median()),
    }
    for w in PAIR_WINDOWS:
        out[f"pair_within_{w}_best"] = float((pairs["gap_best"] <= w).mean())
        out[f"pair_within_{w}_single"] = float(pairs[f"within_{w}_single"].mean())
    for bs in BLOCK_SIZES:
        out[f"track_one_block_B{bs}_best"] = float((tracks[f"blocks_B{bs}_best"] == 1).mean())
        out[f"track_one_block_B{bs}_single"] = float(tracks[f"one_block_B{bs}_single"].mean())
    out[f"track_fragments_w{FRAGMENT_WINDOW}_best"] = float(tracks[f"fragments_w{FRAGMENT_WINDOW}_best"].mean())
    out[f"track_fragments_w{FRAGMENT_WINDOW}_single"] = float(tracks[f"fragments_w{FRAGMENT_WINDOW}_single"].mean())
    return out
