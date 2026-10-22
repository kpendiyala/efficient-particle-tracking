"""Hit serializations: each function maps an event to a list of permutations.

An event is a dict of 1D arrays with at least ``eta``, ``phi`` (rad) and ``r``
(transverse radius, mm). A permutation ``perm`` lists hit indices in sequence
order, so ``perm[0]`` is the first hit the model sees. Multi-order schemes
(LSH with several hashes, helix with several curvature bins) return one
permutation per order, mirroring how HEPTv2 gives each head its own order.
"""
from typing import Callable

import numpy as np

from .curves import hilbert_index_2d, quantize, zorder_index_2d

# TrackML solenoid field (T). Curvature conversion: R[m] = pT[GeV] / (0.3 * B[T]).
B_FIELD_T = 2.0

Event = dict[str, np.ndarray]
Ordering = Callable[..., list[np.ndarray]]


def _norm(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / max(v.std(), 1e-12)


def phi_sort(event: Event, **_) -> list[np.ndarray]:
    """Plain azimuthal ordering (HEPTv2's "phi" ablation)."""
    return [np.argsort(event["phi"], kind="stable")]


def eta_phi_curve(event: Event, curve: str = "hilbert", bits: int = 10, phi=None, **_) -> list[np.ndarray]:
    """Space-filling curve over (eta, phi), as in PTv3 / HEPTv2's Z-H ablation."""
    phi = event["phi"] if phi is None else phi
    gx = quantize(event["eta"], bits, -4.0, 4.0)
    gy = quantize(phi, bits, -np.pi, np.pi)
    fn = hilbert_index_2d if curve == "hilbert" else zorder_index_2d
    return [np.argsort(fn(gx, gy, bits), kind="stable")]


def lsh(event: Event, num_hashes: int = 3, num_regions: int = 100, seed: int = 0, **_) -> list[np.ndarray]:
    """OR-AND E2LSH ordering in (eta, phi), approximating HEPTv2's encoder.

    Per hash: an AND step partitions hits into a coarse (eta, phi) quantile
    grid with a random aspect ratio and roughly ``num_regions`` cells; inside
    each cell, hits are sorted by a random 1D projection. Several hashes form
    the OR step. This follows ``heptv2/model/attention.py`` in spirit but is not
    bit-exact; swap in their code if exact parity is needed.
    """
    rng = np.random.default_rng(seed)
    eta, phi = _norm(event["eta"]), _norm(event["phi"])
    eta_rank = np.argsort(np.argsort(eta, kind="stable"), kind="stable") / len(eta)
    phi_rank = np.argsort(np.argsort(phi, kind="stable"), kind="stable") / len(phi)
    perms = []
    for _ in range(num_hashes):
        aspect = np.exp(rng.uniform(-1.0, 1.0))
        n_eta = max(1, int(round(np.sqrt(num_regions * aspect))))
        n_phi = max(1, int(round(num_regions / n_eta)))
        region = np.minimum((eta_rank * n_eta).astype(np.int64), n_eta - 1) * n_phi + np.minimum(
            (phi_rank * n_phi).astype(np.int64), n_phi - 1
        )
        a = rng.normal(size=2)
        proj = a[0] * eta + a[1] * phi
        perms.append(np.lexsort((proj, region)))
    return perms


def helix_phi0(phi: np.ndarray, r: np.ndarray, kappa: float) -> np.ndarray:
    """Azimuth at the origin for a track of signed curvature ``kappa`` (1/mm).

    A track from the beamline with radius R reaches radius r at
    phi(r) = phi0 - q * asin(r / (2R)), so undoing the bend gives phi0.
    """
    shifted = phi + np.arcsin(np.clip(kappa * r / 2.0, -1.0, 1.0))
    return np.mod(shifted + np.pi, 2 * np.pi) - np.pi


def curvature_bins(num_bins: int, pt_min_gev: float = 0.9, b_field_t: float = B_FIELD_T) -> np.ndarray:
    """Signed curvatures (1/mm) evenly spanning tracks with pT >= pt_min."""
    if num_bins == 1:
        return np.zeros(1)
    kappa_max = 0.3 * b_field_t / pt_min_gev / 1000.0
    return np.linspace(-kappa_max, kappa_max, num_bins)


def helix(event: Event, num_bins: int = 4, pt_min_gev: float = 0.9, curve: str = "hilbert", bits: int = 10, **_) -> list[np.ndarray]:
    """Track-aligned orders: undo the bend for each curvature bin, then serialize (eta, phi0).

    With ``num_bins=1`` the single bin is kappa=0 and this reduces exactly to
    ``eta_phi_curve`` on (eta, phi), so the curve baseline is the helix ablation's
    control.
    """
    perms = []
    for kappa in curvature_bins(num_bins, pt_min_gev):
        phi0 = helix_phi0(event["phi"], event["r"], kappa)
        perms += eta_phi_curve(event, curve=curve, bits=bits, phi=phi0)
    return perms


def random_order(event: Event, seed: int = 0, **_) -> list[np.ndarray]:
    return [np.random.default_rng(seed).permutation(len(event["phi"]))]


ORDERINGS: dict[str, Ordering] = {
    "phi": phi_sort,
    "hilbert": lambda e, **kw: eta_phi_curve(e, curve="hilbert", **kw),
    "zorder": lambda e, **kw: eta_phi_curve(e, curve="zorder", **kw),
    "lsh": lsh,
    "helix": helix,
    "random": random_order,
}


def parse_kwargs(args: str) -> dict:
    """Parse ``key=value,...`` with ints and floats converted."""
    kwargs = {}
    for item in filter(None, args.split(",")):
        key, _, value = item.partition("=")
        kwargs[key] = float(value) if "." in value else int(value) if value.lstrip("-").isdigit() else value
    return kwargs


def parse_spec(spec: str) -> tuple[str, dict]:
    """Parse ``name[:key=value,...]``, e.g. ``helix:num_bins=8`` or ``lsh:num_hashes=8``."""
    name, _, args = spec.partition(":")
    kwargs = parse_kwargs(args)
    if name not in ORDERINGS:
        raise ValueError(f"Unknown ordering '{name}'. Known: {', '.join(ORDERINGS)}")
    return name, kwargs


def serialize(event: Event, spec: str) -> list[np.ndarray]:
    name, kwargs = parse_spec(spec)
    return ORDERINGS[name](event, **kwargs)
