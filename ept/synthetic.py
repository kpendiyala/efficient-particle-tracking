"""Toy events: helical tracks from the beamline crossing barrel cylinders.

Used by the tests and for sanity checks before real data is on the PVC. Same
output keys as ``ept.data.trackml.load_event``.
"""
import numpy as np

from .serialization.orderings import B_FIELD_T

PIXEL_RADII_MM = (32.0, 72.0, 116.0, 172.0)
STRIP_RADII_MM = (260.0, 360.0, 500.0, 660.0, 820.0, 1020.0)


def make_event(
    n_tracks: int = 500,
    n_noise: int = 500,
    radii=PIXEL_RADII_MM,
    pt_range=(0.9, 5.0),
    eta_max: float = 2.5,
    z0_sigma_mm: float = 50.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    radii = np.asarray(radii)
    # pT roughly falling like 1/pT^2 between the bounds.
    u = rng.uniform(size=n_tracks)
    lo, hi = pt_range
    pt = 1.0 / (1.0 / lo - u * (1.0 / lo - 1.0 / hi))
    q = rng.choice([-1, 1], size=n_tracks)
    phi0 = rng.uniform(-np.pi, np.pi, size=n_tracks)
    eta = rng.uniform(-eta_max, eta_max, size=n_tracks)
    z0 = rng.normal(0.0, z0_sigma_mm, size=n_tracks)
    radius_mm = pt / (0.3 * B_FIELD_T) * 1000.0

    tr, rr = np.meshgrid(np.arange(n_tracks), radii, indexing="ij")
    tr, rr = tr.ravel(), rr.ravel()
    ok = rr < 2 * radius_mm[tr]
    tr, rr = tr[ok], rr[ok]
    half_angle = np.arcsin(rr / (2 * radius_mm[tr]))
    phi = np.mod(phi0[tr] - q[tr] * half_angle + np.pi, 2 * np.pi) - np.pi
    z = z0[tr] + 2 * radius_mm[tr] * half_angle * np.sinh(eta[tr])

    nr = rng.choice(radii, size=n_noise)
    nphi = rng.uniform(-np.pi, np.pi, size=n_noise)
    nz = rng.uniform(z.min(), z.max(), size=n_noise)

    r = np.concatenate([rr, nr])
    phi = np.concatenate([phi, nphi])
    z = np.concatenate([z, nz])
    pid = np.concatenate([tr + 1, np.zeros(n_noise, dtype=np.int64)])
    theta = np.arctan2(r, z)
    return {
        "x": r * np.cos(phi),
        "y": r * np.sin(phi),
        "z": z,
        "r": r,
        "phi": phi,
        "eta": -np.log(np.tan(theta / 2.0)),
        "path": np.sqrt(r**2 + z**2),
        "pid": pid,
        "pt": np.concatenate([pt[tr], np.zeros(n_noise)]),
        "q": np.concatenate([q[tr], np.zeros(n_noise, dtype=np.int64)]),
        "p_eta": np.concatenate([eta[tr], np.zeros(n_noise)]),
    }
