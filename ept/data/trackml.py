"""Read raw TrackML events (Kaggle CSVs) into flat numpy arrays.

Each event ``eventXXXXXXXXX`` has ``-hits``, ``-truth`` and ``-particles`` CSVs
(optionally gzipped). Coordinates are in mm and momenta in GeV.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# TrackML volume ids: 7/8/9 pixel (endcap-, barrel, endcap+),
# 12/13/14 short strips, 16/17/18 long strips.
PIXEL_VOLUMES = (7, 8, 9)


def list_events(data_dir: str | Path) -> list[str]:
    data_dir = Path(data_dir)
    names = {p.name.split("-")[0] for p in data_dir.glob("event*-hits.csv*")}
    return sorted(names)


def _read(data_dir: Path, event: str, kind: str) -> pd.DataFrame:
    for suffix in (".csv", ".csv.gz"):
        path = data_dir / f"{event}-{kind}{suffix}"
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"No {kind} file for {event} in {data_dir}")


def calc_eta(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Pseudorapidity from hit position, as in HEPT's tracking.py."""
    theta = np.arctan2(r, z)
    return -np.log(np.tan(theta / 2.0))


def load_event(data_dir: str | Path, event: str, detector: str = "pixel") -> dict[str, np.ndarray]:
    """Load one event. ``detector`` is ``pixel`` or ``full`` (pixel + strips).

    Returned arrays are per hit: ``x, y, z, r, phi, eta, path`` (distance from
    origin, used to order hits along a track), ``volume, layer``, ``pid``
    (0 = noise), and the hit's particle ``pt``, ``q`` and particle ``eta``.
    """
    data_dir = Path(data_dir)
    hits = _read(data_dir, event, "hits")
    truth = _read(data_dir, event, "truth")[["hit_id", "particle_id"]]
    particles = _read(data_dir, event, "particles")
    hits = hits.merge(truth, on="hit_id", how="left")
    if detector == "pixel":
        hits = hits[hits.volume_id.isin(PIXEL_VOLUMES)]
    elif detector != "full":
        raise ValueError("detector must be 'pixel' or 'full'")

    particles = particles.assign(
        pt=np.hypot(particles.px, particles.py),
        p_eta=np.arcsinh(particles.pz / np.hypot(particles.px, particles.py).clip(lower=1e-9)),
    )[["particle_id", "pt", "q", "p_eta"]]
    hits = hits.merge(particles, on="particle_id", how="left").fillna({"pt": 0.0, "q": 0, "p_eta": 0.0})

    x, y, z = (hits[c].to_numpy(np.float64) for c in ("x", "y", "z"))
    r = np.hypot(x, y)
    return {
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "phi": np.arctan2(y, x),
        "eta": calc_eta(r, z),
        "path": np.sqrt(r**2 + z**2),
        "volume": hits.volume_id.to_numpy(),
        "layer": hits.layer_id.to_numpy(),
        "pid": hits.particle_id.fillna(0).to_numpy(np.int64),
        "pt": hits.pt.to_numpy(np.float64),
        "q": hits.q.to_numpy(np.int64),
        "p_eta": hits.p_eta.to_numpy(np.float64),
    }


def select_tracks(event: dict[str, np.ndarray], pt_min: float = 0.9, eta_max: float = 4.0, min_hits: int = 3) -> np.ndarray:
    """Hits of target tracks: HEPTv2's pT > 0.9 GeV, |eta| < 4, with >= min_hits hits here."""
    pid = event["pid"]
    ok = (pid != 0) & (event["pt"] > pt_min) & (np.abs(event["p_eta"]) < eta_max)
    counts = pd.Series(pid[ok]).value_counts()
    keep = counts.index[counts >= min_hits].to_numpy()
    return ok & np.isin(pid, keep)
