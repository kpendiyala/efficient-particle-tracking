"""Phase 0: measure how well each serialization keeps track hits together.

No training. For every event and ordering spec, computes adjacency and
co-blocking metrics (see ``ept/diagnostics/adjacency.py``), split by pT bin.

Examples:
    # Smoke test on toy events, no data needed
    python scripts/phase0_diagnostics.py --synthetic 3 --out runs/phase0-toy

    # Real TrackML, pixel only, first 20 events of train_1
    python scripts/phase0_diagnostics.py --data-dir /data/raw/trackml/train_1 \
        --detector pixel --events 20 --out /data/runs/phase0/pixel

Spec syntax: ``ordering[:k=v,...][|perturbation[:k=v,...]]``, e.g.
``helix:num_bins=8`` or ``hilbert|within_block_shuffle:block_size=128``.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ept.data.trackml import list_events, load_event, select_tracks  # noqa: E402
from ept.diagnostics.adjacency import evaluate, summarize  # noqa: E402
from ept.serialization import PERTURBATIONS, parse_kwargs, serialize  # noqa: E402
from ept.synthetic import PIXEL_RADII_MM, STRIP_RADII_MM, make_event  # noqa: E402

DEFAULT_SPECS = [
    "random",
    "phi",
    "zorder",
    "hilbert",
    "lsh:num_hashes=3",
    "lsh:num_hashes=8",  # same number of orders as helix:num_bins=8, for a fair "best" comparison
    "lsh:num_hashes=24",  # HEPTv2: 8 heads x 3 hashes per layer
    "helix:num_bins=2",
    "helix:num_bins=4",
    "helix:num_bins=8",
    "hilbert|reverse",
    "hilbert|within_block_shuffle:block_size=128",
    "hilbert|block_shuffle:block_size=128",
]
PT_BINS = [0.9, 1.5, 3.0, 10.0, np.inf]


def apply_spec(event: dict, spec: str) -> list[np.ndarray]:
    base, _, pert = spec.partition("|")
    perms = serialize(event, base)
    if pert:
        name, _, args = pert.partition(":")
        if name not in PERTURBATIONS:
            raise ValueError(f"Unknown perturbation '{name}'. Known: {', '.join(PERTURBATIONS)}")
        perms = [PERTURBATIONS[name](p, **parse_kwargs(args)) for p in perms]
    return perms


def run_event(task: tuple) -> list[dict]:
    source, name, detector, specs, pt_min = task
    if source == "synthetic":
        radii = PIXEL_RADII_MM if detector == "pixel" else PIXEL_RADII_MM + STRIP_RADII_MM
        event = make_event(n_tracks=2000, n_noise=2000, radii=radii, seed=int(name))
    else:
        event = load_event(source, name, detector)
    selected = select_tracks(event, pt_min=pt_min)
    track_pt = pd.Series(event["pt"][selected], index=event["pid"][selected]).groupby(level=0).first()
    pt_bin = pd.cut(track_pt, PT_BINS, right=False).astype(str)

    rows = []
    for spec in specs:
        t0 = time.perf_counter()
        perms = apply_spec(event, spec)
        serialize_ms = (time.perf_counter() - t0) * 1000
        pairs, tracks = evaluate(perms, event["pid"], event["path"], selected)
        pairs["pt_bin"] = pt_bin.reindex(pairs["pid"]).to_numpy()
        tracks["pt_bin"] = pt_bin.reindex(tracks.index).to_numpy()
        groups = [("all", pairs, tracks)] + [
            (b, pairs[pairs.pt_bin == b], tracks[tracks.pt_bin == b]) for b in sorted(pt_bin.unique())
        ]
        for label, p, t in groups:
            if len(t) == 0:
                continue
            rows.append(
                {
                    "event": name,
                    "spec": spec,
                    "pt_bin": label,
                    "n_hits": len(event["pid"]),
                    "n_orders": len(perms),
                    "serialize_ms": serialize_ms,
                    **summarize(p, t),
                }
            )
    return rows


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal Markdown table (avoids a tabulate dependency)."""
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="Directory of raw TrackML event CSVs (e.g. train_1)")
    src.add_argument("--synthetic", type=int, help="Use N toy events instead of real data")
    ap.add_argument("--detector", choices=["pixel", "full"], default="pixel")
    ap.add_argument("--events", type=int, default=10, help="Max number of events to process")
    ap.add_argument("--specs", nargs="+", default=DEFAULT_SPECS)
    ap.add_argument("--pt-min", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.synthetic:
        tasks = [("synthetic", str(i), args.detector, args.specs, args.pt_min) for i in range(args.synthetic)]
    else:
        names = list_events(args.data_dir)[: args.events]
        if not names:
            sys.exit(f"No TrackML events found in {args.data_dir}")
        tasks = [(args.data_dir, n, args.detector, args.specs, args.pt_min) for n in names]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for event_rows in pool.map(run_event, tasks) for r in event_rows]

    per_event = pd.DataFrame(rows)
    per_event.to_csv(out / "per_event.csv", index=False)
    metric_cols = [c for c in per_event.columns if c not in ("event", "spec", "pt_bin")]
    summary = per_event.groupby(["pt_bin", "spec"], sort=False)[metric_cols].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    headline = [
        "pair_gap_median_best",
        "pair_within_8_best",
        "pair_within_8_single",
        "track_one_block_B128_best",
        "track_one_block_B1024_single",
        "track_fragments_w16_single",
    ]
    with open(out / "summary.md", "w") as f:
        for label, grp in summary.groupby("pt_bin", sort=False):
            f.write(f"\n### pT bin: {label}\n\n")
            f.write(to_markdown(grp[["spec", "n_orders", *headline]]))
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
