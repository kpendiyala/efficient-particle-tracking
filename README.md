# efficient-particle-tracking
Geometry-aware hit serialization and linear-time sequence models for particle tracking

## Where this stands

**Phase 0** asks, with no training: *how well does each hit ordering keep a track's
hits together, and in which sense?* The answer decides whether track-aligned
(helix) serialization is worth building into a model.

Two kinds of token mixer need two different things from an ordering:

| Mixer | Needs | Metric (in `ept/diagnostics/adjacency.py`) |
|---|---|---|
| Recurrent (Mamba-2, Gated DeltaNet) | the next hit on a track to be *close in sequence* | gap between physically consecutive hits: `pair_gap_median`, `pair_within_{1,8,64,512}` |
| Block-local attention (HEPT, HEPTv2) | a track's hits to land in the *same block* | `track_one_block_B{128,1024}`, `track_fragments_w16` |

For schemes with several orders (LSH hashes, helix curvature bins), `_best` takes the best
order per pair/track and `_single` averages over orders. **Compare `_best` only at equal
order counts** (for example `lsh:num_hashes=8` vs `helix:num_bins=8`), since more orders
always help `_best`.

### Orderings (`ept/serialization/`)

| Spec | What it is |
|---|---|
| `random` | Uniform shuffle (floor) |
| `phi` | Sort by azimuth (HEPTv2's φ ablation) |
| `hilbert`, `zorder` | Space-filling curve over (η, φ) (PTv3, HEPTv2's Z-H ablation) |
| `lsh:num_hashes=K` | OR-AND E2LSH in (η, φ), approximating HEPTv2's encoder (not bit-exact) |
| `helix:num_bins=K` | For K signed curvatures covering p<sub>T</sub> ≥ 0.9 GeV, undo the bend (φ₀ = φ + asin(κr/2)), then Hilbert over (η, φ₀). `num_bins=1` is exactly `hilbert`. |

Perturbations from the original order-sensitivity RQ compose with any of these:
`hilbert|reverse`, `hilbert|within_block_shuffle:block_size=128`, `hilbert|block_shuffle:block_size=128`.

## Running phase 0

Locally, on toy helices (no data needed):

```bash
pip install -e ".[dev]"
pytest -q
python scripts/phase0_diagnostics.py --synthetic 3 --detector full --out runs/phase0-toy
```

On Nautilus (namespace `cms-ml`, PVC `track-ssm-vol` mounted at `/data`):

1. Accept the [TrackML competition rules](https://www.kaggle.com/competitions/trackml-particle-identification/rules) on Kaggle, generate a token at [kaggle.com/settings/api](https://www.kaggle.com/settings/api), then create the secret:
   `kubectl -n cms-ml create secret generic kaggle-credentials --from-literal=KAGGLE_API_TOKEN=...`
   (A legacy `kaggle.json` also works: `--from-file=kaggle.json=$HOME/.kaggle/kaggle.json`.)
2. Fetch data: `kubectl apply -f k8s/fetch-trackml-job.yaml` (100-event `train_sample` plus detector geometry → `/data/raw/trackml/`). Check the unzipped folder name in the logs. The diagnostics Job expects `train_100_events`.
3. If this repo is private, create a read-only token secret: `kubectl -n cms-ml create secret generic github-token --from-literal=GITHUB_TOKEN=...`
4. Run: `kubectl apply -f k8s/phase0-diagnostics-job.yaml` → `/data/runs/phase0/{pixel,full}/summary.md`
5. Copy results out with `k8s/pvc-shell.yaml` (instructions in the file).

`k8s/gpu-train-job.template.yaml` is the starting point for phase 1 GPU jobs (HEPTv2 reproduction).

## Go / no-go for helix ordering

Helix ordering earns a place as a method if, compared with `lsh:num_hashes=8` and `hilbert`, it clearly lowers
`pair_gap_median` / raises `pair_within_8` for p<sub>T</sub> < 1.5 GeV tracks, in either the pixel or full detector.
Back-of-envelope: at 0.9 GeV in 2 T, a track bends about 0.06 rad across the pixel detector but about
0.35 rad out to the strips, so expect a small effect in `pixel` and a larger one in `full`. The toy
events in `tests/test_adjacency.py` reproduce that direction. Real data has to confirm it.

## Layout

```
ept/serialization/   orderings, space-filling curves, perturbations
ept/diagnostics/     adjacency and co-blocking metrics
ept/data/trackml.py  raw TrackML CSV loader (pixel or full detector)
ept/synthetic.py     toy helical events for tests
scripts/             phase0_diagnostics.py
k8s/                 data fetch Job, phase 0 Job, GPU Job template, PVC shell
```
