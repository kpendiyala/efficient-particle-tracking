"""Compare ept's NumPy port of HEPTv2 LSH ordering with HEPTv2's own torch functions.

Not run in CI (needs torch, einops and a HEPTv2 checkout):
    git clone --depth 1 https://github.com/Graph-COM/HEPTv2 /tmp/heptv2
    pip install torch einops
    python scripts/check_heptv2_parity.py /tmp/heptv2 .

Pulls the exact function sources out of the HEPTv2 checkout (no package import,
which would need torch_scatter), feeds both sides the same regions and alpha,
and compares the resulting permutations.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

HEPTV2, EPT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(EPT))
from ept.serialization.heptv2_lsh import get_regions as np_get_regions, sort_keys  # noqa: E402
from ept.synthetic import PIXEL_RADII_MM, STRIP_RADII_MM, make_event  # noqa: E402


def pull(path, names):
    tree = ast.parse(path.read_text())
    return "\n\n".join(ast.get_source_segment(path.read_text(), n) for n in tree.body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)


ns = {"torch": torch, "rearrange": rearrange, "nn": torch.nn, "os": __import__("os")}
exec(pull(HEPTV2 / "heptv2/utils/hash_utils.py", {"get_regions", "E2LSH", "uniform"}), ns)
exec(pull(HEPTV2 / "heptv2/data/preprocess.py", {"_quantile_partition_from_sorted", "build_region_indices"}), ns)
src = pull(HEPTV2 / "heptv2/model/attention.py", {"lsh_coords", "get_geo_shift_single"}).replace("@torch.no_grad()", "")
ns["_env_bool"] = lambda *a, **k: False
exec(src, ns)

num_hashes, num_heads = 3, 8
worst_perm, worst_regions = 0.0, 0.0
for seed, radii in [(0, PIXEL_RADII_MM), (1, PIXEL_RADII_MM + STRIP_RADII_MM)]:
    ev = make_event(n_tracks=3000, n_noise=3000, radii=radii, seed=seed)
    coords = torch.tensor(np.stack([ev["eta"], ev["phi"]], 1), dtype=torch.float64)
    torch.manual_seed(seed)
    regions = ns["get_regions"](100, num_hashes, num_heads).double()  # (c, a, h)
    e2lsh = ns["E2LSH"](n_hashes=num_hashes, n_heads=num_heads, dim=2).double()
    valid = torch.ones(len(coords), dtype=torch.bool)
    region_indices, regions_h = ns["build_region_indices"](coords, valid, regions)
    hashed = ns["lsh_coords"](e2lsh, coords, num_heads)
    hash_shift = rearrange(hashed.max(-1, keepdim=True).values - hashed.min(-1, keepdim=True).values, "c h d -> (c h) d")
    shifts = ns["get_geo_shift_single"](regions_h, hash_shift, region_indices, num_hashes)
    ref = (hashed + shifts).argsort(dim=-1).reshape(num_hashes * num_heads, -1).numpy()

    ours = np.argsort(sort_keys(coords.numpy(), regions_h.numpy(), e2lsh.alpha.detach().numpy()), axis=1, kind="stable")
    mismatch = (ref != ours).mean()
    worst_perm = max(worst_perm, mismatch)
    print(f"seed {seed}: {len(coords)} hits, {ref.shape[0]} orders, fraction of positions that differ = {mismatch:.2e}")

    # get_regions layout: replay HEPTv2's torch.rand draws through the NumPy port.
    torch.manual_seed(seed)
    draws = np.array([[torch.rand(1).item() for _ in range(2)] for _ in range(num_hashes * num_heads)])

    class Replay:
        def uniform(self, size):
            assert size == draws.shape
            return draws

    ours_regions = np_get_regions(100, num_hashes, num_heads, Replay())
    regions_err = np.abs(ours_regions - regions_h.numpy()).max()
    print(f"seed {seed}: get_regions max abs difference = {regions_err:.2e}")
    worst_regions = max(worst_regions, regions_err)

# get_regions runs in float32 in torch, hence a ~1e-7 difference there.
ok = worst_perm == 0 and worst_regions < 1e-5
print("PARITY OK" if ok else f"MISMATCH (permutations {worst_perm:.2e}, regions {worst_regions:.2e})")
sys.exit(0 if ok else 1)
