import numpy as np
import pytest

from ept.serialization import PERTURBATIONS, serialize
from ept.serialization.curves import hilbert_index_2d, zorder_index_2d
from ept.serialization.orderings import helix_phi0
from ept.synthetic import make_event


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_hilbert_is_a_continuous_curve(bits):
    n = 1 << bits
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x, y = x.ravel(), y.ravel()
    d = hilbert_index_2d(x, y, bits)
    assert sorted(d) == list(range(n * n))
    order = np.argsort(d)
    steps = np.abs(np.diff(x[order])) + np.abs(np.diff(y[order]))
    assert np.all(steps == 1)


def test_zorder_is_a_bijection():
    n = 16
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    assert sorted(zorder_index_2d(x.ravel(), y.ravel(), 4)) == list(range(n * n))


@pytest.mark.parametrize(
    "spec,n_orders",
    [("random", 1), ("phi", 1), ("hilbert", 1), ("zorder", 1), ("lsh:num_hashes=3", 3), ("helix:num_bins=4", 4)],
)
def test_specs_return_permutations(spec, n_orders):
    event = make_event(n_tracks=50, n_noise=20)
    perms = serialize(event, spec)
    assert len(perms) == n_orders
    for p in perms:
        assert sorted(p) == list(range(len(event["phi"])))


def test_single_bin_helix_equals_hilbert():
    event = make_event(n_tracks=200, n_noise=50)
    assert np.array_equal(serialize(event, "helix:num_bins=1")[0], serialize(event, "hilbert")[0])


def test_helix_phi0_undoes_the_bend():
    event = make_event(n_tracks=100, n_noise=0, seed=3)
    kappa = event["q"] * 0.3 * 2.0 / event["pt"] / 1000.0
    phi0 = np.array([helix_phi0(event["phi"][i : i + 1], event["r"][i : i + 1], k)[0] for i, k in enumerate(kappa)])
    # Every hit of a track maps back to the same starting azimuth.
    for pid in np.unique(event["pid"]):
        vals = phi0[event["pid"] == pid]
        spread = np.angle(np.exp(1j * (vals - vals[0])))
        assert np.max(np.abs(spread)) < 1e-9


@pytest.mark.parametrize("name", list(PERTURBATIONS))
def test_perturbations_keep_permutation(name):
    perm = np.random.default_rng(0).permutation(1000)
    out = PERTURBATIONS[name](perm, block_size=64)
    assert sorted(out) == list(range(1000))


def test_within_block_shuffle_keeps_block_membership():
    perm = np.arange(256)
    out = PERTURBATIONS["within_block_shuffle"](perm, block_size=64)
    assert all(set(out[i : i + 64]) == set(perm[i : i + 64]) for i in range(0, 256, 64))
