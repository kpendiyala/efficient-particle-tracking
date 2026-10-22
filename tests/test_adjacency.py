import numpy as np

from ept.data.trackml import select_tracks
from ept.diagnostics import evaluate, summarize
from ept.serialization import serialize
from ept.synthetic import PIXEL_RADII_MM, STRIP_RADII_MM, make_event


def test_perfect_order_scores_perfectly():
    event = make_event(n_tracks=300, n_noise=100)
    selected = select_tracks(event)
    # Oracle: group by track, then along the track.
    perm = np.lexsort((event["path"], event["pid"]))
    s = summarize(*evaluate([perm], event["pid"], event["path"], selected))
    assert s["pair_gap_median_best"] == 1
    assert s["pair_within_1_best"] == 1.0
    assert s["track_fragments_w16_best"] == 1.0


def test_random_order_is_worst():
    event = make_event(n_tracks=500, n_noise=500)
    selected = select_tracks(event)
    scores = {
        spec: summarize(*evaluate(serialize(event, spec), event["pid"], event["path"], selected))
        for spec in ("random", "phi", "hilbert")
    }
    assert scores["random"]["pair_within_8_single"] < scores["phi"]["pair_within_8_single"]
    assert scores["random"]["track_one_block_B128_single"] < scores["hilbert"]["track_one_block_B128_single"]


def test_helix_improves_adjacency_for_low_pt_tracks_over_long_radius():
    """The physics hypothesis in miniature: over strip radii, low-pT tracks bend
    enough that undoing the curvature brings hits closer in sequence."""
    event = make_event(
        n_tracks=3000, n_noise=1000, radii=PIXEL_RADII_MM + STRIP_RADII_MM, pt_range=(0.9, 1.5), seed=1
    )
    selected = select_tracks(event)
    gap = {
        spec: summarize(*evaluate(serialize(event, spec), event["pid"], event["path"], selected))[
            "pair_gap_median_best"
        ]
        for spec in ("hilbert", "helix:num_bins=8")
    }
    assert gap["helix:num_bins=8"] < gap["hilbert"]
