"""Smoke test for point-pair assisted registration.

Run from the repository root with: python3 -I devtools/assist_smoke.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.environ.get("POINTWISE_SITE", "/opt/pointwise-env/site-packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from glazebench.assist import suggest_registration
from glazebench.demo import load_training
from glazebench.geometry import (
    DEFAULT_CORNERS,
    apply_homography,
    fit_homography,
    homography,
    registration_corners_parameter,
)
from glazebench.model import InputError
from glazebench.photos import png_bytes
from glazebench.store import Store


def approx_equal(a, b, tol=0.01):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


def main():
    workdir = Path(tempfile.mkdtemp(prefix="gb-assist-"))
    store = Store(workdir)
    tiles, created = load_training(store)
    assert created, "training tiles should seed"
    tile = store.tile(tiles[0]["id"])
    base = next(p for p in tile["photos"] if p["id"] == tile["plane"]["photo_id"])
    rake = next(p for p in tile["photos"] if p["role"] == "rake-left")
    true_corners = tile["registrations"][rake["id"]]["corners"]

    # 1. Homography recovery from point pairs sampled on the known mapping.
    base_to_side = homography(DEFAULT_CORNERS, true_corners)
    rng = np.random.default_rng(7)
    base_pts = [[0.2, 0.25], [0.55, 0.18], [0.8, 0.45], [0.35, 0.7], [0.7, 0.82], [0.15, 0.55]]
    side_pts = apply_homography(base_to_side, np.asarray(base_pts)).tolist()
    fitted = fit_homography(np.asarray(base_pts), np.asarray(side_pts))
    recovered = apply_homography(fitted, np.asarray(DEFAULT_CORNERS))
    assert approx_equal(recovered, true_corners, tol=1e-6), "fit must recover exact homography from exact pairs"
    print("1. exact homography recovery OK")

    # 2. Full suggest pipeline with slightly noisy human-like clicks.
    noisy_side = (np.asarray(side_pts) + rng.normal(0, 0.0025, size=np.asarray(side_pts).shape)).tolist()
    payload = {"pairs": [{"base": b, "side": s} for b, s in zip(base_pts, noisy_side)]}
    result = suggest_registration(store, tile, rake["id"], payload)
    assert len(result["corners"]) == 4
    assert approx_equal(result["corners"], true_corners, tol=0.02), (result["corners"], true_corners)
    assert result["checks"]["pair_count"] == 6
    assert result["evidence"].size[0] > 100
    print("2. noisy-pair suggestion OK; residuals px:", result["residuals_px"])
    print("   checks:", json.dumps(result["checks"], indent=None))
    print("   warnings:", result["warnings"])

    # 3. A deliberately wrong pair must trigger a residual warning.
    bad = [dict(item) for item in payload["pairs"]]
    bad[2] = {"base": base_pts[2], "side": [min(0.98, noisy_side[2][0] + 0.06), noisy_side[2][1]]}
    result_bad = suggest_registration(store, tile, rake["id"], {"pairs": bad})
    assert any("does not fit" in w for w in result_bad["warnings"]), result_bad["warnings"]
    print("3. outlier pair warning OK")

    # 4. Clustered pairs must trigger the spread/extrapolation warning.
    clustered_b = [[0.42, 0.44], [0.46, 0.47], [0.5, 0.43], [0.48, 0.52]]
    clustered_s = apply_homography(base_to_side, np.asarray(clustered_b)).tolist()
    result_cluster = suggest_registration(
        store, tile, rake["id"],
        {"pairs": [{"base": b, "side": s} for b, s in zip(clustered_b, clustered_s)]},
    )
    assert any("clustered" in w for w in result_cluster["warnings"]), result_cluster["warnings"]
    print("4. clustered-pair warning OK")

    # 5. Off-frame corner: shift the side image truth so the right-hand
    # corners land outside the unit frame, then verify relaxed validation.
    shifted = [[c[0] + 0.25, c[1]] for c in true_corners]
    validated = registration_corners_parameter(shifted)
    assert validated[1][0] > 1.0 and validated[2][0] > 1.0, validated
    try:
        registration_corners_parameter([[0.2, 0.2], [0.2, 0.8], [0.8, 0.8], [0.8, 0.2]])
    except InputError:
        pass
    else:
        raise AssertionError("reversed-order quad should be rejected")
    try:
        registration_corners_parameter([[0.1, 0.1], [0.9, 0.1], [0.5, 0.5], [0.9, 0.9]])
    except InputError:
        pass
    else:
        raise AssertionError("folded/concave quad should be rejected")
    print("5. relaxed corner validation (off-frame accepted, folded rejected) OK")

    # 6. Input validation paths.
    for bad_payload, phrase in [
        ({"pairs": payload["pairs"][:3]}, "between 4 and 12"),
        ({"pairs": [payload["pairs"][0], {**payload["pairs"][0]}, *payload["pairs"][1:]]}, "duplicated"),
        ({}, "between 4 and 12"),
    ]:
        try:
            suggest_registration(store, tile, rake["id"], bad_payload)
        except InputError as error:
            assert phrase in str(error), (phrase, error)
        else:
            raise AssertionError(f"expected error containing {phrase!r}")
    print("6. pair input validation OK")

    # 7. Base photo cannot be assisted; missing tile plane errors.
    try:
        suggest_registration(store, tile, base["id"], payload)
    except InputError as error:
        assert "another capture" in str(error)
    else:
        raise AssertionError("base photo assist must fail")
    print("7. base-photo guard OK")

    out = workdir / "evidence.png"
    out.write_bytes(png_bytes(result["evidence"]))
    shutil.rmtree(workdir)
    print("ALL ASSIST SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
