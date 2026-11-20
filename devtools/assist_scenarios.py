"""Scenario checks: repetitive stripes and half-captured photos.

Run from the repository root with: python3 -I devtools/assist_scenarios.py
"""
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.environ.get("POINTWISE_SITE", "/opt/pointwise-env/site-packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from glazebench.assist import suggest_registration
from glazebench.demo import load_training
from glazebench.geometry import DEFAULT_CORNERS, apply_homography, homography
from glazebench.photos import png_bytes, store_capture
from glazebench.store import Store, make_id


def stripe_image(width, height, period, phase=0.0):
    y, x = np.mgrid[0:height, 0:width]
    bands = (0.5 + 0.42 * np.sin(2 * np.pi * x / period + phase)).astype(np.float32)
    noise = (np.random.default_rng(3).normal(0, 0.01, size=(height, width))).astype(np.float32)
    gray = np.clip(bands + noise, 0, 1)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8))


def main():
    workdir = Path(tempfile.mkdtemp(prefix="gb-scenario-"))
    store = Store(workdir)
    tiles, _ = load_training(store)
    tile = store.tile(tiles[0]["id"])

    # --- Scenario A: repetitive stripes, shifted by roughly one period -----
    width = height = 520
    period = 26
    base_img = stripe_image(width, height, period)
    shifted_img = stripe_image(width, height, period, phase=2 * np.pi * 4 / period)
    base_record = store_capture(store.blobs, png_bytes(base_img), "diffuse", "stripe-base.png", "", make_id())
    side_record = store_capture(store.blobs, png_bytes(shifted_img), "rake-left", "stripe-side.png", "", make_id())
    tile["photos"].extend([base_record, side_record])
    tile["plane"] = {"photo_id": base_record["id"], "corners": DEFAULT_CORNERS, "size": [520, 520]}
    tile["registrations"] = {}
    pairs = [
        {"base": [0.2, 0.25], "side": [0.2, 0.25]},
        {"base": [0.45, 0.3], "side": [0.45, 0.3]},
        {"base": [0.7, 0.55], "side": [0.7, 0.55]},
        {"base": [0.35, 0.75], "side": [0.35, 0.75]},
        {"base": [0.8, 0.2], "side": [0.8, 0.2]},
    ]
    result = suggest_registration(store, tile, side_record["id"], {"pairs": pairs})
    print("A. stripe checks:", result["checks"]["distinctiveness"])
    print("A. warnings:", *result["warnings"], sep="\n  - ")
    assert any("repeated stripes" in w or "interchangeable" in w for w in result["warnings"]), result["warnings"]

    # --- Scenario B: half-captured side photo (right side missing) ----------
    tile_b = store.tile(tiles[1]["id"])
    rake_b = next(p for p in tile_b["photos"] if p["role"] == "rake-right")
    with Image.open(store.blobs.require(rake_b["working_digest"])) as source:
        full = source.convert("RGBA")
    w, h = full.size
    # Keep the original canvas and erase the right portion, so normalized
    # coordinates are unchanged but the photo genuinely lacks that support.
    alpha = np.asarray(full.getchannel("A")).copy()
    alpha[:, int(w * 0.62):] = 0
    full.putalpha(Image.fromarray(alpha))
    half_record = store_capture(
        store.blobs, png_bytes(full), "rake-left", "half-capture.png", "", make_id(),
    )
    tile_b["photos"].append(half_record)
    true_corners = tile_b["registrations"][rake_b["id"]]["corners"]
    mapping = homography(DEFAULT_CORNERS, true_corners)
    # Only pick pairs whose true side location lies inside the visible area.
    base_pts = [[0.18, 0.2], [0.3, 0.5], [0.22, 0.78], [0.45, 0.35], [0.5, 0.7], [0.15, 0.55], [0.1, 0.3], [0.35, 0.85]]
    side_pts = apply_homography(mapping, np.asarray(base_pts))
    visible = (side_pts >= 0).all(axis=1) & (side_pts[:, 0] <= 0.6) & (side_pts[:, 1] <= 1.0)
    base_pts = [point for point, keep in zip(base_pts, visible) if keep]
    side_pts = side_pts[visible]
    assert len(base_pts) >= 4, (len(base_pts), side_pts.tolist())
    payload = {"pairs": [{"base": b, "side": s.tolist()} for b, s in zip(base_pts, side_pts)]}
    result_b = suggest_registration(store, tile_b, half_record["id"], payload)
    print("B. support:", result_b["checks"]["side_support_percent"], "%")
    print("B. common support:", result_b["checks"]["common_support_percent"], "%")
    print("B. off-frame corners:", result_b["checks"]["off_frame_corners"])
    print("B. warnings:", *result_b["warnings"], sep="\n  - ")
    assert result_b["checks"]["side_support_percent"] < 75, "erased area must reduce warped support"

    # The suggested quadrilateral still has to be savable through the relaxed
    # registration validator and produce a warped frame with partial support.
    from glazebench.geometry import (
        registration_corners_parameter,
        specimen_frame,
        warp_capture,
    )
    from glazebench.photos import open_working

    validated = registration_corners_parameter(result_b["corners"])
    assert len(validated) == 4
    tile_b["registrations"][half_record["id"]] = {
        "base_photo_id": tile_b["plane"]["photo_id"],
        "corners": result_b["corners"],
    }
    frame = specimen_frame(store.blobs, tile_b, half_record["id"])
    support = float(frame.valid.mean())
    print("B. warped support fraction:", round(support, 3))
    assert 0.05 < support < 0.8
    print("SCENARIO TESTS PASSED")


if __name__ == "__main__":
    main()
