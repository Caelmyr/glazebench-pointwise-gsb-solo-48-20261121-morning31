"""End-to-end checks for the dual-exposure polarized glare feature.

Run with an isolated interpreter: python -I verify_glare.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glazebench.analysis import evaluate
from glazebench.demo import load_training
from glazebench.model import InputError
from glazebench.store import Store
from glazebench.geometry import specimen_frame
from glazebench.lighting import blend_parallel_exposures, measure_glare
from glazebench.parameters import TOOLS, analysis_parameters


def glare_payload(tile, *, darker=True, **overrides):
    photos = {p["role"]: p for p in tile["photos"]}
    options = {key: field["default"] for key, field in TOOLS["glare"]["fields"].items()}
    options.update(overrides)
    payload = {
        "version": tile["version"],
        "kind": "glare",
        "photo_id": photos["parallel"]["id"],
        "comparison_id": photos["cross"]["id"],
        "region_id": None,
        "options": options,
    }
    if darker:
        payload["options"]["darker_parallel_id"] = photos["parallel-dark"]["id"]
    return payload


def main():
    data_dir = Path(tempfile.mkdtemp(prefix="gb-verify-"))
    store = Store(data_dir)
    tiles, created = load_training(store)
    assert created, "expected a fresh training seed"
    tile = store.tile(tiles[0]["id"])
    assert len(tile["photos"]) == 6, len(tile["photos"])
    roles = [p["role"] for p in tile["photos"]]
    assert "parallel-dark" in roles, roles

    # Baseline: single-exposure glare still works exactly as before.
    single = evaluate(store, tile["id"], glare_payload(tile, darker=False))
    assert single["metrics"]["dark_exposure"] is None
    assert single["pictures"]["darker"] is None
    assert single["provenance"]["darker_parallel_photo"] is None
    base_unknown = np.asarray(Image.open(store.blobs.require(single["pictures"]["overlay"])).convert("RGB"))

    # Dual exposure: dark frame recovers some of the bright clipped hotspot.
    merged = evaluate(store, tile["id"], glare_payload(tile))
    m = merged["metrics"]
    dx = m["dark_exposure"]
    print("dark exposure metrics:", dx)
    assert dx is not None
    assert dx["recovered_pixels"] > 0, "darker exposure should recover clipped hotspot pixels"
    assert dx["recovered_percent_of_measured"] > 0
    assert dx["normalization_gain"] > 1.0
    assert merged["pictures"]["darker"] is not None
    assert merged["provenance"]["darker_parallel_photo"]["role"] == "parallel-dark"
    # Some pixels must remain unmeasured (clipped in both or outside dark frame).
    assert dx["still_clipped_pixels"] > 0, "the saturated core should clip in both exposures"
    assert dx["outside_darker_frame_pixels"] > 0, "the inset darker frame leaves an unsupported rim"
    cross_frame = specimen_frame(store.blobs, tile, next(p["id"] for p in tile["photos"] if p["role"] == "cross"))
    dark_frame_check = specimen_frame(store.blobs, tile, next(p["id"] for p in tile["photos"] if p["role"] == "parallel-dark"))
    primary_frame_check = specimen_frame(store.blobs, tile, next(p["id"] for p in tile["photos"] if p["role"] == "parallel"))
    plane_area = np.ones(cross_frame.extent[::-1], dtype=bool)
    rim_in_common = int(np.count_nonzero(plane_area & cross_frame.valid & ~dark_frame_check.valid))
    both_clipped = int(np.count_nonzero(
        plane_area & primary_frame_check.valid & dark_frame_check.valid
        & primary_frame_check.clipped & dark_frame_check.clipped & cross_frame.valid
    ))
    # Pixels saturated in both exposures are the newly retained unmeasured core.
    assert both_clipped == dx["still_clipped_pixels"]
    assert m["support"]["unmeasured_after_blend_pixels"] == both_clipped
    assert rim_in_common > 0
    # Recovered pixels sit inside the dark frame and were clipped in the primary.
    from glazebench.lighting import blend_parallel_exposures
    _, recovered_mask, _, _, _ = blend_parallel_exposures(primary_frame_check, dark_frame_check, plane_area)
    assert not np.any(recovered_mask & ~dark_frame_check.valid)
    assert np.all(primary_frame_check.clipped[recovered_mask])

    # The merged run records recovery against the single-exposure support counters.
    single_measured = single["metrics"]["support"]["unclipped_pixels"]
    assert m["support"]["recovered_pixels"] == dx["recovered_pixels"]
    print("single unclipped:", single_measured, "recovered:", dx["recovered_pixels"])

    # Outline pixels (amber ~ (245,199,107)) must be present on the merged overlay only.
    merged_img = np.asarray(Image.open(store.blobs.require(merged["pictures"]["overlay"])).convert("RGB"))
    def amber_count(img):
        return int(np.count_nonzero((img[:, :, 0] > 225) & (img[:, :, 1] > 160) & (img[:, :, 1] < 225) & (img[:, :, 2] < 150)))
    assert amber_count(merged_img) > 0, "recovered-region outline missing"
    assert amber_count(base_unknown) == 0, "single-exposure overlay must not carry an outline"
    print("amber outline px:", amber_count(merged_img))

    # Caching: same payload returns the stored record.
    again = evaluate(store, tile["id"], glare_payload(tile))
    assert again["id"] == merged["id"]

    # Reject selecting the main parallel photo as its own darker capture.
    bad = glare_payload(tile)
    bad["options"]["darker_parallel_id"] = bad["photo_id"]
    try:
        evaluate(store, tile["id"], bad)
    except InputError as error:
        print("self dark rejected:", error)
    else:
        raise AssertionError("same-photo darker selection was accepted")

    # Reject a cross-polarized photo in the darker slot.
    bad = glare_payload(tile)
    bad["options"]["darker_parallel_id"] = bad["comparison_id"]
    try:
        evaluate(store, tile["id"], bad)
    except InputError as error:
        print("wrong-role dark rejected:", error)
    else:
        raise AssertionError("cross photo accepted as darker parallel")

    # Omitting the field entirely stays on the standard two-photo comparison.
    payload = glare_payload(tile)
    payload["options"].pop("darker_parallel_id")
    omitted = evaluate(store, tile["id"], payload)
    assert omitted["metrics"]["dark_exposure"] is None

    # With two parallel-dark captures, an explicit choice still works and is respected.
    tile = store.tile(tile["id"])
    source_dark = next(p for p in tile["photos"] if p["role"] == "parallel-dark")
    clone = {**source_dark, "id": "111111111111111111111111", "source_digest": source_dark["source_digest"][::-1]}

    def add_clone(doc):
        doc["photos"].append(clone)
        doc["registrations"][clone["id"]] = dict(doc["registrations"][source_dark["id"]])

    tile2 = store.edit_tile(tile["id"], tile["version"], add_clone, "capture", "test duplicate dark")
    payload_original = glare_payload(tile2)
    payload_original["options"]["darker_parallel_id"] = source_dark["id"]
    explicit_two = evaluate(store, tile2["id"], payload_original)
    assert explicit_two["metrics"]["dark_exposure"]["capture_id"] == source_dark["id"]
    payload_clone = glare_payload(tile2)
    payload_clone["options"]["darker_parallel_id"] = clone["id"]
    picked_clone = evaluate(store, tile2["id"], payload_clone)
    assert picked_clone["metrics"]["dark_exposure"]["capture_id"] == clone["id"]
    tile = store.tile(tile["id"])

    # Direct blending checks: hard switch, no averaging seam; both-clipped stays excluded.
    direct_frame = specimen_frame(store.blobs, tile, next(p["id"] for p in tile["photos"] if p["role"] == "parallel"))
    dark_frame = specimen_frame(store.blobs, tile, next(p["id"] for p in tile["photos"] if p["role"] == "parallel-dark"))
    area = np.ones(direct_frame.extent[::-1], dtype=bool)
    merged_luma, recovered, still_clipped, gain, method = blend_parallel_exposures(direct_frame, dark_frame, area)
    assert gain > 1.0 and method["estimated"]
    assert np.count_nonzero(recovered) > 0
    assert np.count_nonzero(still_clipped) > 0
    # Recovered values must come verbatim from the scaled dark frame (no interpolation).
    from glazebench.photos import luminance
    scaled_dark = luminance(dark_frame.linear) * gain
    assert np.allclose(merged_luma[recovered], scaled_dark[recovered], atol=1e-6)
    # Outside the recovered switch the merged field is exactly the primary luminance.
    primary_luma = luminance(direct_frame.linear)
    untouched = area & ~recovered & direct_frame.valid
    assert np.array_equal(merged_luma[untouched], primary_luma[untouched])
    print("blend gain:", round(gain, 4), "recovered:", int(recovered.sum()), "still clipped:", int(still_clipped.sum()))

    # measure_glare contract with a dark frame returns five values and a recovered mask.
    cross_frame = specimen_frame(store.blobs, tile, next(p["id"] for p in tile["photos"] if p["role"] == "cross"))
    result, percent, valid, unknown, rec_mask = measure_glare(
        direct_frame, cross_frame, area, area,
        analysis_parameters({"version": 1, "kind": "glare", "options": {}})["options"],
        dark_frame,
    )
    assert rec_mask.shape == area.shape
    assert bool(np.any(rec_mask))
    assert not np.any(valid & unknown), "valid and unknown masks must be disjoint"
    # Pixels clipped in both parallel frames cannot be valid.
    assert not np.any(valid & still_clipped)
    print("glare strong %:", result["strong_glare_percent"], "hotspots:", result["hotspot_count"])

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
