"""Regression: every measurement mode still runs on the seeded training tiles."""
import tempfile
from pathlib import Path

from glazebench.analysis import evaluate
from glazebench.demo import load_training
from glazebench.parameters import TOOLS
from glazebench.store import Store

import sys
sys.path.insert(0, str(Path(__file__).parent))
from verify_glare import glare_payload


def main():
    store = Store(Path(tempfile.mkdtemp(prefix="gb-reg-")))
    tiles, _ = load_training(store)
    tile = store.tile(tiles[0]["id"])
    photos = {p["role"]: p for p in tile["photos"]}
    region = next(r for r in tile["regions"] if r["purpose"] == "inspection")

    def run(kind, **extra):
        options = {key: field["default"] for key, field in TOOLS[kind]["fields"].items()}
        payload = {
            "version": tile["version"], "kind": kind,
            "photo_id": None, "comparison_id": None,
            "region_id": region["id"], "options": options,
        }
        payload.update(extra)
        return evaluate(store, tile["id"], payload)

    single_tools = ["color", "defects", "texture", "craze", "rim", "profile"]
    for kind in single_tools:
        record = run(kind)
        assert record["pictures"]["overlay"], kind

    run("coverage", region_id=None, options={
        **{key: field["default"] for key, field in TOOLS["coverage"]["fields"].items()},
        "bare_region": next(r["id"] for r in tile["regions"] if r["purpose"] == "bare"),
        "glazed_region": next(r["id"] for r in tile["regions"] if r["purpose"] == "glazed"),
    })

    direction = run(
        "direction",
        photo_id=photos["rake-left"]["id"],
        comparison_id=photos["rake-right"]["id"],
        region_id=region["id"],
    )
    assert direction["metrics"]["pair"]["comparison_role"] == "rake-right"

    # Glare with explicit pair selection, single and dual exposure.
    single = evaluate(store, tile["id"], glare_payload(tile, darker=False))
    dual = evaluate(store, tile["id"], glare_payload(tile))
    assert single["metrics"]["dark_exposure"] is None
    assert dual["metrics"]["dark_exposure"]["recovered_pixels"] > 0

    # Fixed-gain balance still works on the dual-exposure path.
    fixed = evaluate(store, tile["id"], glare_payload(tile, balance="fixed", fixed_gain=1.2))
    assert fixed["metrics"]["balance"]["estimated"] is False
    assert fixed["metrics"]["dark_exposure"] is not None

    # Study comparison across two tiles still accepts the glare metric.
    from glazebench.study import create_study
    other = store.tile(tiles[1]["id"])
    other_dual = evaluate(store, other["id"], glare_payload(other))
    study = create_study(store, {
        "title": "Glare study", "group_by": "glaze", "metric": "glare",
        "result_ids": [dual["id"], other_dual["id"]],
    })
    assert len(study["rows"]) == 2

    print("REGRESSION PASSED")


if __name__ == "__main__":
    main()
