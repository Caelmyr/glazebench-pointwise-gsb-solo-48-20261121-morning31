"""End-to-end HTTP checks for the assist endpoint and compatibility.

Run from the repository root with: python3 -I devtools/assist_http.py
"""
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.environ.get("POINTWISE_SITE", "/opt/pointwise-env/site-packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glazebench.demo import load_training
from glazebench.geometry import DEFAULT_CORNERS, apply_homography, homography
from glazebench.store import Store
from glazebench.web import create_app


def main():
    workdir = Path(tempfile.mkdtemp(prefix="gb-http-"))
    store = Store(workdir)
    tiles, _ = load_training(store)
    app = create_app(data_dir=str(workdir))
    client = app.test_client()

    tile = store.tile(tiles[0]["id"])
    rake = next(p for p in tile["photos"] if p["role"] == "rake-left")
    true_corners = tile["registrations"][rake["id"]]["corners"]
    mapping = homography(DEFAULT_CORNERS, true_corners)
    base_pts = [[0.2, 0.25], [0.55, 0.18], [0.8, 0.45], [0.35, 0.7], [0.7, 0.82]]
    side_pts = apply_homography(mapping, base_pts)
    payload = {"pairs": [{"base": b, "side": s.tolist()} for b, s in zip(base_pts, side_pts)]}

    response = client.post(
        f"/api/specimens/{tile['id']}/registrations/{rake['id']}/assist", json=payload,
    )
    assert response.status_code == 200, response.get_json()
    body = response.get_json()
    assert set(body) >= {"corners", "pairs", "residuals_px", "warnings", "checks", "evidence_png"}
    assert body["evidence_png"].startswith("data:image/png;base64,")
    assert len(body["corners"]) == 4 and len(body["pairs"]) == 5
    print("1. assist endpoint OK; warnings:", body["warnings"])

    # Validation error becomes the standard 422 envelope.
    bad = client.post(
        f"/api/specimens/{tile['id']}/registrations/{rake['id']}/assist",
        json={"pairs": payload["pairs"][:2]},
    )
    assert bad.status_code == 422 and "error" in bad.get_json()
    print("2. assist validation 422 OK:", bad.get_json()["error"])

    # Apply the suggested corners through the existing save endpoint.
    saved = client.put(
        f"/api/specimens/{tile['id']}/registrations/{rake['id']}",
        json={"version": tile["version"], "corners": body["corners"]},
    )
    assert saved.status_code == 200, saved.get_json()
    updated = saved.get_json()
    record = updated["registrations"][rake["id"]]
    assert record["corners"] == body["corners"]
    assert record["support_percent"] > 15
    print("3. save suggested corners OK; support:", record["support_percent"], "%")

    # The previously saved, hand-entered registrations on the other photos
    # must still load and be editable through the same endpoint.
    other = next(p for p in tile["photos"] if p["role"] == "cross")
    legacy = tile["registrations"][other["id"]]
    shown = client.get(f"/api/specimens/{tile['id']}").get_json()
    assert shown["registrations"][other["id"]]["corners"] == legacy["corners"]
    tweaked = [[c[0] + 0.004, c[1]] for c in legacy["corners"]]
    edit = client.put(
        f"/api/specimens/{tile['id']}/registrations/{other['id']}",
        json={"version": updated["version"], "corners": tweaked},
    )
    assert edit.status_code == 200, edit.get_json()
    assert edit.get_json()["registrations"][other["id"]]["corners"] == tweaked
    print("4. legacy hand-marked registration still loads and edits OK")

    # The registered viewer image still renders for both records.
    for photo_id in (rake["id"], other["id"]):
        view = client.get(f"/api/specimens/{tile['id']}/viewer/{photo_id}")
        assert view.status_code == 200 and view.mimetype == "image/png"
    print("5. registered viewer renders for assisted and legacy records OK")

    # Assisting the base photo is rejected.
    base_id = tile["plane"]["photo_id"]
    guard = client.post(
        f"/api/specimens/{tile['id']}/registrations/{base_id}/assist", json=payload,
    )
    assert guard.status_code == 422
    print("6. base-photo guard OK:", guard.get_json()["error"])
    print("HTTP TESTS PASSED")


if __name__ == "__main__":
    main()
