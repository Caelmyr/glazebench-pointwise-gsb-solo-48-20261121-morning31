"""Live-server curl-style flow for the assist feature."""
import json
import os
import sys
import urllib.request

BASE = "http://127.0.0.1:19971"


def call(method, path, body=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request) as response:
        raw = response.read()
        return response.status, (json.loads(raw) if raw else None)


def main():
    _, tiles = call("GET", "/api/specimens")
    tile_id = tiles["items"][0]["id"]
    _, tile = call("GET", f"/api/specimens/{tile_id}")
    base_id = tile["plane"]["photo_id"]
    rake = next(p for p in tile["photos"] if p["role"] == "rake-left")
    true_corners = tile["registrations"][rake["id"]]["corners"]

    # Build pairs by hand from normalized coordinates (mimics UI clicks).
    base_pts = [[0.21, 0.26], [0.54, 0.2], [0.78, 0.47], [0.33, 0.72], [0.68, 0.8]]
    # Use the known pose mapping only to place the side clicks correctly.
    from glazebench.geometry import DEFAULT_CORNERS, apply_homography, homography
    mapping = homography(DEFAULT_CORNERS, true_corners)
    side_pts = apply_homography(mapping, base_pts).tolist()
    payload = {"pairs": [{"base": b, "side": s} for b, s in zip(base_pts, side_pts)]}

    status, body = call("POST", f"/api/specimens/{tile_id}/registrations/{rake['id']}/assist", payload)
    assert status == 200
    assert body["evidence_png"].startswith("data:image/png;base64,")
    print("assist warnings:", body["warnings"])
    print("checks:", json.dumps(body["checks"]))

    status, saved = call("PUT", f"/api/specimens/{tile_id}/registrations/{rake['id']}", {
        "version": tile["version"], "corners": body["corners"],
    })
    assert status == 200
    assert saved["registrations"][rake["id"]]["corners"] == body["corners"]
    print("saved support:", saved["registrations"][rake["id"]]["support_percent"])

    # Viewer + compare still work.
    import urllib.request as u
    view = u.urlopen(f"{BASE}/api/specimens/{tile_id}/viewer/{rake['id']}")
    assert view.status == 200 and view.headers["Content-Type"] == "image/png"
    print("viewer bytes:", len(view.read()))
    print("LIVE FLOW PASSED")


if __name__ == "__main__":
    sys.path.insert(0, os.environ.get("POINTWISE_SITE", "/opt/pointwise-env/site-packages"))
    sys.path.insert(0, "/workspace")
    main()
