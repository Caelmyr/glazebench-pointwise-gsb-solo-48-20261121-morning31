"""HTTP-level smoke test for the dual-exposure glare workflow."""
import json
import tempfile
from pathlib import Path

from glazebench.demo import load_training
from glazebench.store import Store
from glazebench.web import create_app


def main():
    data_dir = Path(tempfile.mkdtemp(prefix="gb-http-"))
    store = Store(data_dir)
    tiles, _ = load_training(store)
    app = create_app(str(data_dir))
    client = app.test_client()

    config = client.get("/api/configuration").get_json()
    field = config["tools"]["glare"]["fields"]["darker_parallel_id"]
    assert field["kind"] == "capture" and field["default"] is None
    assert "parallel-dark" in config["photo_roles"]

    tile = client.get(f"/api/specimens/{tiles[0]['id']}").get_json()
    roles = {p["role"]: p for p in tile["photos"]}
    dark_id = roles["parallel-dark"]["id"]

    payload = {
        "version": tile["version"],
        "kind": "glare",
        "photo_id": roles["parallel"]["id"],
        "comparison_id": roles["cross"]["id"],
        "region_id": None,
        "options": {
            "balance": "auto",
            "fixed_gain": 1,
            "glare_fraction": 0.2,
            "minimum_hotspot": 30,
            "balance_region": None,
            "darker_parallel_id": dark_id,
        },
    }
    response = client.post(f"/api/specimens/{tile['id']}/results", json=payload)
    assert response.status_code == 201, response.get_json()
    record = response.get_json()
    assert record["metrics"]["dark_exposure"]["capture_id"] == dark_id
    assert record["pictures"]["darker"]
    assert record["provenance"]["darker_parallel_photo"]["role"] == "parallel-dark"
    # JSON round-trips cleanly (store already serialized it once).
    json.loads(json.dumps(record, allow_nan=False))

    for layer in ("overlay", "source", "paired", "darker"):
        image = client.get(f"/api/results/{record['id']}/image/{layer}")
        assert image.status_code == 200 and image.data[:8] == b"\x89PNG\r\n\x1a\n", layer

    # Blank field keeps the classic two-photo comparison.
    payload["options"]["darker_parallel_id"] = None
    classic = client.post(f"/api/specimens/{tile['id']}/results", json=payload).get_json()
    assert classic["metrics"]["dark_exposure"] is None
    assert classic["pictures"]["darker"] is None

    # Invalid capture id shape is a 422.
    payload["options"]["darker_parallel_id"] = "not-an-id"
    bad = client.post(f"/api/specimens/{tile['id']}/results", json=payload)
    assert bad.status_code == 422

    print("HTTP SMOKE PASSED")


if __name__ == "__main__":
    main()
