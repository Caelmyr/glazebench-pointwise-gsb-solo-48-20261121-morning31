"""Evidence packet round-trip including the darker parallel layer."""
import io
import tempfile
import zipfile
from pathlib import Path

from glazebench.analysis import evaluate
from glazebench.demo import load_training
from glazebench.evidence import create_evidence
from glazebench.intake import inspect_packet
from glazebench.store import Store
from glazebench.parameters import TOOLS

import sys
sys.path.insert(0, str(Path(__file__).parent))
from verify_glare import glare_payload


def main():
    store = Store(Path(tempfile.mkdtemp(prefix="gb-ev-")))
    tiles, _ = load_training(store)
    tile = store.tile(tiles[0]["id"])
    record = evaluate(store, tile["id"], glare_payload(tile))

    packet = create_evidence(store, tile["id"], {
        "result_ids": [record["id"]],
        "title": "Dual exposure packet",
        "include_sources": True,
        "view": "overlay",
    })
    raw = store_zip(store, packet)
    names = zipfile.ZipFile(io.BytesIO(raw)).namelist()
    assert any(name.endswith("-darker.png") for name in names), names
    assert any("originals/" in name for name in names)

    imported, created = inspect_packet(store, raw)
    assert created
    measurement = imported["measurements"][0]
    assert "darker" in measurement["pictures"]
    captured_roles = {item["role"] for item in imported["originals"]}
    assert "parallel-dark" in captured_roles, captured_roles
    print("EVIDENCE ROUND-TRIP PASSED:", sorted(names))


def store_zip(store, entry):
    from glazebench.evidence import evidence_zip
    return evidence_zip(store, entry)


if __name__ == "__main__":
    main()
