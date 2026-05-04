"""Phase 4 tests — the HTTP surface via qh.

Uses qh's TestClient (no real server bound) so the tests are fast and
hermetic. The HTTP server's stores are replaced with in-memory ones via
`reset_stores` so the user's app data folder is never touched.

The whole module is skipped when the `[http]` extras (qh / fastapi /
uvicorn) aren't installed.
"""

from __future__ import annotations

import importlib.util
import io
import os

import pytest

# Module-level skip if any of the [http] extras are missing — the import
# of `lookbook.http` would already cascade-fail otherwise.
pytest.importorskip("qh", reason="qh not installed (lookbook[http])")
pytest.importorskip("fastapi", reason="fastapi not installed (lookbook[http])")

from PIL import Image

from lookbook import get_stores
from lookbook.http import mk_lookbook_app, reset_stores


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def memory_server(tmp_path):
    """Server-wide stores swapped for in-memory; auto-restored after test."""
    # Use tmp_path-backed stores for the `images` slot only — that one
    # needs filesystem so paths can roundtrip — but keep manifest/runs
    # in dict to keep tests fast.
    stores = get_stores(
        images_store={},
        manifest_store={},
        runs_store={},
        embeddings={},
    )
    reset_stores(stores)
    yield stores
    reset_stores(None)


@pytest.fixture
def image_dir(tmp_path):
    """A directory with 6 distinct synthetic PNGs."""
    import numpy as np

    rng = np.random.default_rng(42)
    for i in range(6):
        arr = rng.uniform(0, 255, (256, 256, 3)).astype(np.uint8)
        Image.fromarray(arr).save(tmp_path / f"img_{i}.png")
    return str(tmp_path)


@pytest.fixture
def client(memory_server):
    """qh TestClient against the lookbook app."""
    from qh.testing import run_app

    app = mk_lookbook_app()
    with run_app(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def test_list_recipes(client):
    r = client.post("/list_recipes", json={})
    assert r.status_code == 200
    body = r.json()
    assert "recipes" in body
    # The shipped recipes include random / funnel / diverse_mock + person profiles.
    assert "random" in body["recipes"]
    assert "funnel" in body["recipes"]
    assert "person_mock" in body["recipes"]


def test_list_plugins(client):
    r = client.post("/list_plugins", json={})
    assert r.status_code == 200
    body = r.json()
    assert "random_score" in body["scorers"]
    assert "top_k" in body["selectors"]
    assert "facility_location" in body["selectors"]
    assert "min_resolution" in body["filters"]
    assert "mock" in body["embedders"]


# ---------------------------------------------------------------------------
# Ingest + score
# ---------------------------------------------------------------------------


def test_ingest_source(client, image_dir):
    r = client.post("/ingest_source", json={"source_path": image_dir})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 6
    assert len(body["image_ids"]) == 6


def test_score_image_by_path(client, image_dir):
    paths = sorted(
        os.path.join(image_dir, fn) for fn in os.listdir(image_dir)
        if fn.endswith(".png")
    )
    r = client.post("/score_image", json={
        "source_path": paths[0],
        "metric_id": "blur",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["metric_id"] == "blur"
    assert isinstance(body["value"], (int, float))


def test_score_image_by_id_after_ingest(client, image_dir, memory_server):
    client.post("/ingest_source", json={"source_path": image_dir})
    image_ids = list(memory_server.images.keys())
    assert image_ids
    r = client.post("/score_image", json={
        "image_id": image_ids[0],
        "metric_id": "blur",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["image_id"] == image_ids[0]
    assert isinstance(body["value"], (int, float))


# ---------------------------------------------------------------------------
# Curate end-to-end
# ---------------------------------------------------------------------------


def test_curate_source_funnel(client, image_dir):
    r = client.post("/curate_source", json={
        "source_path": image_dir,
        "k": 3,
        "recipe": "funnel",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["report"]["n_candidates"] == 6
    # All 6 are 256x256 — default min_long_side=1024 drops them all.
    # That's fine; the test verifies the full pipeline ran.
    assert "kept" in body
    assert "run_id" in body


def test_curate_source_funnel_relaxed(client, image_dir):
    r = client.post("/curate_source", json={
        "source_path": image_dir,
        "k": 3,
        "recipe": "funnel_relaxed",
    })
    assert r.status_code == 200
    body = r.json()
    # funnel_relaxed has min_long_side=512 but our images are 256x256 —
    # still drops them all, but via a known path.
    assert body["report"]["n_candidates"] == 6


def test_curate_source_random(client, image_dir):
    """Random recipe has no filters, so all candidates survive to selection."""
    r = client.post("/curate_source", json={
        "source_path": image_dir,
        "k": 3,
        "recipe": "random",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["report"]["n_candidates"] == 6
    assert body["report"]["n_kept"] == 3


def test_curate_unknown_recipe_returns_400(client, image_dir):
    r = client.post("/curate_source", json={
        "source_path": image_dir,
        "k": 3,
        "recipe": "no_such_recipe",
    })
    # Whether qh maps the ValueError to 400 or 500 depends on its handler;
    # either way the request must not silently succeed.
    assert r.status_code != 200


# ---------------------------------------------------------------------------
# Annotations + runs
# ---------------------------------------------------------------------------


def test_get_annotations(client, image_dir, memory_server):
    client.post("/curate_source", json={
        "source_path": image_dir,
        "k": 3,
        "recipe": "random",
    })
    image_ids = list(memory_server.images.keys())
    r = client.post("/get_annotations", json={"image_id": image_ids[0]})
    assert r.status_code == 200
    body = r.json()
    assert body["image_id"] == image_ids[0]
    metric_ids = {a["metric_id"] for a in body["annotations"]}
    assert "random_score" in metric_ids


def test_list_runs_and_get_run(client, image_dir):
    r1 = client.post("/curate_source", json={
        "source_path": image_dir,
        "k": 3,
        "recipe": "random",
    })
    run_id = r1.json()["run_id"]

    r2 = client.post("/list_runs", json={})
    assert r2.status_code == 200
    runs = r2.json()["runs"]
    assert run_id in runs

    r3 = client.post("/get_run", json={"run_id": run_id})
    assert r3.status_code == 200
    assert r3.json()["run_id"] == run_id


def test_get_run_unknown_returns_error(client):
    r = client.post("/get_run", json={"run_id": "definitely-not-a-run"})
    assert r.status_code != 200


# ---------------------------------------------------------------------------
# Image lookup
# ---------------------------------------------------------------------------


def test_get_image_metadata(client, image_dir, memory_server):
    client.post("/ingest_source", json={"source_path": image_dir})
    image_ids = list(memory_server.images.keys())
    r = client.post("/get_image", json={"image_id": image_ids[0]})
    assert r.status_code == 200
    body = r.json()
    assert body["image_id"] == image_ids[0]
    assert body["n_bytes"] > 0
    assert os.path.isfile(body["path"])


def test_get_image_unknown_id(client):
    r = client.post("/get_image", json={"image_id": "no_such_id"})
    assert r.status_code == 200  # We return a JSON error, not HTTP error.
    assert "error" in r.json()
