"""Phase 5 tests — the MCP surface via fastmcp.

`fastmcp.Client(server)` runs the server in-memory over an internal
transport. We avoid `pytest-asyncio` (not in the dev env) by driving
each async tool call through `asyncio.run` inside a sync test.

The module is skipped when fastmcp or qh isn't installed (the MCP layer
depends on the HTTP layer's route functions).
"""

from __future__ import annotations

import asyncio
import io
import json
import os

import numpy as np
import pytest

# Module-level skip when the [mcp] / [http] extras are missing.
pytest.importorskip("fastmcp", reason="fastmcp not installed (lookbook[mcp])")
pytest.importorskip("qh", reason="qh not installed (lookbook[http])")

from PIL import Image

from lookbook import get_stores
from lookbook.http import reset_stores
from lookbook.mcp import mk_lookbook_mcp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_server():
    stores = get_stores(
        images_store={}, manifest_store={}, runs_store={}, embeddings={},
    )
    reset_stores(stores)
    yield stores
    reset_stores(None)


@pytest.fixture
def image_dir(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(6):
        arr = rng.uniform(0, 255, (256, 256, 3)).astype(np.uint8)
        Image.fromarray(arr).save(tmp_path / f"img_{i}.png")
    return str(tmp_path)


@pytest.fixture
def mcp_server(memory_server):
    return mk_lookbook_mcp()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_call(server, name: str, args: dict) -> dict:
    from fastmcp import Client

    async with Client(server) as client:
        result = await client.call_tool(name, args)
    if hasattr(result, "data") and result.data is not None:
        return result.data
    text = result.content[0].text
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return {"_raw": text}


async def _async_list_tools(server) -> list:
    from fastmcp import Client

    async with Client(server) as client:
        return await client.list_tools()


def call(server, name: str, args: dict) -> dict:
    return asyncio.run(_async_call(server, name, args))


def list_tools(server) -> list:
    return asyncio.run(_async_list_tools(server))


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_tool_registry(mcp_server):
    tools = list_tools(mcp_server)
    names = {t.name for t in tools}
    expected = {
        "list_recipes", "list_plugins", "ingest_source", "curate_source",
        "score_image", "get_annotations", "list_runs", "get_run", "get_image",
    }
    assert expected <= names, f"missing: {expected - names}"


def test_tool_descriptions_present(mcp_server):
    """Each tool must carry a description so the LLM knows when to call it."""
    tools = list_tools(mcp_server)
    for t in tools:
        assert t.description and len(t.description) > 20, (
            f"tool {t.name!r} has too short a description"
        )


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def test_list_recipes_tool(mcp_server):
    body = call(mcp_server, "list_recipes", {})
    assert "recipes" in body
    assert "funnel" in body["recipes"]
    assert "person_mock" in body["recipes"]


def test_list_plugins_tool(mcp_server):
    body = call(mcp_server, "list_plugins", {})
    assert "random_score" in body["scorers"]
    assert "facility_location" in body["selectors"]
    assert "min_resolution" in body["filters"]


# ---------------------------------------------------------------------------
# Ingest + score
# ---------------------------------------------------------------------------


def test_ingest_source_tool(mcp_server, image_dir):
    body = call(mcp_server, "ingest_source", {"source_path": image_dir})
    assert body["n"] == 6
    assert len(body["image_ids"]) == 6


def test_score_image_by_path(mcp_server, image_dir):
    paths = sorted(
        os.path.join(image_dir, fn) for fn in os.listdir(image_dir)
        if fn.endswith(".png")
    )
    body = call(mcp_server, "score_image", {
        "metric_id": "blur",
        "source_path": paths[0],
    })
    assert body["metric_id"] == "blur"
    assert isinstance(body["value"], (int, float))


def test_score_image_by_id(mcp_server, image_dir, memory_server):
    call(mcp_server, "ingest_source", {"source_path": image_dir})
    image_ids = list(memory_server.images.keys())
    body = call(mcp_server, "score_image", {
        "metric_id": "blur",
        "image_id": image_ids[0],
    })
    assert body["image_id"] == image_ids[0]
    assert isinstance(body["value"], (int, float))


# ---------------------------------------------------------------------------
# Curate
# ---------------------------------------------------------------------------


def test_curate_random_recipe(mcp_server, image_dir):
    body = call(mcp_server, "curate_source", {
        "source_path": image_dir,
        "k": 3,
        "recipe": "random",
    })
    assert body["report"]["n_candidates"] == 6
    assert body["report"]["n_kept"] == 3
    assert "run_id" in body


def test_curate_funnel_recipe(mcp_server, image_dir):
    body = call(mcp_server, "curate_source", {
        "source_path": image_dir,
        "k": 3,
        "recipe": "funnel",
    })
    assert body["report"]["n_candidates"] == 6
    assert "run_id" in body


# ---------------------------------------------------------------------------
# Annotations + runs
# ---------------------------------------------------------------------------


def test_get_annotations_after_curate(mcp_server, image_dir, memory_server):
    call(mcp_server, "curate_source", {
        "source_path": image_dir,
        "k": 3,
        "recipe": "random",
    })
    image_ids = list(memory_server.images.keys())
    body = call(mcp_server, "get_annotations", {"image_id": image_ids[0]})
    metric_ids = {a["metric_id"] for a in body["annotations"]}
    assert "random_score" in metric_ids


def test_list_and_get_run(mcp_server, image_dir):
    r1 = call(mcp_server, "curate_source", {
        "source_path": image_dir,
        "k": 3,
        "recipe": "random",
    })
    run_id = r1["run_id"]

    runs = call(mcp_server, "list_runs", {})
    assert run_id in runs["runs"]

    rec = call(mcp_server, "get_run", {"run_id": run_id})
    assert rec["run_id"] == run_id


# ---------------------------------------------------------------------------
# Image metadata
# ---------------------------------------------------------------------------


def test_get_image_metadata(mcp_server, image_dir, memory_server):
    call(mcp_server, "ingest_source", {"source_path": image_dir})
    image_ids = list(memory_server.images.keys())
    body = call(mcp_server, "get_image", {"image_id": image_ids[0]})
    assert body["image_id"] == image_ids[0]
    assert body["n_bytes"] > 0
    assert os.path.isfile(body["path"])
