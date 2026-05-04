"""MCP surface for lookbook, built with `fastmcp`.

Each verb from `lookbook.http` is exposed as an MCP tool that an LLM
agent can call. The tool functions are intentionally the same callables
used by the HTTP layer — JSON-able args, JSON-able returns. This means
swapping between an HTTP frontend and an MCP frontend is "wire it once,
expose it twice."

Run the server over stdio with `lookbook mcp` (the default transport
that Claude Desktop and the Anthropic SDK expect).
"""

from __future__ import annotations

from typing import Any, Optional

from lookbook.http import (
    curate_source,
    get_annotations,
    get_image,
    get_run,
    ingest_source,
    list_plugins,
    list_recipes,
    list_runs,
    score_image,
)


# ---------------------------------------------------------------------------
# MCP tool wrappers
# ---------------------------------------------------------------------------
#
# We rewrap each http function as a thin top-level function with a tighter
# signature and a description that's tuned for an LLM caller. This gives
# us MCP-friendly schemas without needing to touch the HTTP versions.


def mcp_list_recipes() -> dict:
    """List every available curation recipe / profile.

    Returns a dict where each value is a recipe spec describing the
    scorers, embedders, filters, selector, and default constraints. Use
    this first to see what `curate_source` will accept for the `recipe`
    argument.
    """
    return list_recipes()


def mcp_list_plugins() -> dict:
    """List every registered scorer / embedder / filter / selector.

    Use this to discover available metric_ids for `score_image` and to
    understand what capabilities each recipe is wiring together.
    """
    return list_plugins()


def mcp_ingest_source(source_path: str) -> dict:
    """Scan a directory or file for images and register them on the server.

    Records `image_id -> path` in the server's image store so later calls
    (`score_image`, `get_image`, `curate_source`) can refer to images by
    id. Returns the list of image_ids and the count.

    Args:
        source_path: absolute path on the server's filesystem.
    """
    return ingest_source(source_path)


def mcp_curate_source(
    source_path: str,
    k: int = 20,
    recipe: str = "funnel",
) -> dict:
    """Run a full curation pipeline on a directory and return the K best.

    The high-level entry point. Picks K reference images from the
    directory using the named recipe (cheap funnel, embeddings + facility
    location, person profile, etc.). Returns a run record containing the
    chosen image_ids, drop attributions per filter, optional cluster
    coverage, scorer ids that ran, and run timing.

    Args:
        source_path: absolute path to the directory of candidate images.
        k: target subset size (the number of images to keep).
        recipe: a recipe / profile name. Use `list_recipes` to see options.

    Common recipes:
      - `random`        : placeholder, deterministic, no real scoring.
      - `funnel`        : Phase 1 cheap filter — fast, no GPU.
      - `funnel_relaxed`: same but lower thresholds for typical phone shots.
      - `diverse`       : DINOv2 embeddings + facility-location.
      - `diverse_clip`  : same with CLIP semantic embeddings.
      - `person`        : full character LoRA recipe (face detection +
                          ArcFace identity + pose-bin quotas).
      - `person_mock`   : person without model downloads (testing).
    """
    return curate_source(source_path=source_path, k=k, recipe=recipe)


def mcp_score_image(
    metric_id: str,
    source_path: str = "",
    image_id: str = "",
) -> dict:
    """Score one image against one metric and cache the result.

    Pass either `source_path` (a path on the server) or `image_id` (must
    already be registered via `ingest_source`). Returns the metric value;
    re-running with the same metric_id and config is free.

    Args:
        metric_id: a registered scorer name (see `list_plugins`).
        source_path: optional, the path to the image.
        image_id: optional, an id from a previous `ingest_source`.
    """
    return score_image(
        source_path=source_path,
        image_id=image_id,
        metric_id=metric_id,
    )


def mcp_get_annotations(image_id: str) -> dict:
    """All annotations stored for one image.

    Includes the metric_id, value, config_hash, cost_tier, backend, and
    timestamp for each annotation. Use this to understand *why* an image
    was kept or rejected — the manifest holds the full provenance.

    Args:
        image_id: as returned from `ingest_source` or `curate_source`.
    """
    return get_annotations(image_id)


def mcp_list_runs() -> dict:
    """All curation runs stored on this server."""
    return list_runs()


def mcp_get_run(run_id: str) -> dict:
    """Fetch a stored run record by id.

    The run record contains: kept and candidate image_ids, scorer_ids,
    selector_id, started_at / finished_at timestamps, and a `report`
    block with drop attributions and (when applicable) cluster coverage.
    """
    return get_run(run_id)


def mcp_get_image(image_id: str) -> dict:
    """Look up an image's path and metadata by id.

    Returns `{image_id, path, n_bytes}` when found. Phase 5 returns
    metadata only — agents that want the raw bytes should read the file
    directly when running on the same host as the server.
    """
    return get_image(image_id)


# ---------------------------------------------------------------------------
# Server builder
# ---------------------------------------------------------------------------


def mk_lookbook_mcp(*, name: str = "lookbook"):
    """Build a FastMCP server exposing the lookbook curation verbs.

    Each verb becomes an MCP tool. The agent's typical sequence is:

        list_recipes() -> pick a recipe
        curate_source(path, k, recipe) -> get back a run record
        get_annotations(image_id) -> understand why each kept image won
        get_run(run_id) -> review the full report

    The server uses the same Stores singleton as `lookbook.http`, so an
    HTTP server and an MCP server backed by the same data folder share
    the manifest and the runs index.
    """
    try:
        from fastmcp import FastMCP  # type: ignore
    except ImportError as e:
        raise ImportError(
            "MCP server requires `fastmcp`. "
            "`pip install lookbook[mcp]` or `pip install fastmcp`."
        ) from e

    mcp = FastMCP(
        name=name,
        instructions=(
            "lookbook curates image pools for personalized model training. "
            "Use `list_recipes` to see what recipes are available, then "
            "`curate_source` to pick K best images. `get_annotations` and "
            "`get_run` give the full provenance behind each decision."
        ),
    )

    # Register each tool with a clean MCP-visible name (drop the `mcp_` prefix).
    mcp.tool(name="list_recipes")(mcp_list_recipes)
    mcp.tool(name="list_plugins")(mcp_list_plugins)
    mcp.tool(name="ingest_source")(mcp_ingest_source)
    mcp.tool(name="curate_source")(mcp_curate_source)
    mcp.tool(name="score_image")(mcp_score_image)
    mcp.tool(name="get_annotations")(mcp_get_annotations)
    mcp.tool(name="list_runs")(mcp_list_runs)
    mcp.tool(name="get_run")(mcp_get_run)
    mcp.tool(name="get_image")(mcp_get_image)

    return mcp


def serve(*, transport: str = "stdio", **kwargs) -> None:
    """Run the lookbook MCP server.

    The default transport is stdio — the protocol Claude Desktop and the
    Anthropic SDK expect when launching an MCP server as a subprocess.
    Pass `transport="http"` (and a port via kwargs) for HTTP-streamable
    transport.
    """
    mcp = mk_lookbook_mcp()
    mcp.run(transport=transport, **kwargs)
