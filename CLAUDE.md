# lookbook — agent & contributor guide

`lookbook` distills N candidate images into K well-chosen ones: per-image
**scoring** (is this image individually good?) separated from set-level
**selection** (does this collection cover the concept?). It is **AI-first** —
most callers are agents, over MCP or HTTP — so the fastest way to use it well
is through the bundled skills.

## Start here (AI-first)

Nine skills live in `.claude/skills/`. Load the one that matches the job:

**Using lookbook**
- [`lookbook-curate`](.claude/skills/lookbook-curate/SKILL.md) — the headline
  workflow: which recipe for which intent, the one-call
  `curate_for_character` / `curate_for_environment` presets, and
  `compare_to_reference`. Read this first if you are *driving* lookbook.
- [`lookbook-diagnose`](.claude/skills/lookbook-diagnose/SKILL.md) — after a
  run: why an image was dropped, what the kept set is missing.
- [`lookbook-recipe`](.claude/skills/lookbook-recipe/SKILL.md) — tuning:
  per-call `(name, kwargs)` overrides and user YAML profiles.

**Building lookbook**
- [`lookbook-dev`](.claude/skills/lookbook-dev/SKILL.md) — architecture,
  current state, the protocols, the manifest. Read this first if you are
  *changing* lookbook.
- [`lookbook-add-scorer`](.claude/skills/lookbook-add-scorer/SKILL.md) — a new
  per-image metric (incl. derived and cross-image ones).
- [`lookbook-add-selector`](.claude/skills/lookbook-add-selector/SKILL.md) — a
  new set-selection algorithm.
- [`lookbook-profile`](.claude/skills/lookbook-profile/SKILL.md) — a new
  subject profile (product, scene, style …) as YAML.
- [`lookbook-storage`](.claude/skills/lookbook-storage/SKILL.md) — the `dol`
  stores, the manifest codec, swapping backends, wiring stores for tests.
- [`lookbook-http`](.claude/skills/lookbook-http/SKILL.md) — the route table
  and how to add a verb (HTTP + its MCP mirror).

Long-form background lives in `misc/docs/`:
[`lookbook_design_report.md`](misc/docs/lookbook_design_report.md) (problem
framing, metrics catalogue) and
[`lookbook_development_plan.md`](misc/docs/lookbook_development_plan.md) (the
authoritative status table; everything under its "original plan of record"
line is history, not current state).

## Capability map

| Want | Reach for |
|---|---|
| Pick K good, diverse images from a pool | `curate(source, k=..., ...)` or the CLI's `--recipe` |
| Pick the best reference frame of a known character | `curate_for_character(source, k=1)` |
| Pick the best plate / prop / style reference | `curate_for_environment(source, k=1)` |
| Let a human choose, round by round | `curate_interactive(source, on_decision=...)` |
| "Is this generation still the same subject?" | `compare_to_reference(reference, candidate)` |
| One metric for one image | `score(ref, metric_id=...)` |
| A real file on disk for any ImageRef | `to_local_path(ref)` / `ref.local_path()` |

Everything above is a top-level `lookbook` export. The HTTP surface
(`mk_lookbook_app`) and the MCP surface (`mk_lookbook_mcp`) expose the same
ten verbs each; the `curate_for_*` presets are Python-only.

## Architecture

Five layers; the heavy ML libs live only at the bottom so the upper layers
stay laptop-installable.

```
Interface       (CLI, HTTP via qh, MCP via fastmcp, Python lib)
Recipe / facade (lookbook.facade, named recipes, YAML profiles)
Orchestration   (lookbook.pipeline, manifest, drop attribution, run records)
Plugin layer    (Scorer | Filter | Embedder | Selector — Protocols in base.py)
Backend         (CLIP, DINOv2, InsightFace, 6DRepNet — wrapped, lazy-imported)
```

Four commitments hold this together:

1. **The manifest is the SSOT.** It is a
   `MutableMapping[(image_id, metric_id), Annotation]` — nothing more. That is
   what lets the same code run over `dict`, `JsonFiles`, `mongodol`, `s3dol`.
2. **Plugins are registered, never subclassed.** `registry.scorers` /
   `filters` / `embedders` / `selectors`. Adding a metric never edits core.
   (The registries are `lookbook.registry.scorers` etc. — `lookbook.scorers`
   is the *submodule* that holds scorer implementations.)
3. **`config_hash` is the cache key.** A scorer is skipped when the manifest
   already holds `(image_id, metric_id)` with a matching `config_hash`. Any
   knob that changes the output must be in the hash.
4. **No `import torch` above the backend layer.** Heavy deps are lazy-imported
   inside scorer/embedder methods, never at module top, and raise an
   `ImportError` naming the right extra.

`lookbook/__init__.py` only re-exports — no logic, no `from __future__ import
annotations` — so `dir(lookbook)` stays the public API plus submodules. New
facade functions go in `lookbook/facade.py`.

## Conventions

- Favor functional style; dataclasses for data; `Protocol` over ABC. Small
  focused helpers (`_underscore` when module-private, no prefix when reused
  across modules).
- Arguments beyond the third position are keyword-only. No magic numbers —
  they become dataclass fields, which is also what makes them tunable via the
  `(name, kwargs)` spec form.
- Every module needs a top-level docstring (auto-extracted for the docs).
- Paths route through `lookbook._paths` (`config2py` under the hood). Never
  hardcode a data/cache/config location anywhere else.

## Tests

`python -m pytest -q` from the repo root — that is also how you get the
current test count, so don't pin a number here. Two rules that matter:

- **Tests stay offline and free.** Pass `dict()` for every store slot
  (`get_stores(images_store={}, manifest_store={}, runs_store={},
  embeddings={})`) so nothing touches the user's app data folder, and use the
  mock backends — `mock_face`, `mock_head_pose`, `arcface_mock`, `mock` — so
  nothing downloads model weights. Real-model smokes are opt-in behind
  `LOOKBOOK_TEST_MODELS=1`.
- HTTP tests swap the server singleton with `lookbook.http.reset_stores(...)`;
  see the fixtures in `tests/test_phase4.py`.
