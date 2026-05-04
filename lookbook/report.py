"""Run report — what happened during a curation pass.

The report is the user-facing artifact alongside the kept set. It explains
*why* images were dropped and what the kept set looks like, in terms a
human (or an LLM agent) can act on.

Phase 1 ships the drop-attribution part. Coverage / set-level diagnosis
lands in Phase 2 alongside embeddings.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Mapping

from lookbook.base import ImageRef, Manifest


@dataclass
class Report:
    """Summary of a pipeline run."""

    n_candidates: int = 0
    n_survivors: int = 0
    n_kept: int = 0
    dropped_by_filter: dict = field(default_factory=dict)
    scorer_ids: list = field(default_factory=list)
    selector_id: str = ""
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_candidates": self.n_candidates,
            "n_survivors": self.n_survivors,
            "n_kept": self.n_kept,
            "dropped_by_filter": dict(self.dropped_by_filter),
            "scorer_ids": list(self.scorer_ids),
            "selector_id": self.selector_id,
            "notes": dict(self.notes),
        }

    def human(self) -> str:
        """Render a short human-readable summary."""
        lines = [
            f"candidates: {self.n_candidates}",
            f"survivors:  {self.n_survivors}",
            f"kept:       {self.n_kept}",
        ]
        if self.dropped_by_filter:
            lines.append("dropped by filter:")
            for fname, count in sorted(
                self.dropped_by_filter.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"  - {fname}: {count}")
        if self.notes:
            lines.append("notes:")
            for k, v in self.notes.items():
                lines.append(f"  - {k}: {v}")
        return "\n".join(lines)


def attribute_drops(
    candidates: Iterable[ImageRef],
    filters: list,
    manifest: Manifest,
) -> tuple[list[ImageRef], dict[str, int]]:
    """Apply filters in order, attributing each drop to the first filter
    that rejected it.

    Returns (survivors, drop_counts_by_filter_name).
    """
    survivors: list[ImageRef] = []
    drops: Counter = Counter()
    for ref in candidates:
        kept = True
        for f in filters:
            if not f.keep(ref, manifest):
                drops[getattr(f, "name", type(f).__name__)] += 1
                kept = False
                break
        if kept:
            survivors.append(ref)
    return survivors, dict(drops)
