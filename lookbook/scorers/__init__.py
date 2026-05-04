"""Scorers: per-image annotation producers.

Each submodule registers its scorers into `lookbook.registry.scorers` on
import. Heavy ML deps must be lazy-imported inside scorer methods, never at
module top.
"""
from lookbook.scorers import technical  # noqa: F401  (registers Phase 0/1 scorers)
