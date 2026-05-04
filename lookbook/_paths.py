"""Default paths for lookbook persistent state, via config2py.

All path resolution funnels through here so that swapping app names or
storage roots is a one-place change.
"""

from __future__ import annotations

import os
from typing import Literal

from config2py import get_app_data_folder, get_app_config_folder

APP_NAME = "lookbook"

FolderKind = Literal["data", "config", "cache", "state"]


def app_folder(kind: FolderKind = "data", *, ensure_exists: bool = True) -> str:
    """Return the user-app folder for `lookbook` of the requested kind.

    >>> p = app_folder("data")
    >>> p.endswith("lookbook")
    True
    """
    if kind == "config":
        return get_app_config_folder(APP_NAME, ensure_exists=ensure_exists)
    return get_app_data_folder(
        APP_NAME, folder_kind=kind, ensure_exists=ensure_exists
    )


def default_data_root() -> str:
    """Default root for persistent run data (manifest, runs, images)."""
    return app_folder("data")


def default_cache_root() -> str:
    """Default root for regeneratable artifacts (model weights, embeddings)."""
    return app_folder("cache")


def default_config_root() -> str:
    """Default root for user-edited recipes and profiles."""
    return app_folder("config")


def subdir(root: str, name: str) -> str:
    """Join `name` under `root` and ensure the directory exists."""
    p = os.path.join(root, name)
    os.makedirs(p, exist_ok=True)
    return p
