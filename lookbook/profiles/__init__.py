"""Subject profiles — declarative recipes loaded from YAML.

A profile is just a recipe with a name and a YAML file. The schema mirrors
the in-code `RECIPES` dict in `lookbook/__main__.py`:

    name: person
    description: short description shown in `lookbook list-recipes`
    scorers: [scorer_id, ...]
    embedders: [embedder_id, ...]   # optional
    filters: [filter_id, ...]       # may include [name, kwargs] entries
    selector: selector_id           # may be [name, kwargs]
    diagnose_clusters: 12           # optional
    constraints: { ... }            # default constraints (e.g. quotas)

Profiles ship in `lookbook/profiles/*.yaml`. Users can override or add
profiles by dropping YAML files into `<config_root>/profiles/` (resolved
via `config2py`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Optional

from lookbook._paths import default_config_root, subdir

# In-code path to shipped profiles.
_PROFILES_DIR = os.path.dirname(os.path.abspath(__file__))


def _user_profiles_dir() -> str:
    return subdir(default_config_root(), "profiles")


def list_profiles() -> list[str]:
    """All profile names available, shipped + user-defined.

    User profiles override shipped ones with the same name.
    """
    names: dict[str, str] = {}
    for d in (_PROFILES_DIR, _user_profiles_dir()):
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith((".yaml", ".yml")):
                names[fn.rsplit(".", 1)[0]] = os.path.join(d, fn)
    return sorted(names)


def _profile_path(name: str) -> Optional[str]:
    """Locate a profile by name. User profiles win over shipped ones."""
    for d in (_user_profiles_dir(), _PROFILES_DIR):
        for ext in (".yaml", ".yml"):
            p = os.path.join(d, name + ext)
            if os.path.isfile(p):
                return p
    return None


def load(name: str) -> dict:
    """Load a profile by name, returning a recipe-shaped dict.

    Raises `KeyError` if the profile is unknown.

    >>> spec = load("person")  # doctest: +SKIP
    >>> "scorers" in spec
    True
    """
    path = _profile_path(name)
    if path is None:
        raise KeyError(
            f"Unknown profile: {name!r}. Known: {list_profiles()}"
        )
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Loading lookbook profiles requires `pyyaml`. "
            "`pip install pyyaml` or `pip install lookbook[profiles]`."
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _normalize(data)


def _normalize(spec: Mapping[str, Any]) -> dict:
    """Coerce YAML-friendly forms into the runtime recipe shape.

    YAML naturally uses lists for `[name, kwargs]` overrides. The runtime
    recipe shape uses tuples for the same purpose; `lookbook.__main__`
    handles either. Here we just ensure the keys are present with safe
    defaults.
    """
    return {
        "name": spec.get("name", ""),
        "description": spec.get("description", ""),
        "scorers": list(spec.get("scorers", [])),
        "embedders": list(spec.get("embedders", [])),
        "filters": list(spec.get("filters", [])),
        "selector": spec.get("selector", "top_k"),
        "diagnose_clusters": int(spec.get("diagnose_clusters", 0)),
        "constraints": dict(spec.get("constraints", {}) or {}),
    }
