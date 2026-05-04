"""Plugin registry for scorers, filters, embedders, selectors.

Anything that implements one of the protocols in `lookbook.base` can be
registered here under a string id and looked up by the orchestrator. This
is the open-closed boundary in concrete form: new metrics are *registered*,
never *subclassed*.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class _Registry(Generic[T]):
    """A tiny string-keyed registry with introspection."""

    def __init__(self, kind: str):
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T, *, overwrite: bool = False) -> T:
        if not overwrite and name in self._items:
            raise KeyError(
                f"{self._kind} already registered: {name!r} "
                f"(pass overwrite=True to replace)"
            )
        self._items[name] = item
        return item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"No {self._kind} registered under name {name!r}. "
                f"Known: {sorted(self._items)}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self) -> Iterable[str]:
        return iter(self._items)


# Public registries, one per protocol. Modules under `scorers/` and
# `selectors/` register into these on import.
scorers = _Registry[Any]("scorer")
filters = _Registry[Any]("filter")
embedders = _Registry[Any]("embedder")
selectors = _Registry[Any]("selector")


def register_scorer(name: str, **kwargs) -> Callable:
    """Decorator form: `@register_scorer("blur")`."""

    def deco(obj):
        scorers.register(name, obj, **kwargs)
        return obj

    return deco


def register_filter(name: str, **kwargs) -> Callable:
    def deco(obj):
        filters.register(name, obj, **kwargs)
        return obj

    return deco


def register_embedder(name: str, **kwargs) -> Callable:
    def deco(obj):
        embedders.register(name, obj, **kwargs)
        return obj

    return deco


def register_selector(name: str, **kwargs) -> Callable:
    def deco(obj):
        selectors.register(name, obj, **kwargs)
        return obj

    return deco
