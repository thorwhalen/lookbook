"""Selectors: choose K from N given the manifest.

Each submodule registers its selectors into `lookbook.registry.selectors`.
"""

from lookbook.selectors import topk  # noqa: F401
from lookbook.selectors import submodular  # noqa: F401
from lookbook.selectors import quota  # noqa: F401
