"""Embedders: vector representations per image.

Each submodule registers its embedders into `lookbook.registry.embedders`
on import. Heavy ML deps must be lazy-imported inside `embed()`, never at
module top.

We pre-set `USE_TF=0` and `USE_FLAX=0` so that `transformers` does not try
to load TensorFlow or Flax — both can blow up if the user's environment
has an incompatible numpy ABI. `setdefault` means we never override a user
who explicitly wants those backends.
"""
import os as _os

_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from lookbook.embedders import mock  # noqa: F401  (always available)
from lookbook.embedders import clip  # noqa: F401  (lazy: needs transformers)
from lookbook.embedders import dinov2  # noqa: F401  (lazy: needs transformers)
