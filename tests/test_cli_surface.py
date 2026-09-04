"""Characterization tests pinning the ``lookbook`` command-line surface.

``tests/cli_goldens/lookbook.json`` was recorded from the *argh* implementation
of :mod:`lookbook.__main__` before the migration to :mod:`cw`, with
``cw.testing.characterize``. Every assertion below therefore compares today's
CLI against the grammar argh produced, not against something written by hand
after the fact.

Three things are pinned, and each one is a bug that would otherwise ship
silently:

1. **The grammar** -- every command, flag, short flag, ``nargs`` and default,
   via the recorded ``usage:`` line and full stdout/stderr of 13 ``argv``
   vectors (``assert_replay``).
2. **The exit codes** -- ``cw.dispatch``/``cw.run`` *return* the exit code where
   argh exited by itself, so ``main()`` must hand it back to the console-script
   shim and the ``__main__`` guard must ``raise SystemExit(main())``. Drop
   either and argument errors start exiting 0; nothing else in this suite
   notices.
3. **No arguments prints usage to stdout and exits 0** -- argh's behaviour,
   which a plain argparse parser with a required subparser does *not* have.

The golden is replayed non-strictly: ``--help`` bodies are compared but a pure
formatting difference is reported rather than failed, because CPython rewrites
argparse's own option column between versions and this repo's CI matrix spans
several. (At migration time the strict comparison was empty on CPython 3.10 and
3.12 alike.)
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cw.testing import assert_replay

GOLDEN_PATH = Path(__file__).parent / "cli_goldens" / "lookbook.json"

# ``prog`` is pinned to "lookbook" inside main(), so the ``python -m`` form and
# the console script produce byte-identical output. Driving the module form
# keeps the test independent of PATH and of the ``.exe`` shim on Windows.
CLI = [sys.executable, "-m", "lookbook"]


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_cli_surface_matches_the_argh_recorded_golden(golden):
    """The whole grammar, replayed against what argh produced."""
    assert_replay(golden, prog=CLI)


def test_golden_carries_no_machine_specific_prog():
    """A golden that names an absolute path can only replay on one computer."""
    raw = GOLDEN_PATH.read_text(encoding="utf-8")
    assert "/Users/" not in raw and "\\\\Users\\\\" not in raw
    assert json.loads(raw)["prog"] == ["lookbook"]


def _run(*argv):
    return subprocess.run(
        CLI + list(argv), capture_output=True, text=True, timeout=120
    )


def test_no_arguments_prints_usage_to_stdout_and_exits_zero():
    """argh's behaviour, preserved. Plain argparse would exit 2 to stderr."""
    r = _run()
    assert r.returncode == 0
    assert r.stdout.startswith("usage: lookbook")
    assert r.stderr == ""


@pytest.mark.parametrize(
    "argv",
    [
        ("no-such-command",),
        ("curate", "--no-such-flag"),
        ("curate",),  # missing the required positional
    ],
)
def test_argument_errors_exit_two(argv):
    """Guards `raise SystemExit(main())` / `return cw.run(...)`.

    Without them the exit code is swallowed and every one of these exits 0.
    """
    assert _run(*argv).returncode == 2


def test_commands_list_is_what_the_parser_dispatches():
    """`COMMANDS` is the single source of truth the help text is built from."""
    from lookbook.__main__ import COMMANDS

    names = {f.__name__ for f in COMMANDS}
    assert names == {"curate", "list_plugins", "list_recipes", "serve", "mcp"}

    help_text = _run("--help").stdout
    for cli_name in ("curate", "list-plugins", "list-recipes", "serve", "mcp"):
        assert cli_name in help_text
