"""``curate_interactive`` keeps the human in the loop, replayable for tests."""

from __future__ import annotations

import pytest

from lookbook import (
    InteractiveDecision,
    PathImageRef,
    curate_interactive,
)


PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _make_refs(tmp_path, n=12):
    """Generate ``n`` PathImageRefs with unique paths so image_ids differ."""
    refs = []
    for i in range(n):
        f = tmp_path / f"img_{i:02d}.png"
        f.write_bytes(PNG_HEADER + bytes([i]))
        refs.append(PathImageRef(path=str(f)))
    return refs


def test_replay_decisions_locks_in_kept(tmp_path):
    """A pre-recorded decision sequence pins down what ends up in `.kept`."""
    refs = _make_refs(tmp_path, n=10)
    # Round 1: keep first ref's id, reject second.
    decisions = [
        InteractiveDecision(keep=(refs[0].image_id,), reject=(refs[1].image_id,)),
        InteractiveDecision(keep=(refs[2].image_id,), stop=True),
    ]
    result = curate_interactive(
        refs,
        on_decision=decisions,
        k=10,
        present=4,
    )
    kept_ids = {r.image_id for r in result.kept}
    assert refs[0].image_id in kept_ids
    assert refs[2].image_id in kept_ids
    assert refs[1].image_id not in kept_ids


def test_loop_stops_when_k_reached(tmp_path):
    refs = _make_refs(tmp_path, n=10)
    decisions = [
        InteractiveDecision(keep=tuple(r.image_id for r in refs[:3])),
        InteractiveDecision(keep=tuple(r.image_id for r in refs[3:5])),
    ]
    result = curate_interactive(
        refs,
        on_decision=decisions,
        k=4,
        present=10,
    )
    assert len(result.kept) == 4


def test_loop_stops_when_pool_exhausted(tmp_path):
    refs = _make_refs(tmp_path, n=4)
    # Reject everything, never keep anything → loop should exit cleanly.
    decisions = [
        InteractiveDecision(reject=tuple(r.image_id for r in refs))
    ]
    result = curate_interactive(
        refs,
        on_decision=decisions,
        k=10,
        present=10,
    )
    assert result.kept == []
    assert result.report["n_rejected"] == 4


def test_callable_decision_receives_round_info(tmp_path):
    refs = _make_refs(tmp_path, n=10)
    seen_rounds: list[int] = []

    def decide(presented, info):
        seen_rounds.append(info["round"])
        return InteractiveDecision(stop=True)

    curate_interactive(refs, on_decision=decide, k=5, present=4)
    assert seen_rounds == [0]


def test_callable_decision_keeps_then_iterates(tmp_path):
    """Decisions can keep one per round and iterate until k is reached."""
    refs = _make_refs(tmp_path, n=10)
    state = {"i": 0}

    def decide(presented, info):
        # Keep the first presented ref each round.
        state["i"] += 1
        return InteractiveDecision(keep=(presented[0].image_id,))

    result = curate_interactive(refs, on_decision=decide, k=3, present=4)
    assert len(result.kept) == 3
    assert state["i"] == 3


def test_max_rounds_caps_iteration(tmp_path):
    """A pathological decision (never stops, never picks anything) terminates."""
    refs = _make_refs(tmp_path, n=10)

    def never_decides(presented, info):
        return InteractiveDecision()  # neither keep, reject, nor stop

    # Without max_rounds protection this would be an infinite loop — and
    # the candidate pool never shrinks because nothing was rejected.
    result = curate_interactive(
        refs, on_decision=never_decides, k=5, present=4, max_rounds=3,
    )
    assert result.kept == []


def test_invalid_k_or_present_raises(tmp_path):
    refs = _make_refs(tmp_path, n=2)
    with pytest.raises(ValueError):
        curate_interactive(refs, on_decision=[], k=0)
    with pytest.raises(ValueError):
        curate_interactive(refs, on_decision=[], k=5, present=0)


def test_decision_keep_wins_over_reject(tmp_path):
    """If the same id is in both keep and reject, keep wins (rejects apply first
    but the keep then re-adds the id to confirmed)."""
    refs = _make_refs(tmp_path, n=4)
    target = refs[0].image_id
    decisions = [
        # First reject everyone; round 2 won't have any candidates.
        InteractiveDecision(
            reject=(target,),
            keep=(target,),
            stop=True,
        ),
    ]
    result = curate_interactive(
        refs, on_decision=decisions, k=4, present=4
    )
    # The implementation applies rejects, then keeps. In the current
    # design, a kept id remains kept even if also in reject set.
    kept_ids = {r.image_id for r in result.kept}
    assert target in kept_ids
