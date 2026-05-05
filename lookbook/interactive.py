"""Interactive curate loop — keep the human in the loop.

The non-interactive :func:`lookbook.curate` runs the pipeline once
and returns its top-k. That works for fully-automated workflows but
loses the human's taste signal: a ranking that ranks "boring but
high-quality" images above "stylistically perfect but slightly
blurry" ones.

:func:`curate_interactive` invites the caller into the loop. Each
round:

1. The pipeline picks the top ``present`` unseen candidates.
2. The caller's ``on_decision(refs, info)`` returns an
   :class:`InteractiveDecision` saying which to keep, which to
   reject, and whether to stop.
3. Kept refs lock into the final selection. Rejected refs leave the
   candidate pool and won't be shown again. Everything else is
   eligible to come back next round (handy when the caller wants to
   defer a decision).
4. The loop terminates when the caller says ``stop``, when the
   confirmed-kept set reaches ``k``, when the candidate pool is
   exhausted, or when ``max_rounds`` is reached.

Two ``on_decision`` shapes are supported:

- A function: ``on_decision(refs, info) -> InteractiveDecision``.
- A list of pre-recorded :class:`InteractiveDecision` items, applied
  one per round. Useful for tests and headless replays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from uuid import uuid4

from lookbook import registry
from lookbook.base import ImageRef
from lookbook.io import ingest as _ingest
from lookbook.pipeline import Pipeline, RunResult
from lookbook.store import Stores


PluginSpec = "Union[str, tuple]"  # mirror of lookbook.__init__'s alias


@dataclass(frozen=True, slots=True, kw_only=True)
class InteractiveDecision:
    """One round's verdict from the caller.

    Attributes:
        keep: ``image_id``s the caller wants to lock into the final
            selection. They're added immediately, removed from the
            candidate pool, and not shown again.
        reject: ``image_id``s the caller never wants to see again in
            this run.
        stop: When True, the loop terminates after applying this
            round's keep/reject and returns whatever is in the
            confirmed-kept set so far (clipped to ``k``).
    """

    keep: tuple[str, ...] = ()
    reject: tuple[str, ...] = ()
    stop: bool = False


DecisionCallable = Callable[[list[ImageRef], dict[str, Any]], InteractiveDecision]


def _as_decision_fn(
    on_decision: DecisionCallable | Sequence[InteractiveDecision],
) -> DecisionCallable:
    """Adapt either form of ``on_decision`` to a callable."""
    if callable(on_decision):
        return on_decision
    decisions = list(on_decision)
    state = {"i": 0}

    def _replay(refs: list[ImageRef], info: dict[str, Any]) -> InteractiveDecision:
        i = state["i"]
        state["i"] = i + 1
        if i < len(decisions):
            return decisions[i]
        return InteractiveDecision(stop=True)

    return _replay


def curate_interactive(
    source,
    *,
    on_decision: DecisionCallable | Sequence[InteractiveDecision],
    k: int = 20,
    present: int = 8,
    max_rounds: int = 20,
    scorer_ids: Sequence[Any] = ("random_score",),
    embedder_ids: Sequence[Any] = (),
    filter_ids: Sequence[Any] = (),
    selector_id: Any = "top_k",
    stores: Optional[Stores] = None,
    constraints: Optional[Mapping[str, Any]] = None,
) -> RunResult:
    """Run the pipeline in rounds, taking keep/reject decisions per round.

    Args mirror :func:`lookbook.curate` plus:

    Args:
        on_decision: A callable or a pre-recorded sequence of
            :class:`InteractiveDecision`. The callable receives the
            round's top candidates and an info dict
            ``{round, n_kept_so_far, n_pool_remaining}``.
        present: How many candidates to surface per round.
        max_rounds: Hard cap on iterations; a defensive guard against
            broken decision callables that never stop.

    Returns:
        A :class:`RunResult` whose ``kept`` list is the confirmed
        set (in the order they were kept). ``candidates`` carries the
        same as the final round's pre-selection candidate pool.
    """
    decision_fn = _as_decision_fn(on_decision)

    refs = (
        list(_ingest(source))
        if not isinstance(source, list)
        else list(source)
    )
    if k <= 0:
        raise ValueError(f"k must be positive (got {k})")
    if present <= 0:
        raise ValueError(f"present must be positive (got {present})")

    # Resolve plugins once.
    from lookbook import _resolve as _resolve_plugin  # noqa: WPS433

    pipeline = Pipeline(
        scorers=[_resolve_plugin(registry.scorers, sp) for sp in scorer_ids],
        embedders=[_resolve_plugin(registry.embedders, sp) for sp in embedder_ids],
        filters=[
            _resolve_plugin(registry.filters, sp, fresh=True) for sp in filter_ids
        ],
        selector=_resolve_plugin(registry.selectors, selector_id),
    )

    pool: list[ImageRef] = list(refs)
    by_id: dict[str, ImageRef] = {r.image_id: r for r in refs}
    confirmed: list[ImageRef] = []
    confirmed_ids: set[str] = set()
    rejected_ids: set[str] = set()

    started = datetime.now(timezone.utc)
    last_result: RunResult | None = None

    for round_idx in range(max_rounds):
        if len(confirmed) >= k:
            break
        eligible = [
            r for r in pool
            if r.image_id not in confirmed_ids and r.image_id not in rejected_ids
        ]
        if not eligible:
            break
        # Re-run the pipeline on the eligible pool to get this round's top.
        last_result = pipeline.run(
            eligible,
            k=min(present, len(eligible)),
            stores=stores,
            constraints=constraints,
        )
        info = {
            "round": round_idx,
            "n_kept_so_far": len(confirmed),
            "n_pool_remaining": len(eligible),
            "k": k,
            "present": present,
        }
        decision = decision_fn(list(last_result.kept), info)

        # Apply rejects first so a single decision can both reject and keep.
        for rid in decision.reject:
            rejected_ids.add(rid)
        for kid in decision.keep:
            if kid in confirmed_ids:
                continue
            ref = by_id.get(kid)
            if ref is None:
                continue
            confirmed.append(ref)
            confirmed_ids.add(kid)
            if len(confirmed) >= k:
                break

        if decision.stop:
            break

    finished = datetime.now(timezone.utc)
    return RunResult(
        run_id=uuid4().hex[:12],
        kept=confirmed[:k],
        candidates=(last_result.candidates if last_result else list(refs)),
        selector_id=str(selector_id),
        scorer_ids=[str(sp) for sp in scorer_ids],
        started_at=started,
        finished_at=finished,
        report={
            "interactive": True,
            "rounds": (round_idx + 1) if last_result is not None else 0,
            "n_rejected": len(rejected_ids),
        },
    )


__all__ = [
    "InteractiveDecision",
    "curate_interactive",
]
