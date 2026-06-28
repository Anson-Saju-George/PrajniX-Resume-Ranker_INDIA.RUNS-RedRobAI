"""Recall-mode registry for the variant harness.

Phase 3 intentionally contains no retrieval intelligence. Modes A and D are
selectable configuration values but transparently fall back to the exact
score-everyone stream used by mode B until Phase 5 implements their internals.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from engine.data import Candidate


REGISTERED_RECALL_MODES = frozenset(
    {"A_retrieval_first", "B_score_everyone", "D_dual_channel"}
)
STUBBED_RECALL_MODES = frozenset({"A_retrieval_first", "D_dual_channel"})


def apply_recall_mode(
    candidates: Iterable[Candidate], recall_mode: str
) -> Iterator[Candidate]:
    """Yield the candidate stream for a registered mode.

    All modes currently yield every record. Keeping this indirection explicit
    allows Phase 5 to replace only A/D internals without changing the pipeline.
    """

    if recall_mode not in REGISTERED_RECALL_MODES:
        raise ValueError(
            f"Unknown recall_mode {recall_mode!r}; expected one of "
            f"{sorted(REGISTERED_RECALL_MODES)}"
        )
    yield from candidates


def uses_score_everyone_fallback(recall_mode: str) -> bool:
    if recall_mode not in REGISTERED_RECALL_MODES:
        raise ValueError(f"Unknown recall_mode {recall_mode!r}")
    return recall_mode in STUBBED_RECALL_MODES
