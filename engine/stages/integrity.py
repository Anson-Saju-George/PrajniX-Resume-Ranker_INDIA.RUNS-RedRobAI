"""Orchestrate Phase 1 integrity checks without ranking candidates."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, TypedDict

from engine.data import Candidate
from engine.features.integrity_features import (
    HARD_CHECKS,
    HARD_FLAG_NAMES,
    SOFT_CHECKS,
)


class IntegrityResult(TypedDict):
    candidate_id: str
    hard_suppress: bool
    soft_flags: list[str]
    evidence: dict[str, dict[str, Any]]


def check_candidate_integrity(candidate: Candidate) -> IntegrityResult:
    """Run every Phase 1 check and return flags with source-field evidence."""

    evidence: dict[str, dict[str, Any]] = {}

    for check in HARD_CHECKS:
        result = check(candidate)
        if result.triggered:
            if not result.evidence:
                raise ValueError(f"Hard flag {result.name} has no evidence")
            evidence[result.name] = result.evidence

    soft_flags: list[str] = []
    for check in SOFT_CHECKS:
        result = check(candidate)
        if result.triggered:
            if not result.evidence:
                raise ValueError(f"Soft flag {result.name} has no evidence")
            soft_flags.append(result.name)
            evidence[result.name] = result.evidence

    hard_suppress = any(name in HARD_FLAG_NAMES for name in evidence)
    return {
        "candidate_id": candidate.candidate_id,
        "hard_suppress": hard_suppress,
        "soft_flags": soft_flags,
        "evidence": evidence,
    }


def run_integrity_checks(candidates: Iterable[Candidate]) -> Iterator[IntegrityResult]:
    """Yield integrity results lazily for a stream of typed candidates."""

    for candidate in candidates:
        yield check_candidate_integrity(candidate)
