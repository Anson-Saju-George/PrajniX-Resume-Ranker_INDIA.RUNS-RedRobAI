"""Apply Phase 1 soft flags as transparent multiplicative down-weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from engine.stages.integrity import IntegrityResult


@dataclass(frozen=True, slots=True)
class AppliedPenalty:
    flag: str
    factor: float
    evidence: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PenaltyResult:
    final_score: float
    combined_factor: float
    applied: tuple[AppliedPenalty, ...]


def apply_penalties(
    base_score: float,
    integrity: IntegrityResult,
    config: Mapping[str, Any],
) -> PenaltyResult:
    """Apply configured soft penalties; hard suppression is handled upstream."""

    if integrity["hard_suppress"]:
        raise ValueError("Hard-suppressed candidates must not reach penalty scoring")

    penalty_config = config["penalties"]
    factor = 1.0
    applied: list[AppliedPenalty] = []
    for flag in integrity["soft_flags"]:
        key = flag
        evidence = integrity["evidence"][flag]
        if flag == "long_notice_period" and evidence.get("severity") == "severe":
            key = "long_notice_period_severe"
        if key not in penalty_config:
            raise KeyError(f"Missing penalty configuration for {key}")
        current_factor = float(penalty_config[key])
        if not 0.0 < current_factor <= 1.0:
            raise ValueError(f"Penalty factor for {key} must be in (0, 1]")
        factor *= current_factor
        applied.append(AppliedPenalty(flag=flag, factor=current_factor, evidence=evidence))

    return PenaltyResult(
        final_score=base_score * factor,
        combined_factor=factor,
        applied=tuple(applied),
    )
