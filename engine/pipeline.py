"""Harness-shaped score-everyone baseline pipeline (variant B)."""

from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from engine.data import Candidate, stream_candidates
from engine.stages.integrity import IntegrityResult, check_candidate_integrity
from engine.stages.output import SubmissionRow, validate_submission_rows
from engine.stages.penalties import PenaltyResult, apply_penalties
from engine.stages.recall import apply_recall_mode, uses_score_everyone_fallback
from engine.stages.reasoning import build_reasoning
from engine.stages.scoring import (
    CandidateFeatures,
    ScoringResult,
    extract_features,
    score_candidate,
)


@dataclass(frozen=True, slots=True)
class _TopCandidate:
    candidate: Candidate
    integrity: IntegrityResult
    features: CandidateFeatures
    scoring: ScoringResult
    penalties: PenaltyResult
    final_score: float


@dataclass(frozen=True, slots=True)
class _PipelineResult:
    rows: tuple[SubmissionRow, ...]
    debug_top100: tuple[dict[str, Any], ...]
    source_candidate_ids: frozenset[str]
    records_streamed: int
    hard_suppressed: int
    survivors_scored: int
    recall_mode: str
    recall_fallback_to_score_everyone: bool
    evaluation_scores: dict[str, float]


__all__ = ["rank_candidates"]


def _debug_record(rank: int, item: _TopCandidate, reasoning: str) -> dict[str, Any]:
    candidate = item.candidate
    return {
        "rank": rank,
        "candidate_id": candidate.candidate_id,
        "current_title": candidate.profile.current_title,
        "current_company": candidate.profile.current_company,
        "years_of_experience": candidate.profile.years_of_experience,
        "location": candidate.profile.location,
        "base_score": item.scoring.base_score,
        "penalty_factor": item.penalties.combined_factor,
        "final_score": item.final_score,
        "aspect_scores": asdict(item.scoring.aspects),
        "soft_flags": item.integrity["soft_flags"],
        "hard_suppress": item.integrity["hard_suppress"],
        "feature_summary": {
            "corroborated_retrieval_skills": [
                skill.name for skill in item.features.skills.corroborated_retrieval
            ],
            "corroborated_vector_databases": [
                skill.name for skill in item.features.skills.corroborated_vector_databases
            ],
            "retrieval_career_roles": item.features.career.retrieval_role_count,
            "evaluation_career_roles": item.features.career.evaluation_role_count,
            "product_roles": item.features.career.product_role_count,
            "production_search_roles": item.features.production.production_role_count,
            "experience_band": item.features.experience.band,
            "last_active_date": item.features.availability.last_active_date,
            "open_to_work": item.features.availability.open_to_work,
        },
        "reasoning": reasoning,
    }


def rank_candidates(
    dataset: str | Path,
    jd: str | Path,
    config: Mapping[str, Any],
) -> _PipelineResult:
    """Stream and score every surviving candidate, retaining only the best 100."""

    dataset_path = Path(dataset)
    jd_path = Path(jd)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if not jd_path.is_file():
        raise FileNotFoundError(jd_path)
    recall_mode = str(config["recall_mode"])

    top_k = int(config.get("top_k", 100))
    if top_k != 100:
        raise ValueError("Challenge output requires top_k=100")
    decimals = int(config.get("score_decimals", 8))
    reference_date = date.fromisoformat(str(config["reference_date"]))

    heap: list[tuple[tuple[float, int], _TopCandidate]] = []
    source_ids: set[str] = set()
    records_streamed = 0
    hard_suppressed = 0
    survivors_scored = 0
    evaluation_ids = set(config.get("_evaluation_candidate_ids", ()))
    evaluation_scores: dict[str, float] = {}

    candidates = apply_recall_mode(stream_candidates(dataset_path), recall_mode)
    for candidate in candidates:
        records_streamed += 1
        if candidate.candidate_id in source_ids:
            raise ValueError(f"Duplicate candidate ID: {candidate.candidate_id}")
        source_ids.add(candidate.candidate_id)

        integrity = check_candidate_integrity(candidate)
        if integrity["hard_suppress"]:
            hard_suppressed += 1
            if candidate.candidate_id in evaluation_ids:
                evaluation_scores[candidate.candidate_id] = -1.0
            continue

        features = extract_features(candidate, reference_date)
        scoring = score_candidate(candidate, features, config)
        penalties = apply_penalties(scoring.base_score, integrity, config)
        final_score = round(penalties.final_score, decimals)
        if candidate.candidate_id in evaluation_ids:
            evaluation_scores[candidate.candidate_id] = final_score
        survivors_scored += 1

        item = _TopCandidate(
            candidate=candidate,
            integrity=integrity,
            features=features,
            scoring=scoring,
            penalties=penalties,
            final_score=final_score,
        )
        numeric_id = int(candidate.candidate_id.removeprefix("CAND_"))
        # Lowest score is worst; at equal score the larger ID is worst.
        quality_key = (final_score, -numeric_id)
        if len(heap) < top_k:
            heapq.heappush(heap, (quality_key, item))
        elif quality_key > heap[0][0]:
            heapq.heapreplace(heap, (quality_key, item))

    selected = sorted(
        (item for _, item in heap),
        key=lambda item: (-item.final_score, item.candidate.candidate_id),
    )
    rows: list[SubmissionRow] = []
    debug: list[dict[str, Any]] = []
    for rank, item in enumerate(selected, start=1):
        reasoning = build_reasoning(item.candidate, item.features)
        rows.append(
            SubmissionRow(
                candidate_id=item.candidate.candidate_id,
                rank=rank,
                score=item.final_score,
                reasoning=reasoning,
            )
        )
        debug.append(_debug_record(rank, item, reasoning))

    guarded_rows = validate_submission_rows(rows, source_ids)
    missing_evaluation_ids = evaluation_ids - set(evaluation_scores)
    if missing_evaluation_ids:
        raise ValueError(
            f"Evaluation IDs absent from candidate stream: {sorted(missing_evaluation_ids)[:5]}"
        )
    return _PipelineResult(
        rows=tuple(guarded_rows),
        debug_top100=tuple(debug),
        source_candidate_ids=frozenset(source_ids),
        records_streamed=records_streamed,
        hard_suppressed=hard_suppressed,
        survivors_scored=survivors_scored,
        recall_mode=recall_mode,
        recall_fallback_to_score_everyone=uses_score_everyone_fallback(recall_mode),
        evaluation_scores=evaluation_scores,
    )
