"""Weighted aspect scoring for the score-everyone baseline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from engine.data import Candidate
from engine.features import (
    availability_features,
    career_features,
    experience_features,
    production_features,
    skill_features,
)


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    skills: skill_features.SkillFeatures
    career: career_features.CareerFeatures
    experience: experience_features.ExperienceFeatures
    production: production_features.ProductionFeatures
    availability: availability_features.AvailabilityFeatures


@dataclass(frozen=True, slots=True)
class AspectScores:
    skills: float
    career: float
    production: float
    experience: float
    availability: float
    integrity: float


@dataclass(frozen=True, slots=True)
class ScoringResult:
    aspects: AspectScores
    base_score: float


def extract_features(candidate: Candidate, reference_date: date) -> CandidateFeatures:
    """Run each reusable feature family exactly once for a candidate."""

    return CandidateFeatures(
        skills=skill_features.extract(candidate),
        career=career_features.extract(candidate),
        experience=experience_features.extract(candidate),
        production=production_features.extract(candidate),
        availability=availability_features.extract(candidate, reference_date),
    )


def _validate_weights(weights: Mapping[str, float]) -> None:
    expected = set(AspectScores.__dataclass_fields__)
    if set(weights) != expected:
        raise ValueError(f"Scoring weights must be exactly {sorted(expected)}")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ValueError("Scoring weights must sum to 1.0")


def score_candidate(
    candidate: Candidate,
    features: CandidateFeatures,
    config: Mapping[str, Any],
) -> ScoringResult:
    """Calculate six normalized, JD-derived aspect scores and a weighted total."""

    weights = config["weights"]
    _validate_weights(weights)
    experience_overrides = config.get("experience_band_fits", {})
    experience_fit = float(
        experience_overrides.get(
            features.experience.band, features.experience.band_fit
        )
    )
    if not 0.0 <= experience_fit <= 1.0:
        raise ValueError("Configured experience-band fit must be in [0, 1]")

    ai_retrieval = (
        0.45 * features.production.production_fit
        + 0.25 * features.career.retrieval_career_fit
        + 0.20 * features.skills.retrieval_fit
        + 0.10 * features.career.evaluation_fit
    )
    python_product = (
        0.45 * features.skills.python_fit
        + 0.35 * features.career.product_engineering_fit
        + 0.20 * features.career.product_exposure
    )
    production_career = (
        0.65 * features.production.production_fit
        + 0.25 * features.career.product_exposure
        + 0.10 * features.career.retrieval_career_fit
    )

    signals = candidate.redrob_signals
    verification = (float(signals.verified_email) + float(signals.verified_phone)) / 2.0
    integrity_confidence = (
        0.45 * features.skills.corroboration_ratio
        + 0.30 * (signals.profile_completeness_score / 100.0)
        + 0.15 * verification
        + 0.10 * float(signals.linkedin_connected)
    )

    aspects = AspectScores(
        skills=min(ai_retrieval, 1.0) * 100.0,
        career=min(python_product, 1.0) * 100.0,
        production=min(production_career, 1.0) * 100.0,
        experience=experience_fit * 100.0,
        availability=features.availability.availability_fit * 100.0,
        integrity=min(integrity_confidence, 1.0) * 100.0,
    )
    base_score = sum(
        getattr(aspects, name) * float(weight) for name, weight in weights.items()
    )
    return ScoringResult(aspects=aspects, base_score=base_score)
