"""Sentinel-aware behavioral availability features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from engine.data import Candidate, is_missing


@dataclass(frozen=True, slots=True)
class AvailabilityFeatures:
    last_active_date: str
    days_since_active: int
    recency_fit: float
    open_to_work: bool
    recruiter_response_rate: float
    interview_completion_rate: float
    applications_30d: int
    github_activity_score: float | None
    offer_acceptance_rate: float | None
    availability_fit: float


def extract(candidate: Candidate, reference_date: date = date(2026, 6, 1)) -> AvailabilityFeatures:
    signals = candidate.redrob_signals
    last_active = date.fromisoformat(signals.last_active_date)
    days_since_active = max((reference_date - last_active).days, 0)
    if days_since_active <= 30:
        recency = 1.0
    elif days_since_active <= 90:
        recency = 0.80
    elif days_since_active <= 180:
        recency = 0.50
    else:
        recency = 0.20

    application_fit = min(signals.applications_submitted_30d / 5.0, 1.0)
    availability = (
        0.30 * recency
        + 0.25 * float(signals.open_to_work_flag)
        + 0.25 * signals.recruiter_response_rate
        + 0.15 * signals.interview_completion_rate
        + 0.05 * application_fit
    )
    github = signals.github_activity_score
    offer = signals.offer_acceptance_rate

    return AvailabilityFeatures(
        last_active_date=signals.last_active_date,
        days_since_active=days_since_active,
        recency_fit=recency,
        open_to_work=signals.open_to_work_flag,
        recruiter_response_rate=signals.recruiter_response_rate,
        interview_completion_rate=signals.interview_completion_rate,
        applications_30d=signals.applications_submitted_30d,
        github_activity_score=None if is_missing(github) else float(github),
        offer_acceptance_rate=None if is_missing(offer) else float(offer),
        availability_fit=min(availability, 1.0),
    )
