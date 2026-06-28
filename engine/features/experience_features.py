"""Experience-band features implementing the JD's asymmetric preference."""

from __future__ import annotations

from dataclasses import dataclass

from engine.data import Candidate


@dataclass(frozen=True, slots=True)
class ExperienceFeatures:
    years: float
    band: str
    band_fit: float


def extract(candidate: Candidate) -> ExperienceFeatures:
    years = candidate.profile.years_of_experience
    if 6.0 <= years <= 8.0:
        band, fit = "ideal_6_to_8", 1.0
    elif 5.0 <= years < 6.0:
        band, fit = "slightly_under", 0.78
    elif 4.0 <= years < 5.0:
        band, fit = "under", 0.52
    elif years < 4.0:
        band, fit = "well_under", max(0.10, years / 8.0)
    elif 8.0 < years <= 10.0:
        band, fit = "slightly_over", 0.90
    elif 10.0 < years <= 12.0:
        band, fit = "over", 0.78
    else:
        band, fit = "well_over", 0.65
    return ExperienceFeatures(years=years, band=band, band_fit=fit)
