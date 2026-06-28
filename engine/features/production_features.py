"""Production deployment and shipped search/ranking system evidence."""

from __future__ import annotations

from dataclasses import dataclass

from engine.data import Candidate


PRODUCTION_TERMS = (
    "production",
    "deployed",
    "deployment",
    "shipped",
    "real users",
    "at scale",
    "serving",
    "on-call",
)
SYSTEM_TERMS = (
    "ranking",
    "retrieval",
    "recommendation",
    "recommender",
    "search",
    "embedding",
    "vector",
)
SCALE_TERMS = ("million", "billion", "gb", "tb", "qps", "latency", "real-time", "at scale")


@dataclass(frozen=True, slots=True)
class ProductionEvidence:
    company: str
    title: str
    duration_months: int
    production_terms: tuple[str, ...]
    system_terms: tuple[str, ...]
    scale_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProductionFeatures:
    relevant_roles: tuple[ProductionEvidence, ...]
    production_role_count: int
    shipped_search_system_count: int
    scale_evidence_count: int
    production_fit: float


def _matches(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    normalized = text.casefold()
    return tuple(term for term in terms if term in normalized)


def extract(candidate: Candidate) -> ProductionFeatures:
    evidence: list[ProductionEvidence] = []
    shipped_systems = 0
    scale_count = 0

    for role in candidate.career_history:
        text = f"{role.title} {role.description}"
        production = _matches(text, PRODUCTION_TERMS)
        systems = _matches(text, SYSTEM_TERMS)
        scale = _matches(text, SCALE_TERMS)
        if production and systems:
            evidence.append(
                ProductionEvidence(
                    company=role.company,
                    title=role.title,
                    duration_months=role.duration_months,
                    production_terms=production,
                    system_terms=systems,
                    scale_terms=scale,
                )
            )
            shipped_systems += 1
            scale_count += bool(scale)

    role_coverage = min(len(evidence) / 2.0, 1.0)
    duration_coverage = min(sum(role.duration_months for role in evidence) / 48.0, 1.0)
    scale_bonus = min(scale_count / 2.0, 1.0)
    production_fit = 0.45 * role_coverage + 0.40 * duration_coverage + 0.15 * scale_bonus

    return ProductionFeatures(
        relevant_roles=tuple(evidence),
        production_role_count=len(evidence),
        shipped_search_system_count=shipped_systems,
        scale_evidence_count=scale_count,
        production_fit=min(production_fit, 1.0),
    )
