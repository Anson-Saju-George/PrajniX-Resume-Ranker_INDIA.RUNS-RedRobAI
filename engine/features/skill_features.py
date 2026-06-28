"""Corroborated skill features derived from typed candidate records."""

from __future__ import annotations

from dataclasses import dataclass

from engine.data import Candidate, is_missing
from engine.features.integrity_features import AI_TITLE_MARKERS


RETRIEVAL_SKILLS = frozenset(
    {
        "bm25",
        "embeddings",
        "information retrieval",
        "learning to rank",
        "rag",
        "recommendation systems",
        "semantic search",
        "sentence transformers",
        "vector search",
    }
)
VECTOR_DATABASE_SKILLS = frozenset(
    {"elasticsearch", "faiss", "milvus", "opensearch", "pgvector", "pinecone", "qdrant", "weaviate"}
)
NICE_TO_HAVE_SKILLS = frozenset(
    {"lora", "qlora", "peft", "fine-tuning llms", "mlops", "bentoml", "kubeflow"}
)

CAREER_RETRIEVAL_TERMS = (
    "embedding",
    "retrieval",
    "ranking",
    "recommendation",
    "recommender",
    "semantic search",
    "vector search",
    "search system",
)


@dataclass(frozen=True, slots=True)
class SkillEvidence:
    name: str
    duration_months: int
    endorsements: int
    assessment_score: float | None
    career_corroborated: bool
    title_coherent: bool
    support: float


@dataclass(frozen=True, slots=True)
class SkillFeatures:
    corroborated_retrieval: tuple[SkillEvidence, ...]
    corroborated_vector_databases: tuple[SkillEvidence, ...]
    corroborated_nice_to_have: tuple[SkillEvidence, ...]
    python: SkillEvidence | None
    relevant_skill_count: int
    corroborated_skill_count: int
    corroboration_ratio: float
    retrieval_fit: float
    python_fit: float


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(term in normalized for term in terms)


def _support_score(
    duration_months: int,
    endorsements: int,
    assessment: float | None,
    career_hit: bool,
    title_hit: bool,
) -> float:
    """Combine only available corroborators; missing assessments are neutral."""

    weighted = [
        (min(duration_months / 36.0, 1.0), 0.35),
        (min(endorsements / 20.0, 1.0), 0.15),
        (1.0 if career_hit else 0.0, 0.35),
        (1.0 if title_hit else 0.0, 0.15),
    ]
    if assessment is not None:
        weighted.append((assessment / 100.0, 0.20))
    numerator = sum(value * weight for value, weight in weighted)
    denominator = sum(weight for _, weight in weighted)
    return min(numerator / denominator, 1.0)


def extract(candidate: Candidate) -> SkillFeatures:
    """Extract skill evidence, requiring duration and career/title coherence."""

    career_text = " ".join(
        f"{role.title} {role.description}" for role in candidate.career_history
    ).casefold()
    title_coherent = _contains_any(candidate.profile.current_title, AI_TITLE_MARKERS)
    assessments = candidate.redrob_signals.skill_assessment_scores
    assessment_lookup = (
        {}
        if is_missing(assessments)
        else {name.casefold(): float(score) for name, score in assessments.items()}
    )

    retrieval: list[SkillEvidence] = []
    vector_databases: list[SkillEvidence] = []
    nice_to_have: list[SkillEvidence] = []
    python: SkillEvidence | None = None
    relevant_count = 0
    corroborated_count = 0

    for skill in candidate.skills:
        normalized = skill.name.casefold()
        category_relevant = normalized in (
            RETRIEVAL_SKILLS | VECTOR_DATABASE_SKILLS | NICE_TO_HAVE_SKILLS | {"python"}
        )
        if not category_relevant:
            continue
        relevant_count += 1

        assessment = assessment_lookup.get(normalized)
        career_hit = normalized in career_text
        if normalized in RETRIEVAL_SKILLS | VECTOR_DATABASE_SKILLS:
            career_hit = career_hit or _contains_any(career_text, CAREER_RETRIEVAL_TERMS)

        # A listed skill contributes only when it has real usage, an endorsement,
        # and either assessed or career evidence plus title/career coherence.
        corroborated = (
            skill.duration_months > 0
            and skill.endorsements > 0
            and (assessment is not None or career_hit)
            and (title_coherent or career_hit)
        )
        evidence = SkillEvidence(
            name=skill.name,
            duration_months=skill.duration_months,
            endorsements=skill.endorsements,
            assessment_score=assessment,
            career_corroborated=career_hit,
            title_coherent=title_coherent,
            support=_support_score(
                skill.duration_months,
                skill.endorsements,
                assessment,
                career_hit,
                title_coherent,
            ),
        )
        if not corroborated:
            continue

        corroborated_count += 1
        if normalized in RETRIEVAL_SKILLS:
            retrieval.append(evidence)
        if normalized in VECTOR_DATABASE_SKILLS:
            vector_databases.append(evidence)
        if normalized in NICE_TO_HAVE_SKILLS:
            nice_to_have.append(evidence)
        if normalized == "python":
            python = evidence

    strongest = sorted(
        [*retrieval, *vector_databases], key=lambda item: (-item.support, item.name)
    )
    coverage = min(len(strongest) / 4.0, 1.0)
    mean_support = (
        sum(item.support for item in strongest[:4]) / len(strongest[:4])
        if strongest
        else 0.0
    )
    retrieval_fit = 0.55 * mean_support + 0.45 * coverage
    python_fit = python.support if python is not None else 0.0

    return SkillFeatures(
        corroborated_retrieval=tuple(sorted(retrieval, key=lambda item: item.name)),
        corroborated_vector_databases=tuple(
            sorted(vector_databases, key=lambda item: item.name)
        ),
        corroborated_nice_to_have=tuple(sorted(nice_to_have, key=lambda item: item.name)),
        python=python,
        relevant_skill_count=relevant_count,
        corroborated_skill_count=corroborated_count,
        corroboration_ratio=(corroborated_count / relevant_count if relevant_count else 0.0),
        retrieval_fit=min(retrieval_fit, 1.0),
        python_fit=python_fit,
    )
