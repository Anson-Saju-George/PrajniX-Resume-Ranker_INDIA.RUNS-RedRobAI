"""Pure, evidence-producing integrity checks for one typed candidate.

Every public check in this module is deterministic: it accepts a ``Candidate``
and returns a named flag plus the exact source values that triggered it. The
module does not rank candidates or assign fitness scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from engine.data import Candidate, is_missing


# Thresholds are policy constants derived from the Phase 1 brief. They are kept
# here, rather than buried in control flow, so later review can audit them.
FUTURE_CERTIFICATION_YEAR = 2026
AI_SKILL_STUFFER_THRESHOLD = 6
LONG_NOTICE_DAYS = 30
SEVERE_NOTICE_DAYS = 90
LOW_RESPONSE_RATE = 0.30
LOW_PROFILE_COMPLETENESS = 50.0
RECENT_CODING_MONTHS = 18


AI_SKILL_NAMES = frozenset(
    {
        "asr",
        "bentoml",
        "bm25",
        "cnn",
        "computer vision",
        "data science",
        "deep learning",
        "diffusion models",
        "elasticsearch",
        "embeddings",
        "faiss",
        "feature engineering",
        "fine-tuning llms",
        "gans",
        "haystack",
        "hugging face transformers",
        "image classification",
        "information retrieval",
        "kubeflow",
        "learning to rank",
        "llamaindex",
        "llms",
        "langchain",
        "lora",
        "machine learning",
        "milvus",
        "mlflow",
        "mlops",
        "nlp",
        "object detection",
        "opencv",
        "opensearch",
        "peft",
        "pgvector",
        "pinecone",
        "prompt engineering",
        "pytorch",
        "qdrant",
        "qlora",
        "rag",
        "recommendation systems",
        "reinforcement learning",
        "scikit-learn",
        "semantic search",
        "sentence transformers",
        "speech recognition",
        "statistical modeling",
        "tensorflow",
        "time series",
        "tts",
        "vector search",
        "weaviate",
        "weights & biases",
        "yolo",
    }
)

AI_TITLE_MARKERS = (
    "ai engineer",
    "artificial intelligence",
    "machine learning",
    "ml engineer",
    "data scientist",
    "applied scientist",
    "research scientist",
    "nlp",
    "recommendation",
    "recommender",
    "search engineer",
    "computer vision",
)

AI_CAREER_MARKERS = (
    "embedding",
    "retrieval",
    "ranking",
    "recommendation",
    "recommender",
    "semantic search",
    "vector search",
    "machine learning",
    "ml model",
    "nlp",
    "natural language",
    "data science",
)

PRODUCTION_MARKERS = (
    "production",
    "deployed",
    "deployment",
    "shipped",
    "real users",
    "at scale",
    "serving",
    "built",
    "implemented",
    "pipeline",
    "system",
    "service",
)

CODING_MARKERS = (
    "hands-on",
    "code",
    "coding",
    "built",
    "implemented",
    "developed",
    "python",
    "java",
    "api",
    "service",
    "pipeline",
    "deployed",
    "production",
)

LEADERSHIP_ONLY_TITLE_MARKERS = (
    "architect",
    "architecture",
    "tech lead",
    "technical lead",
    "engineering manager",
    "director of engineering",
    "head of engineering",
)

RESEARCH_MARKERS = (
    "research",
    "academic",
    "laboratory",
    " lab",
    "postdoc",
    "phd",
)

CONSULTING_FIRMS = frozenset(
    {"wipro", "tcs", "infosys", "accenture", "cognizant", "capgemini"}
)


@dataclass(frozen=True, slots=True)
class IntegrityFlag:
    """Result of one integrity feature check."""

    name: str
    triggered: bool
    evidence: dict[str, Any]


IntegrityCheck = Callable[[Candidate], IntegrityFlag]


def _flag(name: str, evidence: dict[str, Any] | None = None) -> IntegrityFlag:
    """Construct a flag while enforcing evidence for every triggered result."""

    triggered = evidence is not None
    return IntegrityFlag(name=name, triggered=triggered, evidence=evidence or {})


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(marker in normalized for marker in markers)


def _current_role(candidate: Candidate):
    """Return the current role when the source has exactly one such entry."""

    current = [role for role in candidate.career_history if role.is_current]
    return current[0] if len(current) == 1 else None


def expert_skill_zero_months(candidate: Candidate) -> IntegrityFlag:
    matches = [
        {
            "name": skill.name,
            "proficiency": skill.proficiency,
            "duration_months": skill.duration_months,
            "endorsements": skill.endorsements,
        }
        for skill in candidate.skills
        if skill.proficiency.casefold() == "expert" and skill.duration_months == 0
    ]
    return _flag(
        "expert_skill_zero_months",
        {"count": len(matches), "skills": matches} if matches else None,
    )


def future_dated_certification(candidate: Candidate) -> IntegrityFlag:
    if is_missing(candidate.certifications):
        return _flag("future_dated_certification")

    matches = [
        {"name": cert.name, "issuer": cert.issuer, "year": cert.year}
        for cert in candidate.certifications
        if cert.year > FUTURE_CERTIFICATION_YEAR
    ]
    return _flag(
        "future_dated_certification",
        {
            "reference_year": FUTURE_CERTIFICATION_YEAR,
            "count": len(matches),
            "certifications": matches,
        }
        if matches
        else None,
    )


def salary_range_reversed(candidate: Candidate) -> IntegrityFlag:
    salary = candidate.redrob_signals.expected_salary_range_inr_lpa
    if salary.minimum <= salary.maximum:
        return _flag("salary_range_reversed")
    return _flag(
        "salary_range_reversed",
        {"minimum_inr_lpa": salary.minimum, "maximum_inr_lpa": salary.maximum},
    )


def last_active_before_signup(candidate: Candidate) -> IntegrityFlag:
    signals = candidate.redrob_signals
    signup = date.fromisoformat(signals.signup_date)
    last_active = date.fromisoformat(signals.last_active_date)
    if last_active >= signup:
        return _flag("last_active_before_signup")
    return _flag(
        "last_active_before_signup",
        {"signup_date": signals.signup_date, "last_active_date": signals.last_active_date},
    )


def impossible_skill_duration(candidate: Candidate) -> IntegrityFlag:
    maximum_months = candidate.profile.years_of_experience * 12 + 12
    matches = [
        {
            "name": skill.name,
            "duration_months": skill.duration_months,
            "maximum_allowed_months": maximum_months,
        }
        for skill in candidate.skills
        if skill.duration_months > maximum_months
    ]
    return _flag(
        "skill_duration_impossible",
        {
            "years_of_experience": candidate.profile.years_of_experience,
            "maximum_allowed_months": maximum_months,
            "count": len(matches),
            "skills": matches,
        }
        if matches
        else None,
    )


def long_notice_period(candidate: Candidate) -> IntegrityFlag:
    days = candidate.redrob_signals.notice_period_days
    if days <= LONG_NOTICE_DAYS:
        return _flag("long_notice_period")
    return _flag(
        "long_notice_period",
        {
            "notice_period_days": days,
            "threshold_days": LONG_NOTICE_DAYS,
            "severity": "severe" if days > SEVERE_NOTICE_DAYS else "long",
            "severe_threshold_days": SEVERE_NOTICE_DAYS,
        },
    )


def low_recruiter_response(candidate: Candidate) -> IntegrityFlag:
    rate = candidate.redrob_signals.recruiter_response_rate
    if rate >= LOW_RESPONSE_RATE:
        return _flag("low_recruiter_response_rate")
    return _flag(
        "low_recruiter_response_rate",
        {"recruiter_response_rate": rate, "threshold_exclusive": LOW_RESPONSE_RATE},
    )


def low_profile_completeness(candidate: Candidate) -> IntegrityFlag:
    completeness = candidate.redrob_signals.profile_completeness_score
    if completeness >= LOW_PROFILE_COMPLETENESS:
        return _flag("low_profile_completeness")
    return _flag(
        "low_profile_completeness",
        {
            "profile_completeness_score": completeness,
            "threshold_exclusive": LOW_PROFILE_COMPLETENESS,
        },
    )


def keyword_stuffer(candidate: Candidate) -> IntegrityFlag:
    """Detect the JD's non-AI-title plus unsupported AI-keyword contradiction."""

    title = candidate.profile.current_title
    if _contains_any(title, AI_TITLE_MARKERS):
        return _flag("keyword_stuffer")

    ai_skills = sorted(
        skill.name for skill in candidate.skills if skill.name.casefold() in AI_SKILL_NAMES
    )
    if len(ai_skills) < AI_SKILL_STUFFER_THRESHOLD:
        return _flag("keyword_stuffer")

    corroborating_roles = []
    for role in candidate.career_history:
        career_text = f"{role.title} {role.description}"
        if _contains_any(career_text, AI_CAREER_MARKERS) and _contains_any(
            career_text, PRODUCTION_MARKERS
        ):
            corroborating_roles.append(
                {"company": role.company, "title": role.title, "start_date": role.start_date}
            )

    if corroborating_roles:
        return _flag("keyword_stuffer")
    return _flag(
        "keyword_stuffer",
        {
            "current_title": title,
            "ai_skill_count": len(ai_skills),
            "ai_skill_threshold": AI_SKILL_STUFFER_THRESHOLD,
            "ai_skills": ai_skills,
            "corroborating_production_ai_roles": corroborating_roles,
        },
    )


def consulting_only_trajectory(candidate: Candidate) -> IntegrityFlag:
    """Gate only when the entire career is at the six JD-named service firms."""

    companies = [role.company for role in candidate.career_history]
    normalized = [company.strip().casefold() for company in companies]
    if not normalized or not all(company in CONSULTING_FIRMS for company in normalized):
        return _flag("consulting_only_trajectory")

    return _flag(
        "consulting_only_trajectory",
        {
            "career_companies": companies,
            "named_consulting_firms": sorted(CONSULTING_FIRMS),
            "whole_history_checked": True,
            "prior_non_consulting_product_exception_found": False,
        },
    )


def pure_research_without_production(candidate: Candidate) -> IntegrityFlag:
    role_evidence = []
    for role in candidate.career_history:
        text = f"{role.title} {role.industry} {role.description}"
        is_research = _contains_any(text, RESEARCH_MARKERS)
        has_production = _contains_any(text, PRODUCTION_MARKERS)
        role_evidence.append(
            {
                "company": role.company,
                "title": role.title,
                "research_markers_present": is_research,
                "production_markers_present": has_production,
            }
        )

    all_research = bool(role_evidence) and all(
        role["research_markers_present"] for role in role_evidence
    )
    any_production = any(role["production_markers_present"] for role in role_evidence)
    if not all_research or any_production:
        return _flag("pure_research_without_production")
    return _flag(
        "pure_research_without_production",
        {"career_roles": role_evidence, "production_deployment_evidence_found": False},
    )


def no_recent_production_code(candidate: Candidate) -> IntegrityFlag:
    current = _current_role(candidate)
    if current is None or current.duration_months < RECENT_CODING_MONTHS:
        return _flag("no_recent_production_code")

    leadership_title = _contains_any(current.title, LEADERSHIP_ONLY_TITLE_MARKERS)
    coding_evidence = _contains_any(current.description, CODING_MARKERS)
    if not leadership_title or coding_evidence:
        return _flag("no_recent_production_code")

    return _flag(
        "no_recent_production_code",
        {
            "current_title": current.title,
            "company": current.company,
            "current_role_duration_months": current.duration_months,
            "required_recent_months": RECENT_CODING_MONTHS,
            "coding_evidence_found": False,
            "description_excerpt": current.description[:300],
        },
    )


BASE_HARD_CHECKS: tuple[IntegrityCheck, ...] = (
    expert_skill_zero_months,
    future_dated_certification,
    keyword_stuffer,
    consulting_only_trajectory,
    pure_research_without_production,
    no_recent_production_code,
)


def multiple_severe_contradictions(candidate: Candidate) -> IntegrityFlag:
    """Identify candidates triggering at least two independent hard checks."""

    triggered = [result.name for check in BASE_HARD_CHECKS if (result := check(candidate)).triggered]
    if len(triggered) < 2:
        return _flag("multiple_severe_contradictions")
    return _flag(
        "multiple_severe_contradictions",
        {"count": len(triggered), "triggered_severe_checks": triggered},
    )


HARD_CHECKS: tuple[IntegrityCheck, ...] = BASE_HARD_CHECKS + (
    multiple_severe_contradictions,
)

SOFT_CHECKS: tuple[IntegrityCheck, ...] = (
    salary_range_reversed,
    last_active_before_signup,
    impossible_skill_duration,
    long_notice_period,
    low_recruiter_response,
    low_profile_completeness,
)

HARD_FLAG_NAMES = frozenset(
    {
        "expert_skill_zero_months",
        "future_dated_certification",
        "keyword_stuffer",
        "consulting_only_trajectory",
        "pure_research_without_production",
        "no_recent_production_code",
        "multiple_severe_contradictions",
    }
)

SOFT_FLAG_NAMES = frozenset(
    {
        "salary_range_reversed",
        "last_active_before_signup",
        "skill_duration_impossible",
        "long_notice_period",
        "low_recruiter_response_rate",
        "low_profile_completeness",
    }
)
