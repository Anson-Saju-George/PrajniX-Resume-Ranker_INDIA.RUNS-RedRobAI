"""Career trajectory and product-company evidence features."""

from __future__ import annotations

from dataclasses import dataclass

from engine.data import Candidate


SERVICE_COMPANIES = frozenset(
    {
        "accenture",
        "capgemini",
        "cognizant",
        "hcl",
        "infosys",
        "mindtree",
        "tcs",
        "tech mahindra",
        "wipro",
    }
)
PRODUCT_INDUSTRIES = frozenset(
    {
        "edtech",
        "e-commerce",
        "fintech",
        "food delivery",
        "internet",
        "marketplace",
        "saas",
        "software",
    }
)
RETRIEVAL_TERMS = (
    "embedding",
    "retrieval",
    "ranking",
    "recommendation",
    "recommender",
    "semantic search",
    "vector search",
    "search system",
    "hybrid search",
)
EVALUATION_TERMS = ("ndcg", "mrr", "map", "a/b test", "offline evaluation", "online evaluation")
PRODUCT_ENGINEERING_TERMS = (
    "python",
    "api",
    "backend",
    "service",
    "pipeline",
    "system design",
    "distributed system",
    "production code",
)


@dataclass(frozen=True, slots=True)
class CareerEvidence:
    company: str
    title: str
    duration_months: int
    is_current: bool
    matched_retrieval_terms: tuple[str, ...]
    matched_evaluation_terms: tuple[str, ...]
    matched_engineering_terms: tuple[str, ...]
    product_company_role: bool


@dataclass(frozen=True, slots=True)
class CareerFeatures:
    roles: tuple[CareerEvidence, ...]
    retrieval_role_count: int
    evaluation_role_count: int
    engineering_role_count: int
    product_role_count: int
    services_role_count: int
    product_exposure: float
    retrieval_career_fit: float
    evaluation_fit: float
    product_engineering_fit: float


def _matches(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    normalized = text.casefold()
    return tuple(term for term in terms if term in normalized)


def extract(candidate: Candidate) -> CareerFeatures:
    roles: list[CareerEvidence] = []
    product_months = 0
    service_months = 0

    for role in candidate.career_history:
        text = f"{role.title} {role.description}"
        company = role.company.strip().casefold()
        industry = role.industry.strip().casefold()
        is_service = company in SERVICE_COMPANIES or industry in {"it services", "consulting"}
        is_product = not is_service and industry in PRODUCT_INDUSTRIES
        if is_product:
            product_months += role.duration_months
        if is_service:
            service_months += role.duration_months

        roles.append(
            CareerEvidence(
                company=role.company,
                title=role.title,
                duration_months=role.duration_months,
                is_current=role.is_current,
                matched_retrieval_terms=_matches(text, RETRIEVAL_TERMS),
                matched_evaluation_terms=_matches(text, EVALUATION_TERMS),
                matched_engineering_terms=_matches(text, PRODUCT_ENGINEERING_TERMS),
                product_company_role=is_product,
            )
        )

    retrieval_roles = sum(bool(role.matched_retrieval_terms) for role in roles)
    evaluation_roles = sum(bool(role.matched_evaluation_terms) for role in roles)
    engineering_roles = sum(bool(role.matched_engineering_terms) for role in roles)
    product_roles = sum(role.product_company_role for role in roles)
    service_roles = sum(
        role.company.strip().casefold() in SERVICE_COMPANIES for role in candidate.career_history
    )
    relevant_months = product_months + service_months
    product_exposure = product_months / relevant_months if relevant_months else 0.0

    return CareerFeatures(
        roles=tuple(roles),
        retrieval_role_count=retrieval_roles,
        evaluation_role_count=evaluation_roles,
        engineering_role_count=engineering_roles,
        product_role_count=product_roles,
        services_role_count=service_roles,
        product_exposure=product_exposure,
        retrieval_career_fit=min(retrieval_roles / 2.0, 1.0),
        evaluation_fit=min(evaluation_roles / 1.0, 1.0),
        product_engineering_fit=min(engineering_roles / 2.0, 1.0),
    )
