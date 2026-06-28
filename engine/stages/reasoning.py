"""Field-grounded one-to-two sentence explanations for selected candidates."""

from __future__ import annotations

from engine.data import Candidate
from engine.stages.scoring import CandidateFeatures


def _fit_sentence(candidate: Candidate, features: CandidateFeatures) -> str:
    production_roles = features.production.relevant_roles
    if production_roles:
        role = sorted(
            production_roles,
            key=lambda item: (-item.duration_months, item.company, item.title),
        )[0]
        terms = list(dict.fromkeys([*role.system_terms, *role.production_terms]))[:3]
        return (
            f"{candidate.profile.years_of_experience:.1f} years of experience; "
            f"the {role.title} role at {role.company} explicitly mentions "
            f"{', '.join(terms)}."
        )

    corroborated = [
        *features.skills.corroborated_retrieval,
        *features.skills.corroborated_vector_databases,
    ]
    if corroborated:
        skill = sorted(corroborated, key=lambda item: (-item.support, item.name))[0]
        return (
            f"Current title is {candidate.profile.current_title} with "
            f"{candidate.profile.years_of_experience:.1f} years of experience; "
            f"{skill.name} is listed with {skill.duration_months} months and "
            f"{skill.endorsements} endorsements."
        )

    return (
        f"Current title is {candidate.profile.current_title} with "
        f"{candidate.profile.years_of_experience:.1f} years of experience at "
        f"{candidate.profile.current_company}."
    )


def build_reasoning(candidate: Candidate, features: CandidateFeatures) -> str:
    """Create exactly two factual sentences using only stored candidate fields."""

    signals = candidate.redrob_signals
    fit = _fit_sentence(candidate, features)
    availability = (
        f"Last active {signals.last_active_date}; open-to-work is "
        f"{'yes' if signals.open_to_work_flag else 'no'}, recruiter response rate is "
        f"{signals.recruiter_response_rate:.0%}, and notice period is "
        f"{signals.notice_period_days} days."
    )
    return f"{fit} {availability}"
