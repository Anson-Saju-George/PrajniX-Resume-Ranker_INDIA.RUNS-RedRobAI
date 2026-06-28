"""Field-grounded one-to-two sentence explanations for selected candidates."""

from __future__ import annotations

from engine.data import Candidate
from engine.stages.scoring import CandidateFeatures


JD_IDEAL_EXPERIENCE_MIN = 6.0
JD_IDEAL_EXPERIENCE_MAX = 8.0
MATERIAL_OVERBAND_NOTE_THRESHOLD = 10.0


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


def _append_experience_adjustment(
    sentence: str, candidate: Candidate, features: CandidateFeatures
) -> str:
    """Explain an applied over-band adjustment using observed years and JD bounds."""

    years = candidate.profile.years_of_experience
    material_penalty_applied = (
        years > MATERIAL_OVERBAND_NOTE_THRESHOLD
        and features.experience.band_fit < 1.0
    )
    if not material_penalty_applied:
        return sentence

    sentence_without_period = sentence.removesuffix(".")
    return (
        f"{sentence_without_period}; over-band penalty applied "
        f"({years:.1f} yrs vs {JD_IDEAL_EXPERIENCE_MIN:.0f}-"
        f"{JD_IDEAL_EXPERIENCE_MAX:.0f} ideal)."
    )


def build_reasoning(candidate: Candidate, features: CandidateFeatures) -> str:
    """Create exactly two factual sentences using only stored candidate fields."""

    signals = candidate.redrob_signals
    fit = _append_experience_adjustment(
        _fit_sentence(candidate, features), candidate, features
    )
    availability = (
        f"Last active {signals.last_active_date}; open-to-work is "
        f"{'yes' if signals.open_to_work_flag else 'no'}, recruiter response rate is "
        f"{signals.recruiter_response_rate:.0%}, and notice period is "
        f"{signals.notice_period_days} days."
    )
    return f"{fit} {availability}"
