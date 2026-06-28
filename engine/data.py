"""Streaming, schema-validated access to the Redrob candidate dataset.

The public loader deliberately returns immutable typed objects rather than raw
dictionaries. This keeps missing-value handling consistent and gives later
pipeline stages one stable interface to candidate data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generator, Mapping, TypeAlias

import fastjsonschema


PathLike: TypeAlias = str | Path


# ---------------------------------------------------------------------------
# Missing-value contract
# ---------------------------------------------------------------------------


class _MissingValue:
    """Singleton marker used when the source explicitly means unavailable."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _MissingValue()
MissingType: TypeAlias = _MissingValue

def is_missing(value: object, field: str | None = None) -> bool:
    """Return whether a value represents unavailable data.

    The source schema uses ``-1`` only for unavailable GitHub activity and offer
    history. Empty lists/maps represent the two documented optional collections.
    The optional ``field`` argument is accepted so call sites can remain explicit,
    but the missing-value behavior is intentionally consistent for callers.
    """

    if value is MISSING or value is None:
        return True
    if value == -1:
        return True
    return isinstance(value, (list, dict, tuple)) and len(value) == 0


def _missing_if_sentinel(value: int | float, field: str) -> float | MissingType:
    """Normalize a documented numeric sentinel without changing real scores."""

    if is_missing(value, field):
        return MISSING
    return float(value)


# ---------------------------------------------------------------------------
# Typed candidate accessors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Profile:
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str


@dataclass(frozen=True, slots=True)
class CareerEntry:
    company: str
    title: str
    start_date: str
    end_date: str | None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str


@dataclass(frozen=True, slots=True)
class EducationEntry:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: str | None
    tier: str


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    proficiency: str
    endorsements: int
    duration_months: int


@dataclass(frozen=True, slots=True)
class Certification:
    name: str
    issuer: str
    year: int


@dataclass(frozen=True, slots=True)
class Language:
    language: str
    proficiency: str


@dataclass(frozen=True, slots=True)
class SalaryRange:
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class RedrobSignals:
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: Mapping[str, float] | MissingType
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float | MissingType
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float | MissingType
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


@dataclass(frozen=True, slots=True)
class Candidate:
    """Immutable candidate view consumed by all later engine stages."""

    candidate_id: str
    profile: Profile
    career_history: tuple[CareerEntry, ...]
    education: tuple[EducationEntry, ...]
    skills: tuple[Skill, ...]
    certifications: tuple[Certification, ...] | MissingType
    languages: tuple[Language, ...]
    redrob_signals: RedrobSignals


# ---------------------------------------------------------------------------
# Parse and schema errors with source-line context
# ---------------------------------------------------------------------------


class CandidateDataError(ValueError):
    """Base error for malformed or schema-invalid candidate input."""


class CandidateJSONError(CandidateDataError):
    """Raised when a JSONL line is blank or is not valid JSON."""


class CandidateSchemaError(CandidateDataError):
    """Raised when a parsed record does not satisfy the candidate schema."""


@lru_cache(maxsize=8)
def _compiled_validator(schema_path: str):
    """Compile and cache the small JSON Schema used for each streamed record."""

    path = Path(schema_path)
    with path.open("r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    return fastjsonschema.compile(schema)


def _schema_path_for(candidate_path: Path, schema_path: PathLike | None) -> Path:
    """Resolve an explicit schema or the schema beside the candidate file."""

    resolved = Path(schema_path) if schema_path is not None else candidate_path.with_name(
        "candidate_schema.json"
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"Candidate schema not found: {resolved}")
    return resolved.resolve()


# ---------------------------------------------------------------------------
# Raw-record to typed-object conversion
# ---------------------------------------------------------------------------


def _candidate_from_record(record: Mapping[str, Any]) -> Candidate:
    """Convert one already-validated mapping into the typed accessor model."""

    profile = record["profile"]
    signals = record["redrob_signals"]
    salary = signals["expected_salary_range_inr_lpa"]

    assessments = signals["skill_assessment_scores"]
    typed_assessments: Mapping[str, float] | MissingType
    if assessments:
        typed_assessments = MappingProxyType(
            {name: float(score) for name, score in assessments.items()}
        )
    else:
        typed_assessments = MISSING

    raw_certifications = record.get("certifications", [])
    typed_certifications: tuple[Certification, ...] | MissingType
    if raw_certifications:
        typed_certifications = tuple(
            Certification(name=item["name"], issuer=item["issuer"], year=item["year"])
            for item in raw_certifications
        )
    else:
        typed_certifications = MISSING

    return Candidate(
        candidate_id=record["candidate_id"],
        profile=Profile(
            anonymized_name=profile["anonymized_name"],
            headline=profile["headline"],
            summary=profile["summary"],
            location=profile["location"],
            country=profile["country"],
            years_of_experience=float(profile["years_of_experience"]),
            current_title=profile["current_title"],
            current_company=profile["current_company"],
            current_company_size=profile["current_company_size"],
            current_industry=profile["current_industry"],
        ),
        career_history=tuple(
            CareerEntry(
                company=item["company"],
                title=item["title"],
                start_date=item["start_date"],
                end_date=item["end_date"],
                duration_months=item["duration_months"],
                is_current=item["is_current"],
                industry=item["industry"],
                company_size=item["company_size"],
                description=item["description"],
            )
            for item in record["career_history"]
        ),
        education=tuple(
            EducationEntry(
                institution=item["institution"],
                degree=item["degree"],
                field_of_study=item["field_of_study"],
                start_year=item["start_year"],
                end_year=item["end_year"],
                grade=item.get("grade"),
                tier=item["tier"],
            )
            for item in record["education"]
        ),
        skills=tuple(
            Skill(
                name=item["name"],
                proficiency=item["proficiency"],
                endorsements=item["endorsements"],
                duration_months=item["duration_months"],
            )
            for item in record["skills"]
        ),
        certifications=typed_certifications,
        languages=tuple(
            Language(language=item["language"], proficiency=item["proficiency"])
            for item in record.get("languages", [])
        ),
        redrob_signals=RedrobSignals(
            profile_completeness_score=float(signals["profile_completeness_score"]),
            signup_date=signals["signup_date"],
            last_active_date=signals["last_active_date"],
            open_to_work_flag=signals["open_to_work_flag"],
            profile_views_received_30d=signals["profile_views_received_30d"],
            applications_submitted_30d=signals["applications_submitted_30d"],
            recruiter_response_rate=float(signals["recruiter_response_rate"]),
            avg_response_time_hours=float(signals["avg_response_time_hours"]),
            skill_assessment_scores=typed_assessments,
            connection_count=signals["connection_count"],
            endorsements_received=signals["endorsements_received"],
            notice_period_days=signals["notice_period_days"],
            expected_salary_range_inr_lpa=SalaryRange(
                minimum=float(salary["min"]), maximum=float(salary["max"])
            ),
            preferred_work_mode=signals["preferred_work_mode"],
            willing_to_relocate=signals["willing_to_relocate"],
            github_activity_score=_missing_if_sentinel(
                signals["github_activity_score"], "github_activity_score"
            ),
            search_appearance_30d=signals["search_appearance_30d"],
            saved_by_recruiters_30d=signals["saved_by_recruiters_30d"],
            interview_completion_rate=float(signals["interview_completion_rate"]),
            offer_acceptance_rate=_missing_if_sentinel(
                signals["offer_acceptance_rate"], "offer_acceptance_rate"
            ),
            verified_email=signals["verified_email"],
            verified_phone=signals["verified_phone"],
            linkedin_connected=signals["linkedin_connected"],
        ),
    )


# ---------------------------------------------------------------------------
# Public streaming API
# ---------------------------------------------------------------------------


def stream_candidates(
    path: PathLike, *, schema_path: PathLike | None = None
) -> Generator[Candidate, None, None]:
    """Yield validated candidates one at a time from a JSONL file.

    Only the current text line, parsed record, and typed candidate are resident
    in this generator. The dataset is never read with ``read()``, ``readlines()``,
    or a whole-file JSON operation.
    """

    candidate_path = Path(path)
    resolved_schema = _schema_path_for(candidate_path, schema_path)
    validator = _compiled_validator(str(resolved_schema))

    with candidate_path.open("r", encoding="utf-8") as candidate_file:
        for line_number, line in enumerate(candidate_file, start=1):
            if not line.strip():
                raise CandidateJSONError(f"Line {line_number}: blank JSONL record")

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CandidateJSONError(
                    f"Line {line_number}: invalid JSON: {error.msg}"
                ) from error

            try:
                validator(record)
            except fastjsonschema.JsonSchemaException as error:
                raise CandidateSchemaError(
                    f"Line {line_number}: schema validation failed: {error.message}"
                ) from error

            yield _candidate_from_record(record)


def load_sample(
    path: PathLike, n: int, *, schema_path: PathLike | None = None
) -> list[Candidate]:
    """Load at most ``n`` validated candidates while preserving stream safety."""

    if n < 0:
        raise ValueError("Sample size n must be non-negative")
    return list(islice(stream_candidates(path, schema_path=schema_path), n))
