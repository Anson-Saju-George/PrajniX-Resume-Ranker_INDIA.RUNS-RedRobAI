"""Create a blind, mixed-slice candidate set for human relevance labeling.

The sampler streams the source dataset through the typed loader, uses frozen
Phase 1 checks and field-level Phase 2 features only for slice membership, and
never writes model scores or source ranks to either judge-set CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.data import Candidate, stream_candidates  # noqa: E402
from engine.features import availability_features, career_features, production_features  # noqa: E402
from engine.features.integrity_features import AI_SKILL_NAMES, AI_TITLE_MARKERS  # noqa: E402
from engine.features.skill_features import RETRIEVAL_SKILLS, VECTOR_DATABASE_SKILLS  # noqa: E402
from engine.stages.integrity import check_candidate_integrity  # noqa: E402


SLICE_NAMES = (
    "baseline_top",
    "jd_keyword_strong",
    "structured_strong_text_weak",
    "sample_submission_suspected_negatives",
    "integrity_flagged",
    "over_band",
    "high_availability_low_fit",
    "low_availability_high_fit",
    "cv_speech_robotics_no_nlp",
    "random_anchor",
)

BLIND_HEADER = ("judge_id", "candidate_id", "slice", "redacted_evidence", "label")
KEY_HEADER = ("candidate_id", "slice")
DEFAULT_SEED = 20260628

CV_SPEECH_ROBOTICS_MARKERS = (
    "computer vision",
    "vision engineer",
    "speech",
    "robotics",
    "robotics engineer",
)
CV_SPEECH_ROBOTICS_SKILLS = frozenset(
    {
        "asr",
        "cnn",
        "computer vision",
        "image classification",
        "object detection",
        "opencv",
        "robotics",
        "speech recognition",
        "tts",
        "yolo",
    }
)
NLP_IR_SKILLS = RETRIEVAL_SKILLS | VECTOR_DATABASE_SKILLS | {
    "nlp",
    "llms",
    "natural language processing",
}


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    """Compact evidence retained for a bounded judge-set pool."""

    candidate_id: str
    redacted_evidence: str
    anonymized_name: str


@dataclass(frozen=True, slots=True)
class SelectedCandidate:
    snapshot: CandidateSnapshot
    slice_name: str


class Reservoir:
    """Fixed-seed bounded reservoir for a potentially large slice."""

    def __init__(self, capacity: int, seed: str) -> None:
        self.capacity = capacity
        self.random = random.Random(seed)
        self.seen = 0
        self.items: list[CandidateSnapshot] = []

    def consider(self, factory: Callable[[], CandidateSnapshot]) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(factory())
            return
        index = self.random.randrange(self.seen)
        if index < self.capacity:
            self.items[index] = factory()


def _find_bundle_file(name: str) -> Path:
    matches = [
        path
        for path in REPOSITORY_ROOT.rglob(name)
        if "__MACOSX" not in path.parts and path.is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}; found {len(matches)}")
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(value: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _redacted_snapshot(candidate: Candidate) -> CandidateSnapshot:
    """Build compact evidence exclusively from observable candidate fields."""

    relevant = AI_SKILL_NAMES | {"python"}
    skills = sorted(
        candidate.skills,
        key=lambda skill: (
            skill.name.casefold() not in relevant,
            -skill.duration_months,
            -skill.endorsements,
            skill.name,
        ),
    )[:8]
    skill_text = ", ".join(
        f"{skill.name} ({skill.duration_months}m, {skill.endorsements} endorsements)"
        for skill in skills
    )

    roles = sorted(
        candidate.career_history,
        key=lambda role: (
            not role.is_current,
            -int(role.start_date.replace("-", "")),
        ),
    )[:3]
    career_text = " / ".join(
        f"{role.title} at {role.company}, {role.duration_months}m: "
        f"{_clean_text(role.description, 170)}"
        for role in roles
    )
    signals = candidate.redrob_signals
    evidence = (
        f"Title: {candidate.profile.current_title} at {candidate.profile.current_company}. "
        f"Experience: {candidate.profile.years_of_experience:.1f} years. "
        f"Location: {candidate.profile.location}, {candidate.profile.country}. "
        f"Skills: {skill_text or 'none listed'}. "
        f"Career: {career_text}. "
        f"Activity: last active {signals.last_active_date}; open to work "
        f"{'yes' if signals.open_to_work_flag else 'no'}; recruiter response "
        f"{signals.recruiter_response_rate:.0%}; notice {signals.notice_period_days} days; "
        f"profile completeness {signals.profile_completeness_score:.0f}%."
    )

    # Names are never intentionally inserted, and this replacement protects
    # against an accidental occurrence inside free-text source fields.
    if candidate.profile.anonymized_name:
        evidence = re.sub(
            re.escape(candidate.profile.anonymized_name),
            "[REDACTED]",
            evidence,
            flags=re.IGNORECASE,
        )
    return CandidateSnapshot(
        candidate_id=candidate.candidate_id,
        redacted_evidence=evidence,
        anonymized_name=candidate.profile.anonymized_name,
    )


def _load_baseline_order(path: Path) -> list[str]:
    """Read only candidate order; model scores are deliberately discarded."""

    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    return [row["candidate_id"] for row in sorted(rows, key=lambda row: int(row["rank"]))]


def _load_sample_submission_order(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return [row["candidate_id"] for row in csv.DictReader(source)]


def _quota_by_slice(total: int) -> dict[str, int]:
    if total < len(SLICE_NAMES):
        raise ValueError(f"--count must be at least {len(SLICE_NAMES)}")
    base, remainder = divmod(total, len(SLICE_NAMES))
    return {
        name: base + (1 if index < remainder else 0)
        for index, name in enumerate(SLICE_NAMES)
    }


def _is_ai_title(title: str) -> bool:
    normalized = title.casefold()
    return any(marker in normalized for marker in AI_TITLE_MARKERS)


def _ordered_pool(items: list[CandidateSnapshot], seed: int, name: str) -> list[CandidateSnapshot]:
    ordered = sorted(items, key=lambda item: item.candidate_id)
    random.Random(f"{seed}:{name}:selection").shuffle(ordered)
    return ordered


def _take_unique(
    pool: list[CandidateSnapshot],
    count: int,
    used: set[str],
) -> list[CandidateSnapshot]:
    selected: list[CandidateSnapshot] = []
    for snapshot in pool:
        if snapshot.candidate_id in used:
            continue
        selected.append(snapshot)
        used.add(snapshot.candidate_id)
        if len(selected) == count:
            break
    return selected


def sample_candidates(
    candidates_path: Path,
    baseline_path: Path,
    sample_submission_path: Path,
    total: int,
    seed: int,
) -> tuple[list[SelectedCandidate], set[str], dict[str, int]]:
    """Stream the pool once and collect bounded, deterministic slice reservoirs."""

    quotas = _quota_by_slice(total)
    capacity = max(200, max(quotas.values()) * 25)
    baseline_order = _load_baseline_order(baseline_path)
    baseline_ids = set(baseline_order)
    sample_order = _load_sample_submission_order(sample_submission_path)
    sample_ids = set(sample_order)

    pools = {
        name: Reservoir(capacity, f"{seed}:{name}")
        for name in SLICE_NAMES
        if name not in {"baseline_top", "sample_submission_suspected_negatives", "integrity_flagged"}
    }
    hard_integrity = Reservoir(capacity, f"{seed}:integrity:hard")
    soft_integrity = Reservoir(capacity, f"{seed}:integrity:soft")
    baseline_snapshots: dict[str, CandidateSnapshot] = {}
    sample_snapshots: dict[str, CandidateSnapshot] = {}
    source_ids: set[str] = set()

    for candidate in stream_candidates(candidates_path):
        source_ids.add(candidate.candidate_id)
        snapshot: CandidateSnapshot | None = None

        def get_snapshot() -> CandidateSnapshot:
            nonlocal snapshot
            if snapshot is None:
                snapshot = _redacted_snapshot(candidate)
            return snapshot

        integrity = check_candidate_integrity(candidate)
        career = career_features.extract(candidate)
        production = production_features.extract(candidate)
        availability = availability_features.extract(candidate)
        skill_names = {skill.name.casefold() for skill in candidate.skills}
        ai_skill_count = sum(name in AI_SKILL_NAMES for name in skill_names)
        explicit_jd_skill_count = sum(
            name in (RETRIEVAL_SKILLS | VECTOR_DATABASE_SKILLS) for name in skill_names
        )

        if candidate.candidate_id in baseline_ids:
            baseline_snapshots[candidate.candidate_id] = get_snapshot()
        if candidate.candidate_id in sample_ids and not _is_ai_title(
            candidate.profile.current_title
        ):
            sample_snapshots[candidate.candidate_id] = get_snapshot()

        if ai_skill_count >= 8:
            pools["jd_keyword_strong"].consider(get_snapshot)
        if (
            production.production_role_count >= 1
            and career.retrieval_role_count >= 1
            # Tier-5-style profiles may still list general ML skills; the key
            # distinction is sparse explicit retrieval/vector keyword coverage.
            and explicit_jd_skill_count <= 1
        ):
            pools["structured_strong_text_weak"].consider(get_snapshot)
        if integrity["hard_suppress"]:
            hard_integrity.consider(get_snapshot)
        elif integrity["soft_flags"]:
            soft_integrity.consider(get_snapshot)
        if candidate.profile.years_of_experience >= 12.0:
            pools["over_band"].consider(get_snapshot)
        if (
            availability.availability_fit >= 0.75
            and production.production_role_count == 0
            and career.retrieval_role_count == 0
            and ai_skill_count <= 2
        ):
            pools["high_availability_low_fit"].consider(get_snapshot)
        if (
            production.production_role_count >= 1
            and career.retrieval_role_count >= 1
            and (
                not candidate.redrob_signals.open_to_work_flag
                or candidate.redrob_signals.notice_period_days > 90
                or candidate.redrob_signals.recruiter_response_rate < 0.30
            )
        ):
            pools["low_availability_high_fit"].consider(get_snapshot)

        title = candidate.profile.current_title.casefold()
        primary_cv_speech_robotics = (
            any(marker in title for marker in CV_SPEECH_ROBOTICS_MARKERS)
            or len(skill_names & CV_SPEECH_ROBOTICS_SKILLS) >= 3
        )
        has_nlp_ir = bool(skill_names & NLP_IR_SKILLS) or career.retrieval_role_count > 0
        if primary_cv_speech_robotics and not has_nlp_ir:
            pools["cv_speech_robotics_no_nlp"].consider(get_snapshot)
        pools["random_anchor"].consider(get_snapshot)

    ordered_pools: dict[str, list[CandidateSnapshot]] = {
        "baseline_top": [
            baseline_snapshots[candidate_id]
            for candidate_id in baseline_order
            if candidate_id in baseline_snapshots
        ],
        "sample_submission_suspected_negatives": [
            sample_snapshots[candidate_id]
            for candidate_id in sample_order
            if candidate_id in sample_snapshots
        ],
    }
    for name, reservoir in pools.items():
        ordered_pools[name] = _ordered_pool(reservoir.items, seed, name)

    integrity_quota = quotas["integrity_flagged"]
    hard_quota = (integrity_quota + 1) // 2
    soft_quota = integrity_quota - hard_quota
    hard_pool = _ordered_pool(hard_integrity.items, seed, "integrity_hard")
    soft_pool = _ordered_pool(soft_integrity.items, seed, "integrity_soft")

    used: set[str] = set()
    selected: list[SelectedCandidate] = []
    selection_order = (
        "baseline_top",
        "sample_submission_suspected_negatives",
        "structured_strong_text_weak",
        "jd_keyword_strong",
        "over_band",
        "high_availability_low_fit",
        "low_availability_high_fit",
        "cv_speech_robotics_no_nlp",
    )
    for name in selection_order:
        chosen = _take_unique(ordered_pools[name], quotas[name], used)
        if len(chosen) != quotas[name]:
            raise RuntimeError(f"Slice {name} has only {len(chosen)} unique candidates")
        selected.extend(SelectedCandidate(item, name) for item in chosen)

    hard_selected = _take_unique(hard_pool, hard_quota, used)
    soft_selected = _take_unique(soft_pool, soft_quota, used)
    integrity_selected = [*hard_selected, *soft_selected]
    if len(integrity_selected) < integrity_quota:
        leftovers = [*hard_pool, *soft_pool]
        integrity_selected.extend(
            _take_unique(leftovers, integrity_quota - len(integrity_selected), used)
        )
    if len(integrity_selected) != integrity_quota:
        raise RuntimeError("Integrity slice could not satisfy its hard/soft quota")
    selected.extend(
        SelectedCandidate(item, "integrity_flagged") for item in integrity_selected
    )

    random_selected = _take_unique(
        ordered_pools["random_anchor"], quotas["random_anchor"], used
    )
    if len(random_selected) != quotas["random_anchor"]:
        raise RuntimeError("Random anchor slice could not satisfy its quota")
    selected.extend(SelectedCandidate(item, "random_anchor") for item in random_selected)

    if len(selected) != total:
        raise RuntimeError(f"Expected {total} selected candidates; got {len(selected)}")
    return selected, source_ids, quotas


def write_judge_files(
    blind_path: Path,
    key_path: Path,
    selected: list[SelectedCandidate],
    seed: int,
) -> list[SelectedCandidate]:
    """Shuffle once, assign blind IDs, and write row-aligned files."""

    shuffled = list(selected)
    random.Random(seed).shuffle(shuffled)
    blind_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    with blind_path.open("w", encoding="utf-8", newline="") as blind_file, key_path.open(
        "w", encoding="utf-8", newline=""
    ) as key_file:
        blind_writer = csv.writer(blind_file)
        key_writer = csv.writer(key_file)
        blind_writer.writerow(BLIND_HEADER)
        key_writer.writerow(KEY_HEADER)
        for index, item in enumerate(shuffled, start=1):
            blind_writer.writerow(
                (
                    f"JUDGE_{index:03d}",
                    item.snapshot.candidate_id,
                    item.slice_name,
                    item.snapshot.redacted_evidence,
                    "",
                )
            )
            key_writer.writerow((item.snapshot.candidate_id, item.slice_name))
    return shuffled


def smoke_test(
    blind_path: Path,
    key_path: Path,
    shuffled: list[SelectedCandidate],
    pre_shuffle: list[SelectedCandidate],
    source_ids: set[str],
    before_hash: str,
    after_hash: str,
) -> bool:
    with blind_path.open("r", encoding="utf-8", newline="") as blind_file:
        blind_reader = csv.DictReader(blind_file)
        blind_rows = list(blind_reader)
        blind_header = tuple(blind_reader.fieldnames or ())
    with key_path.open("r", encoding="utf-8", newline="") as key_file:
        key_reader = csv.DictReader(key_file)
        key_rows = list(key_reader)
        key_header = tuple(key_reader.fieldnames or ())

    no_score_columns = blind_header == BLIND_HEADER and not {
        "score",
        "rank",
        "model_score",
    }.intersection(blind_header)
    shuffled_order = [item.snapshot.candidate_id for item in shuffled]
    original_order = [item.snapshot.candidate_id for item in pre_shuffle]
    order_is_blind = (
        shuffled_order != original_order
        and shuffled_order != sorted(shuffled_order)
        and any(
            blind_rows[index - 1]["slice"] != blind_rows[index]["slice"]
            for index in range(1, len(blind_rows))
        )
    )
    slice_counts = Counter(row["slice"] for row in blind_rows)
    all_slices = all(slice_counts[name] > 0 for name in SLICE_NAMES)
    labels_blank = all(row["label"] == "" for row in blind_rows)

    private_names = {
        item.snapshot.candidate_id: item.snapshot.anonymized_name.casefold()
        for item in shuffled
    }
    readable_and_redacted = all(
        row["redacted_evidence"].startswith("Title:")
        and "Career:" in row["redacted_evidence"]
        and "Activity:" in row["redacted_evidence"]
        and (
            not private_names[row["candidate_id"]]
            or private_names[row["candidate_id"]]
            not in row["redacted_evidence"].casefold()
        )
        for row in blind_rows
    )
    ids = [row["candidate_id"] for row in blind_rows]
    ids_valid = len(ids) == len(set(ids)) and all(item in source_ids for item in ids)
    key_matches = (
        key_header == KEY_HEADER
        and len(key_rows) == len(blind_rows)
        and all(
            key_row == {"candidate_id": blind_row["candidate_id"], "slice": blind_row["slice"]}
            for key_row, blind_row in zip(key_rows, blind_rows)
        )
    )
    dataset_untouched = before_hash == after_hash

    checks = {
        "blind file has no score/rank column and fixed-seed order is shuffled": (
            no_score_columns and order_is_blind
        ),
        "every requested slice is represented": all_slices,
        "label column is entirely blank": labels_blank,
        "redacted evidence is readable and excludes anonymized_name": readable_and_redacted,
        "candidate IDs are unique/real and key matches row-for-row": ids_valid and key_matches,
        "candidates.jsonl is unmodified": dataset_untouched,
    }
    print("PHASE 4 SMOKE TEST")
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {label}")
    print("PER-SLICE COUNTS")
    for name in SLICE_NAMES:
        print(f"{name}: {slice_counts[name]}")
    print(f"candidate_sha256_before={before_hash}")
    print(f"candidate_sha256_after={after_hash}")
    return all(checks.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blind mixed-slice judge set")
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--baseline", type=Path, default=Path("outputs/submission_baseline.csv")
    )
    parser.add_argument(
        "--blind", type=Path, default=Path("outputs/judge_set_blind.csv")
    )
    parser.add_argument("--key", type=Path, default=Path("outputs/judge_set_key.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates_path = _find_bundle_file("candidates.jsonl")
    sample_submission_path = _find_bundle_file("sample_submission.csv")
    if not args.baseline.is_file():
        raise FileNotFoundError(args.baseline)

    before_hash = _sha256(candidates_path)
    selected, source_ids, quotas = sample_candidates(
        candidates_path,
        args.baseline,
        sample_submission_path,
        args.count,
        args.seed,
    )
    shuffled = write_judge_files(args.blind, args.key, selected, args.seed)
    after_hash = _sha256(candidates_path)
    passed = smoke_test(
        args.blind,
        args.key,
        shuffled,
        selected,
        source_ids,
        before_hash,
        after_hash,
    )
    print(f"configured_count={args.count}")
    print(f"fixed_seed={args.seed}")
    print(f"quota_by_slice={quotas}")
    print(f"blind={args.blind.resolve()}")
    print(f"key={args.key.resolve()}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
