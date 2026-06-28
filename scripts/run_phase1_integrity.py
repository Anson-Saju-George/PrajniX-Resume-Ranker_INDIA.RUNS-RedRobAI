"""Run the deterministic Phase 1 integrity shield and count-match checks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.data import stream_candidates  # noqa: E402
from engine.features.integrity_features import (  # noqa: E402
    AI_SKILL_STUFFER_THRESHOLD,
    FUTURE_CERTIFICATION_YEAR,
    HARD_FLAG_NAMES,
    LONG_NOTICE_DAYS,
    LOW_PROFILE_COMPLETENESS,
    LOW_RESPONSE_RATE,
    SEVERE_NOTICE_DAYS,
)
from engine.stages.integrity import run_integrity_checks  # noqa: E402


GROUND_TRUTH_EVENT_COUNTS = {
    "salary_range_reversed": 18_865,
    "last_active_before_signup": 7_496,
    "expert_skill_zero_months": 84,
    "future_dated_certification": 23,
    "skill_duration_impossible": 16_391,
}


def _event_increment(flag_name: str, evidence: dict[str, Any]) -> int:
    """Count affected rows/items for count-match flags, candidates otherwise."""

    if flag_name in {
        "expert_skill_zero_months",
        "future_dated_certification",
        "skill_duration_impossible",
    }:
        return int(evidence["count"])
    return 1


def _assert_result_evidence(result: dict[str, Any]) -> None:
    """Fail immediately if a returned flag lacks explainable evidence."""

    evidence = result["evidence"]
    for name in result["soft_flags"]:
        if name not in evidence or not evidence[name]:
            raise AssertionError(f"{result['candidate_id']}: {name} lacks evidence")

    if result["hard_suppress"] and not any(name in HARD_FLAG_NAMES for name in evidence):
        raise AssertionError(
            f"{result['candidate_id']}: hard suppression lacks hard-flag evidence"
        )

    for name, values in evidence.items():
        if not values:
            raise AssertionError(f"{result['candidate_id']}: {name} evidence is empty")


def analyze(
    candidates_path: Path,
    schema_path: Path,
    limit: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Stream candidates, execute checks, and aggregate deterministic counts."""

    started = time.perf_counter()
    candidates: Iterable[Any] = stream_candidates(candidates_path, schema_path=schema_path)
    if limit is not None:
        candidates = islice(candidates, limit)

    flag_candidate_counts: Counter[str] = Counter()
    flag_event_counts: Counter[str] = Counter()
    hard_suppress_count = 0
    soft_only_count = 0
    records_processed = 0
    sample_results: list[dict[str, Any]] = []

    for result in run_integrity_checks(candidates):
        records_processed += 1
        _assert_result_evidence(result)

        if records_processed <= 1_000:
            sample_results.append(result)
        if result["hard_suppress"]:
            hard_suppress_count += 1
        elif result["soft_flags"]:
            soft_only_count += 1

        for name, evidence in result["evidence"].items():
            flag_candidate_counts[name] += 1
            flag_event_counts[name] += _event_increment(name, evidence)

    elapsed = time.perf_counter() - started
    summary = {
        "phase": 1,
        "records_processed": records_processed,
        "json_errors": 0,
        "schema_errors": 0,
        "runtime_seconds": round(elapsed, 3),
        "hard_suppress_count": hard_suppress_count,
        "soft_only_count": soft_only_count,
        "flag_candidate_counts": dict(sorted(flag_candidate_counts.items())),
        "flag_event_counts": dict(sorted(flag_event_counts.items())),
        "policy": {
            "future_certification_year": FUTURE_CERTIFICATION_YEAR,
            "ai_skill_stuffer_threshold": AI_SKILL_STUFFER_THRESHOLD,
            "long_notice_days": LONG_NOTICE_DAYS,
            "severe_notice_days": SEVERE_NOTICE_DAYS,
            "low_recruiter_response_rate": LOW_RESPONSE_RATE,
            "low_profile_completeness_score": LOW_PROFILE_COMPLETENESS,
            "consulting_gate": "all career companies must be one of the six JD-named firms",
        },
        "evidence_validation": "PASS",
        "whole_file_loaded": False,
    }
    return summary, sample_results


def print_count_match(summary: dict[str, Any], full_run: bool) -> bool:
    """Print PASS/FAIL against documented counts without changing thresholds."""

    if not full_run:
        print("COUNT MATCH: skipped for bounded preflight")
        return True

    all_passed = True
    counts = summary["flag_event_counts"]
    for name, expected in GROUND_TRUTH_EVENT_COUNTS.items():
        observed = counts.get(name, 0)
        tolerance = max(2, round(expected * 0.005))
        passed = abs(observed - expected) <= tolerance
        all_passed &= passed
        print(
            f"{'PASS' if passed else 'FAIL'}: {name} observed={observed} "
            f"expected~{expected} tolerance=+/-{tolerance}"
        )
    return all_passed


def write_outputs(
    summary_path: Path,
    sample_path: Path,
    summary: dict[str, Any],
    sample_results: list[dict[str, Any]],
) -> None:
    """Write the two ignored Phase 1 artifacts after a successful full scan."""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with sample_path.open("w", encoding="utf-8", newline="\n") as sample_file:
        for result in sample_results:
            sample_file.write(json.dumps(result, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 integrity shield")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--summary", type=Path, default=Path("outputs/integrity_summary.json"))
    parser.add_argument(
        "--sample", type=Path, default=Path("outputs/integrity_flags_sample.jsonl")
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")

    summary, sample_results = analyze(args.candidates, args.schema, args.limit)
    full_run = args.limit is None
    count_match_passed = print_count_match(summary, full_run)

    if full_run and not args.no_write and count_match_passed:
        write_outputs(args.summary, args.sample, summary, sample_results)

    print(f"records_processed={summary['records_processed']}")
    print(f"runtime_seconds={summary['runtime_seconds']}")
    print(f"hard_suppress_count={summary['hard_suppress_count']}")
    print(f"soft_only_count={summary['soft_only_count']}")
    print(f"evidence_validation={summary['evidence_validation']}")
    if full_run and not args.no_write:
        print(f"summary_path={args.summary.resolve()}")
        print(f"sample_path={args.sample.resolve()}")

    return 0 if count_match_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
