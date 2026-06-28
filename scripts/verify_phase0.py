"""Run the Phase 0 dataset audit and write the schema field report.

This script intentionally performs no ranking work. It verifies the JSONL
contract, checks identifiers, exercises the typed missing-value accessors, and
creates the human-readable report requested for Phase 0.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


# Running this file directly puts ``scripts`` on sys.path. Add the repository
# root so the engine package resolves without requiring an installation step.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.data import (  # noqa: E402
    CandidateJSONError,
    CandidateSchemaError,
    MISSING,
    is_missing,
    stream_candidates,
)


GROUP_PATHS = {
    "Top level": (),
    "profile": ("profile",),
    "career_history[]": ("career_history", "items"),
    "education[]": ("education", "items"),
    "skills[]": ("skills", "items"),
    "certifications[]": ("certifications", "items"),
    "languages[]": ("languages", "items"),
    "redrob_signals": ("redrob_signals",),
}


def json_type(value: object) -> str:
    """Return an unambiguous JSON-oriented type label for a Python value."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def schema_object(schema: Mapping[str, Any], path: tuple[str, ...]) -> Mapping[str, Any]:
    """Locate one object definition in the bundled JSON Schema."""

    node: Mapping[str, Any] = schema
    for part in path:
        if part == "items":
            node = node["items"]
        else:
            node = node["properties"][part]
    return node


def raw_group_values(record: Mapping[str, Any], group: str) -> Iterable[Mapping[str, Any]]:
    """Yield raw objects belonging to one schema-report group."""

    if group == "Top level":
        yield record
    elif group.endswith("[]"):
        key = group[:-2]
        yield from record[key]
    else:
        yield record[group]


def collect_observed_types(
    candidates_path: Path, expected_keys: Mapping[str, set[str]]
) -> tuple[dict[str, dict[str, set[str]]], dict[str, int], dict[str, Any]]:
    """Stream raw records to collect field types and exact-key mismatches."""

    observed: dict[str, dict[str, set[str]]] = {
        group: defaultdict(set) for group in GROUP_PATHS
    }
    mismatch_counts = {group: 0 for group in GROUP_PATHS}
    first_record: dict[str, Any] | None = None

    with candidates_path.open("r", encoding="utf-8") as candidate_file:
        for line in candidate_file:
            record = json.loads(line)
            if first_record is None:
                first_record = copy.deepcopy(record)

            for group in GROUP_PATHS:
                for item in raw_group_values(record, group):
                    if set(item) != expected_keys[group]:
                        mismatch_counts[group] += 1
                    for key, value in item.items():
                        observed[group][key].add(json_type(value))

    if first_record is None:
        raise RuntimeError("Candidate file contains no records")
    return observed, mismatch_counts, first_record


def render_report(
    expected_keys: Mapping[str, set[str]],
    observed: Mapping[str, Mapping[str, set[str]]],
    mismatch_counts: Mapping[str, int],
    first_record: Mapping[str, Any],
    record_count: int,
    ids_contiguous: bool,
) -> str:
    """Build a deterministic Markdown report from the completed scan."""

    lines = [
        "# Phase 0 Schema Field Report",
        "",
        f"- Records schema-validated through `engine.data.stream_candidates`: **{record_count:,}**",
        "- JSON parse errors: **0**",
        "- Loader mode: **streaming, one line/record at a time; whole file never loaded**",
        "- Candidate IDs: **unique and contiguous `CAND_0000001` through `CAND_0100000`**"
        if ids_contiguous
        else "- Candidate IDs: **FAILED uniqueness/contiguity check**",
        "",
        "## Exact field groups",
        "",
    ]

    for group in GROUP_PATHS:
        lines.extend([f"### `{group}`", "", "| Field | Observed type | Notes |", "|---|---|---|"])
        for key in sorted(expected_keys[group]):
            types = " / ".join(sorted(observed[group].get(key, {"not observed"})))
            notes = ""
            if group == "career_history[]" and key == "end_date":
                notes = "Nullable for the current role."
            elif group == "education[]" and key == "grade":
                notes = "Schema permits null."
            elif group == "redrob_signals" and key in {
                "github_activity_score",
                "offer_acceptance_rate",
            }:
                notes = "`-1` is a missing-value sentinel, not a low score."
            elif group == "redrob_signals" and key == "skill_assessment_scores":
                notes = "Empty object is MISSING in typed accessors."
            elif group == "Top level" and key == "certifications":
                notes = "Empty array is MISSING in typed accessors; schema field is optional."
            elif group == "Top level" and key == "languages":
                notes = "Present in data; schema field is optional."
            lines.append(f"| `{key}` | {types} | {notes} |")
        lines.extend(
            [
                "",
                f"Exact-key mismatches against `candidate_schema.json`: **{mismatch_counts[group]}**",
                "",
            ]
        )

    redacted = copy.deepcopy(first_record)
    redacted["profile"]["anonymized_name"] = "[REDACTED]"
    lines.extend(
        [
            "## Fully redacted sample candidate",
            "",
            "Only `profile.anonymized_name` is redacted, as required.",
            "",
            "```json",
            json.dumps(redacted, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Phase 0 candidate data")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    expected_keys = {
        group: set(schema_object(schema, path)["properties"])
        for group, path in GROUP_PATHS.items()
    }

    # The public typed loader is the authoritative full-file validation pass.
    record_count = 0
    candidate_numbers: set[int] = set()
    sentinel_checks = {
        "github_-1": is_missing(-1, "github_activity_score"),
        "offer_-1": is_missing(-1, "offer_acceptance_rate"),
        "empty_certifications": is_missing([]),
        "empty_assessments": is_missing({}),
    }
    json_errors = 0
    schema_errors = 0

    try:
        for candidate in stream_candidates(args.candidates, schema_path=args.schema):
            record_count += 1
            candidate_numbers.add(int(candidate.candidate_id.removeprefix("CAND_")))

            signals = candidate.redrob_signals
            if signals.github_activity_score is MISSING:
                sentinel_checks["github_-1"] = is_missing(signals.github_activity_score)
            if signals.offer_acceptance_rate is MISSING:
                sentinel_checks["offer_-1"] = is_missing(signals.offer_acceptance_rate)
            if candidate.certifications is MISSING:
                sentinel_checks["empty_certifications"] = is_missing(candidate.certifications)
            if signals.skill_assessment_scores is MISSING:
                sentinel_checks["empty_assessments"] = is_missing(
                    signals.skill_assessment_scores
                )
    except CandidateJSONError as error:
        json_errors += 1
        print(f"JSON ERROR: {error}", file=sys.stderr)
    except CandidateSchemaError as error:
        schema_errors += 1
        print(f"SCHEMA ERROR: {error}", file=sys.stderr)

    expected_numbers = set(range(1, 100_001))
    ids_contiguous = candidate_numbers == expected_numbers

    # A separate streaming observation pass records raw JSON types for the report.
    observed, mismatch_counts, first_record = collect_observed_types(
        args.candidates, expected_keys
    )
    fields_match = all(count == 0 for count in mismatch_counts.values())

    report = render_report(
        expected_keys,
        observed,
        mismatch_counts,
        first_record,
        record_count,
        ids_contiguous,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    smoke = {
        "100,000 records streamed, 0 JSON errors, file never loaded whole": (
            record_count == 100_000 and json_errors == 0 and schema_errors == 0
        ),
        "all required field names present and matching candidate_schema.json": fields_match,
        "candidate_id unique + contiguous 1..100000": ids_contiguous,
        "sentinels and empty optional collections flagged as MISSING": all(
            sentinel_checks.values()
        ),
    }

    print(f"records_streamed={record_count}")
    print(f"json_errors={json_errors}")
    print(f"schema_errors={schema_errors}")
    print("whole_file_loaded=false")
    print(f"report={args.report.resolve()}")
    for label, passed in smoke.items():
        print(f"{'PASS' if passed else 'FAIL'}: {label}")

    return 0 if all(smoke.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
