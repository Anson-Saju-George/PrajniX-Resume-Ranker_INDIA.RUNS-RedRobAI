"""Config-driven variant harness with reproducible per-variant artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.pipeline import rank_candidates  # noqa: E402
from engine.stages.output import write_submission  # noqa: E402


REGISTERED_VARIANTS = ("A", "B", "D")
AI_DOMAIN_TITLE_MARKERS = (
    "ai ",
    "artificial intelligence",
    "machine learning",
    "ml engineer",
    "(ml)",
    "data scientist",
    "nlp",
    "search engineer",
    "recommendation",
    "applied scientist",
)


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


def _load_harness_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = {
        "variants",
        "aspect_weights",
        "penalty_profiles",
        "reference_date",
        "top_k",
        "score_decimals",
    } - set(config)
    if missing:
        raise ValueError(f"Harness configuration is missing keys: {sorted(missing)}")
    return config


def _resolve_variant(
    harness: Mapping[str, Any],
    variant_name: str,
    evaluation_ids: set[str],
) -> dict[str, Any]:
    """Resolve one named variant without embedding policy in Python code."""

    if variant_name not in harness["variants"]:
        raise ValueError(f"Variant {variant_name!r} is not registered in scoring.yaml")
    variant = harness["variants"][variant_name]
    strength = variant["penalty_strength"]
    if strength not in harness["penalty_profiles"]:
        raise ValueError(f"Unknown penalty_strength {strength!r}")
    return {
        "variant_name": variant_name,
        "recall_mode": variant["recall_mode"],
        "penalty_strength": strength,
        "weights": dict(harness["aspect_weights"]),
        "penalties": dict(harness["penalty_profiles"][strength]),
        "reference_date": harness["reference_date"],
        "top_k": harness["top_k"],
        "score_decimals": harness["score_decimals"],
        "_evaluation_candidate_ids": evaluation_ids,
    }


def _load_optional_judge_labels(
    labeled_path: Path, key_path: Path
) -> tuple[dict[str, int] | None, str]:
    """Load complete 0–3 labels joined row-for-row through the frozen key file."""

    if not labeled_path.is_file():
        return None, "not_present"
    if not key_path.is_file():
        return None, "skipped_missing_key"

    with labeled_path.open("r", encoding="utf-8", newline="") as labeled_file:
        labeled_rows = list(csv.DictReader(labeled_file))
    with key_path.open("r", encoding="utf-8", newline="") as key_file:
        key_rows = list(csv.DictReader(key_file))
    if len(labeled_rows) != len(key_rows):
        return None, "skipped_row_count_mismatch"

    labels: dict[str, int] = {}
    for labeled, key in zip(labeled_rows, key_rows):
        candidate_id = key["candidate_id"]
        if labeled.get("candidate_id") != candidate_id:
            return None, "skipped_key_alignment_mismatch"
        raw_label = labeled.get("label", "").strip()
        if raw_label not in {"0", "1", "2", "3"}:
            return None, "skipped_incomplete_labels"
        labels[candidate_id] = int(raw_label)
    return labels, "available"


def _dcg(relevances: list[int], cutoff: int) -> float:
    return sum(
        (2**relevance - 1) / math.log2(index + 2)
        for index, relevance in enumerate(relevances[:cutoff])
    )


def _judge_metrics(scores: Mapping[str, float], labels: Mapping[str, int]) -> dict[str, float]:
    """Compute challenge metrics over the deliberately sampled judged pool."""

    ranked_ids = sorted(labels, key=lambda cid: (-scores[cid], cid))
    relevance = [labels[candidate_id] for candidate_id in ranked_ids]

    def ndcg(cutoff: int) -> float:
        ideal = sorted(relevance, reverse=True)
        denominator = _dcg(ideal, cutoff)
        return _dcg(relevance, cutoff) / denominator if denominator else 0.0

    binary = [int(value >= 3) for value in relevance]
    relevant_total = sum(binary)
    hits = 0
    precision_sum = 0.0
    for index, is_relevant in enumerate(binary, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / index
    average_precision = precision_sum / relevant_total if relevant_total else 0.0
    return {
        "NDCG@10": ndcg(10),
        "NDCG@50": ndcg(50),
        "MAP": average_precision,
        "P@10": sum(binary[:10]) / 10.0,
    }


def _manual_csv_checks(path: Path, valid_ids: frozenset[str]) -> dict[str, bool]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    scores = [float(row["score"]) for row in rows]
    ids = [row["candidate_id"] for row in rows]
    ranks = [int(row["rank"]) for row in rows]
    return {
        "exactly_100_rows": len(rows) == 100,
        "ranks_1_to_100": ranks == list(range(1, 101)),
        "scores_finite": all(math.isfinite(score) for score in scores),
        "scores_non_increasing": all(
            scores[index - 1] >= scores[index] for index in range(1, len(scores))
        ),
        "ids_unique_and_real": len(ids) == len(set(ids))
        and all(candidate_id in valid_ids for candidate_id in ids),
        "ties_deterministic": all(
            scores[index - 1] != scores[index] or ids[index - 1] < ids[index]
            for index in range(1, len(rows))
        ),
    }


def _run_variant(
    variant_name: str,
    harness: Mapping[str, Any],
    candidates_path: Path,
    jd_path: Path,
    validator_path: Path,
    output_root: Path,
    judge_labels: dict[str, int] | None,
    judge_status: str,
) -> tuple[dict[str, Any], bool]:
    variant_dir = output_root / variant_name
    submission_path = variant_dir / "submission.csv"
    debug_path = variant_dir / "debug_top100.json"
    metrics_path = variant_dir / "metrics.json"
    evaluation_ids = set(judge_labels or {})
    config = _resolve_variant(harness, variant_name, evaluation_ids)

    started = time.perf_counter()
    result = rank_candidates(candidates_path, jd_path, config)
    runtime = time.perf_counter() - started
    write_submission(submission_path, result.rows)

    variant_dir.mkdir(parents=True, exist_ok=True)
    debug_payload = {
        "variant": variant_name,
        "recall_mode": result.recall_mode,
        "recall_fallback_to_score_everyone": result.recall_fallback_to_score_everyone,
        "penalty_strength": config["penalty_strength"],
        "records_streamed": result.records_streamed,
        "hard_suppressed": result.hard_suppressed,
        "survivors_scored": result.survivors_scored,
        "ranking_runtime_seconds": round(runtime, 3),
        "top100": result.debug_top100,
    }
    debug_path.write_text(
        json.dumps(debug_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    validator = subprocess.run(
        [sys.executable, str(validator_path), str(submission_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    csv_checks = _manual_csv_checks(submission_path, result.source_candidate_ids)
    titles = [item["current_title"] for item in result.debug_top100]
    title_distribution = dict(sorted(Counter(titles).items()))
    top10_titles = titles[:10]
    suspicious_top10 = [
        title
        for title in top10_titles
        if not any(marker in title.casefold() for marker in AI_DOMAIN_TITLE_MARKERS)
    ]
    honeypot_count = sum(item["hard_suppress"] for item in result.debug_top100)

    metrics: dict[str, Any] = {
        "variant": variant_name,
        "recall_mode": result.recall_mode,
        "recall_fallback_to_score_everyone": result.recall_fallback_to_score_everyone,
        "penalty_strength": config["penalty_strength"],
        "runtime_seconds": round(runtime, 3),
        "top100_honeypot_proxy_count": honeypot_count,
        "top100_honeypot_proxy_rate": honeypot_count / 100.0,
        "title_distribution": title_distribution,
        "top10_domain_sanity": {
            "pass": not suspicious_top10,
            "titles": top10_titles,
            "suspicious_titles": suspicious_top10,
        },
        "validator": {
            "pass": validator.returncode == 0,
            "output": (validator.stdout.strip() or validator.stderr.strip()),
        },
        "format_checks": csv_checks,
        "judge_metrics_status": judge_status,
    }
    if judge_labels is not None:
        metrics["judge_metrics"] = _judge_metrics(
            result.evaluation_scores, judge_labels
        )

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    passed = (
        validator.returncode == 0
        and all(csv_checks.values())
        and runtime < 300.0
        and honeypot_count < 10
    )
    print(
        f"{'PASS' if passed else 'FAIL'}: variant {variant_name} | "
        f"recall={result.recall_mode} fallback={result.recall_fallback_to_score_everyone} "
        f"runtime={runtime:.3f}s validator={validator.returncode == 0}"
    )
    print(f"  submission={submission_path.resolve()}")
    print(f"  debug={debug_path.resolve()}")
    print(f"  metrics={metrics_path.resolve()}")
    return metrics, passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one or more registered variants")
    parser.add_argument(
        "--variant", nargs="+", required=True, choices=REGISTERED_VARIANTS
    )
    parser.add_argument("--config", type=Path, default=Path("configs/scoring.yaml"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/variants")
    )
    parser.add_argument(
        "--labeled-judge", type=Path, default=Path("outputs/judge_set_labeled.csv")
    )
    parser.add_argument(
        "--judge-key", type=Path, default=Path("outputs/judge_set_key.csv")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates_path = _find_bundle_file("candidates.jsonl")
    jd_path = _find_bundle_file("job_description.docx")
    validator_path = _find_bundle_file("validate_submission.py")
    harness = _load_harness_config(args.config)
    judge_labels, judge_status = _load_optional_judge_labels(
        args.labeled_judge, args.judge_key
    )

    before_hash = _sha256(candidates_path)
    all_passed = True
    for variant_name in args.variant:
        _, passed = _run_variant(
            variant_name,
            harness,
            candidates_path,
            jd_path,
            validator_path,
            args.output_root,
            judge_labels,
            judge_status,
        )
        all_passed &= passed
    after_hash = _sha256(candidates_path)
    unchanged = before_hash == after_hash
    print(f"{'PASS' if unchanged else 'FAIL'}: candidates.jsonl SHA-256 unchanged")
    print(f"candidate_sha256_before={before_hash}")
    print(f"candidate_sha256_after={after_hash}")
    print(f"judge_metrics_status={judge_status}")
    return 0 if all_passed and unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
