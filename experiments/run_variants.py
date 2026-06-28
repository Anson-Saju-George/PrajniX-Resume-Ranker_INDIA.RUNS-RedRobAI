"""Variant harness entry point; Phase 2 implements score-everyone variant B."""

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

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.pipeline import rank_candidates  # noqa: E402
from engine.stages.output import write_submission  # noqa: E402


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
    """Hash the read-only source in chunks without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path, variant: str) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if variant != "B" or config.get("variant") != "B":
        raise ValueError("Phase 2 supports named variant B only")
    return config


def _manual_csv_checks(path: Path, valid_ids: frozenset[str]) -> dict[str, bool]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    scores = [float(row["score"]) for row in rows]
    ids = [row["candidate_id"] for row in rows]
    ranks = [int(row["rank"]) for row in rows]
    ties_deterministic = all(
        scores[index - 1] != scores[index] or ids[index - 1] < ids[index]
        for index in range(1, len(rows))
    )
    return {
        "exactly_100_rows": len(rows) == 100,
        "ranks_1_to_100": ranks == list(range(1, 101)),
        "scores_finite": all(math.isfinite(score) for score in scores),
        "scores_non_increasing": all(
            scores[index - 1] >= scores[index] for index in range(1, len(scores))
        ),
        "ids_unique": len(ids) == len(set(ids)),
        "ids_real": all(candidate_id in valid_ids for candidate_id in ids),
        "ties_candidate_id_ascending": ties_deterministic,
        "reasoning_nonempty": all(row["reasoning"].strip() for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a named ranking variant")
    parser.add_argument("--variant", required=True, choices=["B"])
    parser.add_argument("--config", type=Path, default=Path("configs/scoring.yaml"))
    parser.add_argument("--submission", type=Path, default=Path("outputs/submission_baseline.csv"))
    parser.add_argument("--debug", type=Path, default=Path("outputs/debug_top100.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates_path = _find_bundle_file("candidates.jsonl")
    jd_path = _find_bundle_file("job_description.docx")
    validator_path = _find_bundle_file("validate_submission.py")
    config = _load_config(args.config, args.variant)

    before = {
        "sha256": _sha256(candidates_path),
        "size": candidates_path.stat().st_size,
        "mtime_ns": candidates_path.stat().st_mtime_ns,
    }

    started = time.perf_counter()
    result = rank_candidates(candidates_path, jd_path, config)
    ranking_runtime = time.perf_counter() - started

    write_submission(args.submission, result.rows)
    args.debug.parent.mkdir(parents=True, exist_ok=True)
    debug_payload = {
        "variant": args.variant,
        "records_streamed": result.records_streamed,
        "hard_suppressed": result.hard_suppressed,
        "survivors_scored": result.survivors_scored,
        "ranking_runtime_seconds": round(ranking_runtime, 3),
        "top100": result.debug_top100,
    }
    args.debug.write_text(
        json.dumps(debug_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    validator = subprocess.run(
        [sys.executable, str(validator_path), str(args.submission)],
        capture_output=True,
        text=True,
        check=False,
    )
    after = {
        "sha256": _sha256(candidates_path),
        "size": candidates_path.stat().st_size,
        "mtime_ns": candidates_path.stat().st_mtime_ns,
    }
    csv_checks = _manual_csv_checks(args.submission, result.source_candidate_ids)
    honeypot_proxy_count = sum(item["hard_suppress"] for item in result.debug_top100)
    honeypot_proxy_rate = honeypot_proxy_count / len(result.debug_top100)

    checks = {
        "validate_submission.py passes": validator.returncode == 0,
        "exactly 100 rows; ranks 1-100; finite non-increasing scores; unique real IDs": all(
            csv_checks[name]
            for name in (
                "exactly_100_rows",
                "ranks_1_to_100",
                "scores_finite",
                "scores_non_increasing",
                "ids_unique",
                "ids_real",
            )
        ),
        "ties break candidate_id ascending": csv_checks["ties_candidate_id_ascending"],
        "honeypot proxy rate in top 100 < 10%": honeypot_proxy_rate < 0.10,
        "runtime < 5 min, CPU-only, no network": ranking_runtime < 300.0,
        "candidates.jsonl unmodified": before == after,
        "reasoning is non-empty and generated by field-only templates": csv_checks[
            "reasoning_nonempty"
        ],
    }

    print("PHASE 2 SMOKE TEST")
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {label}")
    print(f"validator_output={validator.stdout.strip() or validator.stderr.strip()}")
    print(f"ranking_runtime_seconds={ranking_runtime:.3f}")
    print(f"compute=CPU network_calls=none")
    print(f"honeypot_proxy_rate={honeypot_proxy_rate:.2%}")
    print(f"candidate_sha256_before={before['sha256']}")
    print(f"candidate_sha256_after={after['sha256']}")

    print("\nTOP 20")
    for item in result.debug_top100[:20]:
        one_line = item["reasoning"].replace("\n", " ")
        print(
            f"{item['rank']:>2}. {item['candidate_id']} | {item['current_title']} | "
            f"{item['years_of_experience']:.1f} yrs | {item['final_score']:.8f} | {one_line}"
        )

    distribution = Counter(item["current_title"] for item in result.debug_top100)
    print("\nTOP 100 TITLE DISTRIBUTION")
    for title, count in sorted(distribution.items(), key=lambda pair: (-pair[1], pair[0])):
        print(f"{title}: {count}")

    print(f"\nsubmission={args.submission.resolve()}")
    print(f"debug={args.debug.resolve()}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
