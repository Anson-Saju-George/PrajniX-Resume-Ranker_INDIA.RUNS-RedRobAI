"""Single Stage-3 entry point for the locked PrajniX ranking pipeline."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from engine.pipeline import rank_candidates
from engine.stages.output import (
    verify_xlsx_against_csv,
    write_submission,
    write_xlsx_from_csv,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "scoring.yaml"
LOCKED_VARIANT = "D_overband_mild_avail_heavy"
REQUIRED_ARTIFACTS = (
    "metadata.json",
    "candidate_ids.npy",
    "candidate_dense.npy",
    "jd_dense.npy",
    "bm25_scores.npy",
    "text_gate.npy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_input(path: Path, filename: str) -> Path:
    """Use the requested path, or the one unambiguous bundled challenge file."""

    if path.is_file():
        return path.resolve()
    if path.name != filename:
        raise FileNotFoundError(path)
    matches = [
        item
        for item in REPOSITORY_ROOT.rglob(filename)
        if "__MACOSX" not in item.parts and item.is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{path} was not found and exactly one bundled {filename} could not be resolved"
        )
    return matches[0].resolve()


def _locked_config() -> dict[str, Any]:
    harness = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if harness.get("default_variant") != LOCKED_VARIANT:
        raise ValueError(
            f"Expected default_variant={LOCKED_VARIANT}; found {harness.get('default_variant')!r}"
        )
    variant = harness["variants"][LOCKED_VARIANT]
    expected = {
        "recall_mode": "D_dual_channel",
        "experience_profile": "overband_mild",
        "penalty_strength": "heavy",
    }
    if any(variant.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Locked variant does not match {expected}")

    recall = dict(harness["recall"])
    recall["artifact_dir"] = str(
        (REPOSITORY_ROOT / recall["artifact_dir"]).resolve()
    )
    return {
        "variant_name": LOCKED_VARIANT,
        "recall_mode": variant["recall_mode"],
        "penalty_strength": variant["penalty_strength"],
        "experience_profile": variant["experience_profile"],
        "weights": dict(harness["aspect_weights"]),
        "penalties": dict(harness["penalty_profiles"][variant["penalty_strength"]]),
        "experience_band_fits": dict(
            harness["experience_profiles"][variant["experience_profile"]]
        ),
        "reference_date": harness["reference_date"],
        "top_k": harness["top_k"],
        "score_decimals": harness["score_decimals"],
        "recall": recall,
    }


def _require_precomputed_artifacts(config: dict[str, Any], candidates: Path) -> None:
    artifact_dir = Path(config["recall"]["artifact_dir"])
    missing = [name for name in REQUIRED_ARTIFACTS if not (artifact_dir / name).is_file()]
    if not missing:
        return
    command = (
        f"python scripts/precompute.py --candidates \"{candidates}\" "
        f"--output-dir \"{artifact_dir}\""
    )
    raise FileNotFoundError(
        "Precomputed retrieval artifacts are missing: "
        f"{', '.join(missing)}. Build them offline first with:\n{command}\n"
        "rank.py never computes embeddings during ranking."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked PrajniX ranker")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--verify-against",
        type=Path,
        help="Optional local regression reference; never affects ranking or exit status",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = _resolve_input(args.candidates, "candidates.jsonl")
    jd = _resolve_input(candidates.with_name("job_description.docx"), "job_description.docx")
    config = _locked_config()
    _require_precomputed_artifacts(config, candidates)

    started = time.perf_counter()
    result = rank_candidates(candidates, jd, config)
    write_submission(args.out, result.rows)
    xlsx_path = write_xlsx_from_csv(args.out)
    xlsx_verification = verify_xlsx_against_csv(args.out, xlsx_path)
    runtime = time.perf_counter() - started
    output_path = args.out.resolve()
    output_hash = _sha256(output_path)
    print(f"Wrote {output_path}")
    print(f"Wrote {xlsx_path.resolve()}")
    print(f"Runtime: {runtime:.3f}s (CPU-only, precomputed artifacts loaded)")
    print(f"SHA-256: {output_hash}")

    if args.verify_against is not None:
        reference = args.verify_against.resolve()
        reference_label = args.verify_against.as_posix()
        if reference.is_file():
            reference_hash = _sha256(reference)
            matches = output_hash == reference_hash
            print(f"Reference SHA-256: {reference_hash}")
            print(
                f"Byte-identical to {reference_label}: "
                f"{'PASS' if matches else 'FAIL'}"
            )
        else:
            print(f"Reference comparison skipped; file not found: {reference_label}")

    if not all(xlsx_verification["checks"].values()):
        print("ERROR: generated XLSX failed internal fidelity checks", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
