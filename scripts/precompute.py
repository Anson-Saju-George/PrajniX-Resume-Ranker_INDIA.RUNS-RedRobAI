"""Offline builder for the retrieval artifacts consumed by rank.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.precompute_recall_artifacts import build_artifacts  # noqa: E402


def _resolve(path: Path, filename: str) -> Path:
    if path.is_file():
        return path.resolve()
    matches = [
        item
        for item in REPOSITORY_ROOT.rglob(filename)
        if "__MACOSX" not in item.parts and item.is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not resolve exactly one {filename}")
    return matches[0].resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute local BM25/dense artifacts")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--jd", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/recall"))
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = _resolve(args.candidates, args.candidates.name)
    requested_jd = args.jd or candidates.with_name("job_description.docx")
    jd = _resolve(requested_jd, "job_description.docx")
    build_artifacts(
        candidates,
        jd,
        args.output_dir,
        args.dimensions,
        args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
