"""Submission rows, strict format guards, and deterministic CSV writing."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HEADER = ("candidate_id", "rank", "score", "reasoning")
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


@dataclass(frozen=True, slots=True)
class SubmissionRow:
    candidate_id: str
    rank: int
    score: float
    reasoning: str


def validate_submission_rows(
    rows: Iterable[SubmissionRow], valid_candidate_ids: set[str] | frozenset[str]
) -> list[SubmissionRow]:
    """Enforce every repository rule before a byte is written."""

    materialized = list(rows)
    if len(materialized) != 100:
        raise ValueError(f"Submission must contain exactly 100 rows; got {len(materialized)}")
    if [row.rank for row in materialized] != list(range(1, 101)):
        raise ValueError("Ranks must appear in ascending order from 1 through 100")

    ids = [row.candidate_id for row in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("Submission candidate IDs must be unique")
    if any(not CANDIDATE_ID_PATTERN.fullmatch(candidate_id) for candidate_id in ids):
        raise ValueError("Submission contains a malformed candidate ID")
    if any(candidate_id not in valid_candidate_ids for candidate_id in ids):
        raise ValueError("Submission contains an ID absent from the source dataset")

    for index, row in enumerate(materialized):
        if not math.isfinite(row.score):
            raise ValueError(f"Rank {row.rank} has a non-finite score")
        if not row.reasoning.strip():
            raise ValueError(f"Rank {row.rank} has empty reasoning")
        if index == 0:
            continue
        previous = materialized[index - 1]
        if previous.score < row.score:
            raise ValueError("Scores must be non-increasing by rank")
        if previous.score == row.score and previous.candidate_id > row.candidate_id:
            raise ValueError("Equal scores must tie-break by candidate_id ascending")
    return materialized


def write_submission(path: str | Path, rows: Iterable[SubmissionRow]) -> None:
    """Write a guarded UTF-8 CSV with the exact required header order."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow((row.candidate_id, row.rank, f"{row.score:.8f}", row.reasoning))
