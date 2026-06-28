"""Merge externally assigned judge labels into the frozen blind judge set."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read a small judge CSV while preserving its declared row order."""

    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def merge_labels(blind_path: Path, labels_path: Path, output_path: Path) -> Counter[int]:
    """Join labels by candidate ID and enforce the frozen 80-row contract."""

    blind_rows = _read_rows(blind_path)
    label_rows = _read_rows(labels_path)
    if len(blind_rows) != 80 or len(label_rows) != 80:
        raise ValueError("Both judge inputs must contain exactly 80 rows")

    labels_by_candidate: dict[str, int] = {}
    for row in label_rows:
        candidate_id = row["candidate_id"]
        if candidate_id in labels_by_candidate:
            raise ValueError(f"Duplicate labeled candidate: {candidate_id}")
        raw_label = row["label"].strip()
        if raw_label not in {"0", "1", "2", "3"}:
            raise ValueError(f"Invalid label for {candidate_id}: {raw_label!r}")
        labels_by_candidate[candidate_id] = int(raw_label)

    blind_ids = [row["candidate_id"] for row in blind_rows]
    if set(blind_ids) != set(labels_by_candidate):
        raise ValueError("Labeled candidate IDs do not exactly match the blind set")

    merged_rows: list[dict[str, str]] = []
    for row in blind_rows:
        merged = dict(row)
        merged["label"] = str(labels_by_candidate[row["candidate_id"]])
        merged_rows.append(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(blind_rows[0]))
        writer.writeheader()
        writer.writerows(merged_rows)

    written = _read_rows(output_path)
    if [row["candidate_id"] for row in written] != blind_ids:
        raise ValueError("Merged output changed the frozen blind row order")
    if any(row["label"].strip() not in {"0", "1", "2", "3"} for row in written):
        raise ValueError("Merged output contains a blank or invalid label")
    return Counter(int(row["label"]) for row in written)


def main() -> int:
    distribution = merge_labels(
        REPOSITORY_ROOT / "outputs" / "judge_set_blind.csv",
        REPOSITORY_ROOT / "outputs" / "judge_set_labels.csv",
        REPOSITORY_ROOT / "outputs" / "judge_set_labeled.csv",
    )
    print("PASS: 80 labels merged; candidate IDs preserve blind-file row order")
    print("Label distribution: " + ", ".join(f"{label}={distribution[label]}" for label in range(4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
