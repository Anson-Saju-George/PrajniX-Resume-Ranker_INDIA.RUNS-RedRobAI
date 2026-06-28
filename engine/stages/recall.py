"""Configurable retrieval-first and dual-channel recall selection."""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from engine.data import stream_candidates
from engine.stages.scoring import extract_features, score_candidate


REGISTERED_RECALL_MODES = frozenset(
    {"A_retrieval_first", "B_score_everyone", "D_dual_channel"}
)


@dataclass(frozen=True, slots=True)
class RecallSelection:
    """Selected IDs plus a bounded background score for judged non-selections."""

    candidate_ids: frozenset[str]
    background_scores: np.ndarray
    text_candidates: int
    structured_candidates: int

    def includes(self, candidate_id: str) -> bool:
        return candidate_id in self.candidate_ids

    def background_score(self, candidate_id: str) -> float:
        index = int(candidate_id.removeprefix("CAND_")) - 1
        if index < 0 or index >= len(self.background_scores):
            raise ValueError(f"Candidate ID is outside artifact range: {candidate_id}")
        return float(self.background_scores[index])


def _artifact_arrays(artifact_dir: Path) -> tuple[np.ndarray, ...]:
    required = (
        "metadata.json",
        "candidate_ids.npy",
        "candidate_dense.npy",
        "jd_dense.npy",
        "bm25_scores.npy",
        "text_gate.npy",
    )
    missing = [name for name in required if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing recall artifacts {missing}; run experiments/precompute_recall_artifacts.py"
        )
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    candidate_ids = np.load(artifact_dir / "candidate_ids.npy", mmap_mode="r")
    dense = np.load(artifact_dir / "candidate_dense.npy", mmap_mode="r")
    jd_dense = np.load(artifact_dir / "jd_dense.npy", mmap_mode="r")
    bm25 = np.load(artifact_dir / "bm25_scores.npy", mmap_mode="r")
    gate = np.load(artifact_dir / "text_gate.npy", mmap_mode="r")
    expected = int(metadata["candidate_count"])
    if len(candidate_ids) != expected or dense.shape != (expected, len(jd_dense)):
        raise ValueError("Recall artifact arrays are not aligned with metadata")
    return candidate_ids, dense, jd_dense, bm25, gate


def _rrf_scores(
    dense: np.ndarray,
    jd_dense: np.ndarray,
    bm25: np.ndarray,
    gate: np.ndarray,
    rrf_k: int,
) -> np.ndarray:
    """Fuse BM25 and precomputed dense-vector ranks over text-eligible rows."""

    eligible = np.flatnonzero(gate)
    dense_scores = np.asarray(dense @ jd_dense, dtype=np.float32)
    bm25_order = eligible[np.argsort(-bm25[eligible], kind="stable")]
    dense_order = eligible[np.argsort(-dense_scores[eligible], kind="stable")]
    fused = np.zeros(len(bm25), dtype=np.float32)
    ranks = np.arange(1, len(eligible) + 1, dtype=np.float32)
    fused[bm25_order] += 1.0 / (rrf_k + ranks)
    fused[dense_order] += 1.0 / (rrf_k + ranks)
    return fused


def _top_text_ids(
    candidate_ids: np.ndarray,
    fused: np.ndarray,
    gate: np.ndarray,
    limit: int,
) -> frozenset[str]:
    eligible = np.flatnonzero(gate)
    order = eligible[np.argsort(-fused[eligible], kind="stable")[:limit]]
    return frozenset(str(candidate_ids[index]) for index in order)


def _structured_channel(
    dataset_path: Path,
    config: Mapping[str, Any],
    candidate_count: int,
    limit: int,
) -> tuple[frozenset[str], np.ndarray]:
    """Score structured features over every candidate and retain only top IDs."""

    reference_date = date.fromisoformat(str(config["reference_date"]))
    structured_scores = np.zeros(candidate_count, dtype=np.float32)
    heap: list[tuple[tuple[float, int], str]] = []
    for candidate in stream_candidates(dataset_path):
        index = int(candidate.candidate_id.removeprefix("CAND_")) - 1
        features = extract_features(candidate, reference_date)
        score = score_candidate(candidate, features, config).base_score
        structured_scores[index] = score
        numeric_id = index + 1
        quality = (score, -numeric_id)
        if len(heap) < limit:
            heapq.heappush(heap, (quality, candidate.candidate_id))
        elif quality > heap[0][0]:
            heapq.heapreplace(heap, (quality, candidate.candidate_id))
    return frozenset(candidate_id for _, candidate_id in heap), structured_scores


def build_recall_selection(
    dataset_path: Path,
    recall_mode: str,
    config: Mapping[str, Any],
) -> RecallSelection | None:
    """Build the configured shortlist without changing downstream scoring."""

    if recall_mode not in REGISTERED_RECALL_MODES:
        raise ValueError(f"Unknown recall_mode {recall_mode!r}")
    if recall_mode == "B_score_everyone":
        return None

    recall_config = config["recall"]
    artifact_dir = Path(recall_config["artifact_dir"])
    candidate_ids, dense, jd_dense, bm25, gate = _artifact_arrays(artifact_dir)
    fused = _rrf_scores(dense, jd_dense, bm25, gate, int(recall_config["rrf_k"]))
    mode_config = recall_config[recall_mode]
    text_ids = _top_text_ids(
        candidate_ids, fused, gate, int(mode_config["text_shortlist_size"])
    )
    fused_max = float(np.max(fused)) or 1.0

    if recall_mode == "A_retrieval_first":
        return RecallSelection(
            candidate_ids=text_ids,
            background_scores=fused / fused_max,
            text_candidates=len(text_ids),
            structured_candidates=0,
        )

    structured_ids, structured_scores = _structured_channel(
        dataset_path,
        config,
        len(candidate_ids),
        int(mode_config["structured_shortlist_size"]),
    )
    combined_background = 0.5 * (fused / fused_max) + 0.5 * (structured_scores / 100.0)
    return RecallSelection(
        candidate_ids=text_ids | structured_ids,
        background_scores=combined_background,
        text_candidates=len(text_ids),
        structured_candidates=len(structured_ids),
    )


def uses_score_everyone_fallback(recall_mode: str) -> bool:
    if recall_mode not in REGISTERED_RECALL_MODES:
        raise ValueError(f"Unknown recall_mode {recall_mode!r}")
    return False
