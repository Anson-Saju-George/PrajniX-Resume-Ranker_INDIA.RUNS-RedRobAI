"""Precompute CPU-only BM25 and dense candidate retrieval artifacts.

This is an offline preparation step. Ranking loads these files and never creates
candidate or job-description embeddings at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from numpy.lib.format import open_memmap
from sklearn.feature_extraction.text import HashingVectorizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from engine.data import Candidate, stream_candidates  # noqa: E402


TOKEN_PATTERN = re.compile(r"[a-z0-9+#.]+")
FIELD_NAMES = ("title", "skills", "career", "profile")
FIELD_WEIGHTS = {"title": 3.0, "skills": 2.0, "career": 3.0, "profile": 1.0}
STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
        "has", "have", "in", "is", "it", "of", "on", "or", "our", "that", "the",
        "their", "this", "to", "using", "we", "will", "with", "you", "your",
        "candidate", "candidates", "experience", "role", "skills", "work", "years",
    }
)
TEXT_GATE_TERMS = (
    "embedding", "retrieval", "ranking", "recommendation", "recommender",
    "semantic search", "vector search", "hybrid search", "information retrieval",
    "bm25", "rag", "faiss", "pinecone", "qdrant", "weaviate", "milvus",
    "elasticsearch", "opensearch", "pgvector", "ndcg", "mrr",
)


def _find_bundle_file(name: str) -> Path:
    matches = [
        path for path in REPOSITORY_ROOT.rglob(name)
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


def _docx_text(path: Path) -> str:
    """Extract paragraph text from DOCX using only the standard library."""

    with zipfile.ZipFile(path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
    return "\n".join(text.text or "" for text in document.iter() if text.tag.endswith("}t"))


def _tokens(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.casefold()) if token not in STOP_WORDS]


def _candidate_fields(candidate: Candidate) -> dict[str, str]:
    return {
        "title": " ".join((candidate.profile.current_title, candidate.profile.headline)),
        "skills": " ".join(skill.name for skill in candidate.skills),
        "career": " ".join(
            f"{role.title} {role.description}" for role in candidate.career_history
        ),
        "profile": candidate.profile.summary,
    }


def _dense_text(fields: dict[str, str]) -> str:
    """Apply integer field weights before deterministic feature hashing."""

    parts: list[str] = []
    for field_name in FIELD_NAMES:
        parts.extend([fields[field_name]] * int(FIELD_WEIGHTS[field_name]))
    return " ".join(parts)


def _text_gate(fields: dict[str, str]) -> bool:
    text = " ".join(fields.values()).casefold()
    return any(term in text for term in TEXT_GATE_TERMS)


def _corpus_statistics(
    candidates_path: Path, query_terms: frozenset[str]
) -> tuple[int, dict[str, float], dict[str, Counter[str]]]:
    document_frequency = {field: Counter() for field in FIELD_NAMES}
    total_lengths = Counter()
    count = 0
    for candidate in stream_candidates(candidates_path):
        count += 1
        for field_name, text in _candidate_fields(candidate).items():
            tokens = _tokens(text)
            total_lengths[field_name] += len(tokens)
            document_frequency[field_name].update(set(tokens) & query_terms)
    averages = {
        field: total_lengths[field] / count if count else 0.0 for field in FIELD_NAMES
    }
    return count, averages, document_frequency


def _bm25_score(
    fields: dict[str, str],
    query_counts: Counter[str],
    document_count: int,
    average_lengths: dict[str, float],
    document_frequency: dict[str, Counter[str]],
) -> float:
    k1 = 1.5
    b = 0.75
    total = 0.0
    for field_name, text in fields.items():
        tokens = _tokens(text)
        term_counts = Counter(tokens)
        average = max(average_lengths[field_name], 1.0)
        normalization = k1 * (1.0 - b + b * len(tokens) / average)
        field_score = 0.0
        for term, query_frequency in query_counts.items():
            frequency = term_counts.get(term, 0)
            if not frequency:
                continue
            seen = document_frequency[field_name].get(term, 0)
            inverse_frequency = math.log(1.0 + (document_count - seen + 0.5) / (seen + 0.5))
            field_score += (
                inverse_frequency
                * (frequency * (k1 + 1.0) / (frequency + normalization))
                * min(query_frequency, 3)
            )
        total += FIELD_WEIGHTS[field_name] * field_score
    return total


def build_artifacts(
    candidates_path: Path,
    jd_path: Path,
    output_dir: Path,
    dimensions: int,
    batch_size: int,
) -> None:
    """Build aligned retrieval arrays in two bounded-memory dataset passes."""

    before_hash = _sha256(candidates_path)
    jd_text = _docx_text(jd_path)
    query_counts = Counter(_tokens(jd_text))
    query_terms = frozenset(query_counts)
    count, average_lengths, document_frequency = _corpus_statistics(
        candidates_path, query_terms
    )
    if count != 100_000:
        raise ValueError(f"Expected 100,000 candidates; found {count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_ids = open_memmap(
        output_dir / "candidate_ids.npy", mode="w+", dtype="<U12", shape=(count,)
    )
    dense = open_memmap(
        output_dir / "candidate_dense.npy",
        mode="w+",
        dtype=np.float32,
        shape=(count, dimensions),
    )
    bm25 = open_memmap(
        output_dir / "bm25_scores.npy", mode="w+", dtype=np.float32, shape=(count,)
    )
    gate = open_memmap(
        output_dir / "text_gate.npy", mode="w+", dtype=np.bool_, shape=(count,)
    )
    vectorizer = HashingVectorizer(
        n_features=dimensions,
        alternate_sign=True,
        analyzer="word",
        ngram_range=(1, 2),
        stop_words="english",
        norm="l2",
    )

    batch_text: list[str] = []
    batch_start = 0
    for index, candidate in enumerate(stream_candidates(candidates_path)):
        fields = _candidate_fields(candidate)
        candidate_ids[index] = candidate.candidate_id
        bm25[index] = _bm25_score(
            fields,
            query_counts,
            count,
            average_lengths,
            document_frequency,
        )
        gate[index] = _text_gate(fields)
        batch_text.append(_dense_text(fields))
        if len(batch_text) == batch_size:
            batch = vectorizer.transform(batch_text).astype(np.float32).toarray()
            dense[batch_start : index + 1] = batch
            batch_start = index + 1
            batch_text.clear()
    if batch_text:
        dense[batch_start:count] = vectorizer.transform(batch_text).astype(np.float32).toarray()

    jd_dense = vectorizer.transform([jd_text]).astype(np.float32).toarray()[0]
    np.save(output_dir / "jd_dense.npy", jd_dense, allow_pickle=False)
    candidate_ids.flush()
    dense.flush()
    bm25.flush()
    gate.flush()

    after_hash = _sha256(candidates_path)
    if before_hash != after_hash:
        raise RuntimeError("candidates.jsonl changed while artifacts were built")
    metadata = {
        "format_version": 1,
        "candidate_count": count,
        "dense_dimensions": dimensions,
        "dense_method": "signed feature hashing of field-weighted word unigrams+bigrams",
        "bm25_method": "field-weighted BM25 over fixed JD query terms",
        "field_weights": FIELD_WEIGHTS,
        "text_gate_terms": list(TEXT_GATE_TERMS),
        "candidate_sha256": before_hash,
        "job_description_sha256": _sha256(jd_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: precomputed retrieval artifacts for {count:,} candidates")
    print(f"text_gate_eligible={int(np.count_nonzero(gate)):,}")
    print(f"artifact_dir={output_dir.resolve()}")
    print(f"candidate_sha256={before_hash}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Phase 5 recall artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("data/recall"))
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_artifacts(
        _find_bundle_file("candidates.jsonl"),
        _find_bundle_file("job_description.docx"),
        args.output_dir,
        args.dimensions,
        args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
