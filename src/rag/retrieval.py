"""Retrieval pipeline with query expansion, hybrid search, and re-ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class RetrievedDocument:
    """A document returned by a retriever."""

    doc_id: str
    content: str
    score: float
    metadata: dict[str, object] | None = None
    source: str | None = None


@dataclass(frozen=True)
class RankedDocument:
    """A document after re-ranking."""

    doc_id: str
    content: str
    hybrid_score: float
    rerank_score: float
    metadata: dict[str, object] | None
    source: str | None
    query_variation: str


def _default_variations(query: str, max_variations: int) -> list[str]:
    variations = [query.strip()]
    if len(variations[0]) > 3:
        variations.append(variations[0].lower())
    variations.append(f'"{variations[0]}"')
    variations.append(f"{variations[0]} contexto")
    return list(dict.fromkeys(variations))[:max_variations]


def generate_query_variations(
    query: str,
    *,
    max_variations: int = 4,
    augmenter: Callable[[str, int], Iterable[str]] | None = None,
) -> list[str]:
    """Generate query variations for expansion."""

    if augmenter is None:
        return _default_variations(query, max_variations)
    variations = list(augmenter(query, max_variations))
    if not variations:
        return _default_variations(query, max_variations)
    return list(dict.fromkeys(variations))[:max_variations]


DenseSearcher = Callable[[str, int], Sequence[RetrievedDocument]]
SparseSearcher = Callable[[str, int], Sequence[RetrievedDocument]]
CrossEncoder = Callable[[str, Sequence[RetrievedDocument]], Sequence[float]]


def _normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    scale = max_score - min_score
    return [(score - min_score) / scale for score in scores]


def _aggregate_results(
    results: Iterable[RetrievedDocument],
    score_weight: float,
    per_doc_scores: dict[str, float],
    doc_lookup: dict[str, RetrievedDocument],
) -> None:
    scored = list(results)
    normalized = _normalize_scores([doc.score for doc in scored])
    for doc, norm in zip(scored, normalized, strict=False):
        doc_lookup.setdefault(doc.doc_id, doc)
        per_doc_scores[doc.doc_id] = per_doc_scores.get(doc.doc_id, 0.0) + norm * score_weight


def hybrid_search(
    queries: Sequence[str],
    *,
    dense_search: DenseSearcher,
    sparse_search: SparseSearcher,
    top_k: int = 10,
    dense_weight: float = 0.6,
) -> list[tuple[str, list[RetrievedDocument]]]:
    """Execute hybrid search for each query variation."""

    combined: list[tuple[str, list[RetrievedDocument]]] = []
    sparse_weight = 1.0 - dense_weight
    for query in queries:
        per_doc_scores: dict[str, float] = {}
        doc_lookup: dict[str, RetrievedDocument] = {}

        dense_results = dense_search(query, top_k)
        sparse_results = sparse_search(query, top_k)

        _aggregate_results(dense_results, dense_weight, per_doc_scores, doc_lookup)
        _aggregate_results(sparse_results, sparse_weight, per_doc_scores, doc_lookup)

        ranked = sorted(
            (
                RetrievedDocument(
                    doc_id=doc_id,
                    content=doc_lookup[doc_id].content,
                    score=score,
                    metadata=doc_lookup[doc_id].metadata,
                    source=doc_lookup[doc_id].source,
                )
                for doc_id, score in per_doc_scores.items()
            ),
            key=lambda item: item.score,
            reverse=True,
        )[:top_k]
        combined.append((query, ranked))
    return combined


def rerank_documents(
    query: str,
    documents: Sequence[RetrievedDocument],
    *,
    cross_encoder: CrossEncoder | None = None,
) -> list[RankedDocument]:
    """Apply cross-encoder re-ranking to documents."""

    if not documents:
        return []
    if cross_encoder is None:
        rerank_scores = [doc.score for doc in documents]
    else:
        rerank_scores = list(cross_encoder(query, documents))
        if len(rerank_scores) != len(documents):
            raise ValueError("Cross-encoder returned mismatched score length.")

    ranked = sorted(
        (
            RankedDocument(
                doc_id=doc.doc_id,
                content=doc.content,
                hybrid_score=doc.score,
                rerank_score=rerank_score,
                metadata=doc.metadata,
                source=doc.source,
                query_variation=query,
            )
            for doc, rerank_score in zip(documents, rerank_scores, strict=False)
        ),
        key=lambda item: item.rerank_score,
        reverse=True,
    )
    return ranked


def build_context_payload(
    ranked: Sequence[RankedDocument],
    *,
    top_k: int = 5,
) -> list[dict[str, object]]:
    """Return structured context payload for downstream generation."""

    payload: list[dict[str, object]] = []
    for doc in ranked[:top_k]:
        payload.append(
            {
                "doc_id": doc.doc_id,
                "content": doc.content,
                "metadata": doc.metadata or {},
                "source": doc.source,
                "scores": {
                    "hybrid": doc.hybrid_score,
                    "rerank": doc.rerank_score,
                },
                "query_variation": doc.query_variation,
            }
        )
    return payload


def run_retrieval_pipeline(
    query: str,
    *,
    dense_search: DenseSearcher,
    sparse_search: SparseSearcher,
    cross_encoder: CrossEncoder | None = None,
    top_k: int = 5,
    max_variations: int = 4,
    augmenter: Callable[[str, int], Iterable[str]] | None = None,
    dense_weight: float = 0.6,
) -> list[dict[str, object]]:
    """Full retrieval pipeline: variations -> hybrid search -> rerank -> payload."""

    variations = generate_query_variations(
        query, max_variations=max_variations, augmenter=augmenter
    )
    hybrid_results = hybrid_search(
        variations,
        dense_search=dense_search,
        sparse_search=sparse_search,
        top_k=top_k,
        dense_weight=dense_weight,
    )

    reranked: list[RankedDocument] = []
    for variation, docs in hybrid_results:
        reranked.extend(
            rerank_documents(variation, docs, cross_encoder=cross_encoder)
        )

    reranked_sorted = sorted(reranked, key=lambda item: item.rerank_score, reverse=True)
    return build_context_payload(reranked_sorted, top_k=top_k)
