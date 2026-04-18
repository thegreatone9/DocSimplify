from __future__ import annotations

"""
Embedding-Based Quality Assurance
==================================
Uses sentence-transformers to verify semantic similarity between original
and simplified text. Catches meaning drift, dropped arguments, and
hallucinations without consuming any LLM tokens.

Requires:
    pip install sentence-transformers

If not installed, all functions gracefully return None / skip.
"""

import warnings


def _load_model():
    """
    Load the sentence-transformer model once and cache it.
    Uses all-MiniLM-L6-v2: 80MB, runs on CPU in <1 second per chunk.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None


# Module-level lazy cache
_model = None


def _get_model():
    """Get the cached model, loading on first call."""
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def is_available() -> bool:
    """Check if sentence-transformers is installed."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def check_semantic_similarity(original: str, simplified: str) -> float | None:
    """
    Compute cosine similarity between original and simplified text embeddings.

    Args:
        original:   The original chunk text.
        simplified: The simplified output.

    Returns:
        Cosine similarity score (0.0-1.0), or None if unavailable.
        Scores above 0.85 indicate good semantic preservation.
        Scores below 0.70 indicate significant meaning drift.
    """
    model = _get_model()
    if model is None:
        return None

    if not original.strip() or not simplified.strip():
        return None

    try:
        embeddings = model.encode([original, simplified], convert_to_tensor=False)
        # Cosine similarity
        import numpy as np
        a, b = embeddings[0], embeddings[1]
        similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return round(similarity, 4)
    except Exception:
        return None


def run_embedding_qa(
    chunks: list[dict],
    checkpoint: dict,
    similarity_threshold: float = 0.75,
) -> dict:
    """
    Run embedding-based QA on all simplified chunks.

    Args:
        chunks:     List of chunk dicts with "text" key.
        checkpoint: Dict of {chunk_id: {simplified_text, ...}}.
        similarity_threshold: Below this score, flag the chunk.

    Returns:
        Dict with:
          - flagged_chunks: list of (chunk_id, score) tuples
          - avg_similarity: float
          - total_checked: int
    """
    model = _get_model()
    if model is None:
        return {"flagged_chunks": [], "avg_similarity": 0.0, "total_checked": 0,
                "error": "sentence-transformers not installed"}

    scores = []
    flagged = []

    for chunk in chunks:
        cid = str(chunk["chunk_id"])
        entry = checkpoint.get(cid, {})
        simplified = entry.get("simplified_text", "")

        if not simplified or len(chunk["text"].split()) < 5:
            continue

        score = check_semantic_similarity(chunk["text"], simplified)
        if score is not None:
            scores.append(score)
            entry["semantic_similarity"] = score

            if score < similarity_threshold:
                flagged.append((cid, score))

    avg = sum(scores) / len(scores) if scores else 0.0

    return {
        "flagged_chunks": flagged,
        "avg_similarity": round(avg, 4),
        "total_checked": len(scores),
    }
