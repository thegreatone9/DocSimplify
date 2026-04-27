from __future__ import annotations

"""
Chunking & Pre-processing Module
=================================
Breaks structured chapter text into LLM-sized chunks with metadata,
and generates supporting context (glossary, book summary) for consistent
simplification.
"""

import json
import re

from src.prompts import build_glossary_prompt, build_summary_prompt


# ── Approximate token counting ──────────────────────────────────────────────

# Common English words list for glossary heuristic fallback
# (top ~200 most common words — anything NOT in this list that appears
#  frequently in the book is likely domain-specific jargon)
_COMMON_WORDS = None


def _load_common_words() -> set[str]:
    """Lazy-load a set of common English words for jargon detection."""
    global _COMMON_WORDS
    if _COMMON_WORDS is not None:
        return _COMMON_WORDS

    # Minimal set of the 300 most common English words
    _COMMON_WORDS = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "is", "are", "was", "were", "been",
        "has", "had", "did", "does", "being", "more", "very", "much",
        "those", "such", "here", "where", "why", "each", "many", "own",
        "may", "should", "still", "through", "between", "must", "both",
        "before", "while", "same", "part", "made", "find", "found",
        "long", "down", "too", "last", "right", "thing", "another",
        "under", "point", "world", "home", "every", "life", "hand",
        "high", "place", "end", "upon", "great", "old", "man", "small",
        "never", "few", "often", "without", "again", "however", "nothing",
        "different", "though", "might", "against", "something", "called",
        "seem", "during", "always", "important", "until", "since", "keep",
        "away", "began", "put", "fact", "set", "need", "might", "head",
        "already", "shall", "become", "rather", "quite", "large", "enough",
        "far", "left", "next", "whole", "yet", "less", "number", "water",
        "second", "later", "around", "able", "hard", "almost", "let",
        "above", "side", "name", "given", "tell", "best", "help",
        "line", "turn", "move", "off", "try", "ask", "kind", "form",
        "change", "mean", "children", "show", "read", "little", "state",
        "word", "words", "said", "really", "book", "open", "began",
        "does", "didn", "don", "going", "things", "taken", "having",
        "used", "using", "case", "three", "along", "once", "whether",
        "chapter", "page", "part", "section", "following", "also",
        "itself", "themselves", "himself", "herself", "within",
    }
    return _COMMON_WORDS


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string.

    Uses a fast character-based approximation (len / 4) since we're using
    local Ollama and don't need exact tokenizer alignment.

    Args:
        text: Input text string.

    Returns:
        Approximate token count.
    """
    # ~4 characters per token is a well-established heuristic for English
    return max(1, len(text) // 4)



def detect_document_structure(chapters: list[dict]) -> dict:
    """
    Analyze the document's structural elements upfront and create a structure map.

    Detects the PRIMARY numbered sequence by finding the longest ascending run
    of numbers (e.g., 1, 2, 3, 4, 5, 6). Any numbered items that break the
    sequence (out-of-order duplicates, footnote-style references) are classified
    as minor references.

    Returns a dict saved as intermediate/doc_structure.json:
      {
        "major_numbered_points": [{"number": "1", "preview": "..."}, ...],
        "minor_references": [{"number": "7", "preview": "..."}, ...],
        "bullet_groups": int,
        "headings": [str, ...],
        "total_major_segments": int,
        "segment_type": "numbered" | "headed" | "unstructured"
      }
    """
    full_text = "\n".join(ch.get("text", "") for ch in chapters)

    # ── Collect all numbered items with position ─────────────────────────
    numbered_pattern = re.compile(
        r"^(\d+)[\.\)]\s(.+?)(?=^\d+[\.\)]\s|\Z)", re.MULTILINE | re.DOTALL
    )

    all_items = []
    for match in numbered_pattern.finditer(full_text):
        num = int(match.group(1))
        content = match.group(2).strip()
        word_count = len(content.split())
        preview = content[:120].replace("\n", " ")
        position = match.start()
        all_items.append({
            "number": num,
            "number_str": match.group(1),
            "preview": preview,
            "word_count": word_count,
            "position": position,
        })

    # ── Find the primary sequence (two-pass) ──────────────────────────────
    # Pass 1: collect all numbers that appear in the document
    all_numbers = [item["number"] for item in all_items]

    # Pass 2: walk through items in order. An item belongs to the primary
    # sequence if:
    #   a) it equals the expected next number, OR
    #   b) it's the expected+1 and the expected doesn't exist at all
    # An item is a footnote/reference if:
    #   a) it's a duplicate of an already-used number, OR
    #   b) it would skip a number that exists later in the document
    major_points = []
    minor_refs = []

    if all_items:
        expected_next = 1
        used_numbers = set()
        # Pre-compute: which numbers appear later for any given position
        remaining_numbers_after = {}
        for i, item in enumerate(all_items):
            remaining_numbers_after[i] = set(
                it["number"] for it in all_items[i+1:]
            )

        for i, item in enumerate(all_items):
            num = item["number"]

            if num == expected_next and num not in used_numbers:
                # Perfect match — part of primary sequence
                major_points.append(item)
                used_numbers.add(num)
                expected_next = num + 1
            elif num > expected_next and num not in used_numbers:
                # Would skip number(s). Check if the skipped numbers exist
                # later in the document — if so, this is a footnote, not
                # a real sequence continuation.
                skipped = set(range(expected_next, num))
                skipped_exist_later = skipped & remaining_numbers_after.get(i, set())

                if skipped_exist_later:
                    # The skipped numbers appear later = this is a footnote
                    minor_refs.append(item)
                elif num <= expected_next + 2:
                    # Skipped numbers don't exist anywhere = genuine skip
                    major_points.append(item)
                    used_numbers.add(num)
                    expected_next = num + 1
                else:
                    minor_refs.append(item)
            else:
                # Duplicate or out-of-order — footnote
                minor_refs.append(item)

    # ── Detect headings ──────────────────────────────────────────────────
    headings = re.findall(r"^#{1,6}\s+(.+)$", full_text, re.MULTILINE)

    # ── Detect bullet groups ─────────────────────────────────────────────
    bullet_items = re.findall(r"^[-*•]\s", full_text, re.MULTILINE)

    # ── Determine primary structure type ─────────────────────────────────
    if len(major_points) >= 3:
        segment_type = "numbered"
        total_major = len(major_points)
    elif len(headings) >= 3:
        segment_type = "headed"
        total_major = len(headings)
    else:
        segment_type = "unstructured"
        total_major = 1

    # Clean output (remove internal fields)
    clean_major = [{"number": p["number_str"], "preview": p["preview"],
                     "word_count": p["word_count"]} for p in major_points]
    clean_minor = [{"number": r["number_str"], "preview": r["preview"],
                     "word_count": r["word_count"]} for r in minor_refs]

    return {
        "major_numbered_points": clean_major,
        "minor_references": clean_minor,
        "bullet_groups": len(bullet_items),
        "headings": headings,
        "total_major_segments": total_major,
        "segment_type": segment_type,
    }


def create_chunks(
    chapters: list[dict],
    max_tokens: int = 3000,
    overlap_tokens: int = 200,
) -> list[dict]:
    """
    Split chapters into LLM-sized chunks respecting document structure.

    Strategy:
      1. Within each chapter, identify structural segments:
         - Numbered sequences (1. ... 2. ... 3. ...)
         - Bullet point sequences (- ... - ...)
         - Heading-delimited sections
         - Plain paragraph groups
      2. Group consecutive structural items together (never split a numbered
         list across chunks).
      3. Accumulate complete segments until max_tokens is reached.
      4. If a single segment exceeds max_tokens, split it at paragraph
         boundaries within that segment.

    Args:
        chapters:       List of chapter dicts from the ingestion step.
        max_tokens:     Maximum tokens per chunk (excluding overlap).
        overlap_tokens: Tokens of trailing context from the previous chunk.

    Returns:
        List of dicts, each with keys:
          - "chunk_id"         : int
          - "chapter"          : str   (which chapter this belongs to)
          - "section"          : str
          - "text"             : str   (the chunk text, including overlap prefix)
          - "token_count"      : int
          - "is_chapter_start" : bool
    """
    chunks = []
    chunk_id = 0
    previous_chunk_tail = ""

    for chapter in chapters:
        chapter_name = chapter.get("chapter", "")
        section_name = chapter.get("section", "")
        full_text = chapter.get("text", "")

        if not full_text.strip():
            continue

        # Split text into structural segments
        segments = _split_into_segments(full_text)

        if not segments:
            continue

        # Accumulate segments into chunks, never splitting a segment
        current_parts = []
        current_tokens = 0
        is_chapter_start = True

        for segment in segments:
            seg_tokens = count_tokens(segment)

            # If a single segment exceeds max_tokens, split it internally
            if seg_tokens > max_tokens:
                # First flush any accumulated content
                if current_parts:
                    chunk_text = "\n\n".join(current_parts)
                    chunk_text_with_overlap = _add_overlap(chunk_text, previous_chunk_tail, overlap_tokens)

                    chunks.append({
                        "chunk_id": chunk_id,
                        "chapter": chapter_name,
                        "section": section_name,
                        "text": chunk_text_with_overlap,
                        "token_count": count_tokens(chunk_text_with_overlap),
                        "is_chapter_start": is_chapter_start,
                    })
                    previous_chunk_tail = chunk_text
                    chunk_id += 1
                    is_chapter_start = False
                    current_parts = []
                    current_tokens = 0

                # Split the oversized segment at paragraph boundaries
                sub_parts = _split_large_segment(segment, max_tokens)
                for sub in sub_parts:
                    chunk_text_with_overlap = _add_overlap(sub, previous_chunk_tail, overlap_tokens)
                    chunks.append({
                        "chunk_id": chunk_id,
                        "chapter": chapter_name,
                        "section": section_name,
                        "text": chunk_text_with_overlap,
                        "token_count": count_tokens(chunk_text_with_overlap),
                        "is_chapter_start": is_chapter_start,
                    })
                    previous_chunk_tail = sub
                    chunk_id += 1
                    is_chapter_start = False

                continue

            # If adding this segment would exceed the limit, flush first
            if current_parts and (current_tokens + seg_tokens) > max_tokens:
                chunk_text = "\n\n".join(current_parts)
                chunk_text_with_overlap = _add_overlap(chunk_text, previous_chunk_tail, overlap_tokens)

                chunks.append({
                    "chunk_id": chunk_id,
                    "chapter": chapter_name,
                    "section": section_name,
                    "text": chunk_text_with_overlap,
                    "token_count": count_tokens(chunk_text_with_overlap),
                    "is_chapter_start": is_chapter_start,
                })
                previous_chunk_tail = chunk_text
                chunk_id += 1
                is_chapter_start = False
                current_parts = []
                current_tokens = 0

            current_parts.append(segment)
            current_tokens += seg_tokens

        # Flush remaining content for this chapter
        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            chunk_text_with_overlap = _add_overlap(
                chunk_text, previous_chunk_tail, overlap_tokens,
                skip_if_chapter_start=is_chapter_start,
            )

            chunks.append({
                "chunk_id": chunk_id,
                "chapter": chapter_name,
                "section": section_name,
                "text": chunk_text_with_overlap,
                "token_count": count_tokens(chunk_text_with_overlap),
                "is_chapter_start": is_chapter_start,
            })
            previous_chunk_tail = chunk_text
            chunk_id += 1

    return chunks


def _split_into_segments(text: str) -> list[str]:
    """
    Split document text into structural segments that should stay together.

    Recognizes:
      - Numbered sequences (1. ... 2. ... 3. ...) — kept as one segment
      - Bullet sequences (- ... or * ...) — kept as one segment
      - Heading-delimited sections
      - Plain paragraphs grouped naturally

    Returns a list of text segments, each representing a complete structural unit.
    """
    lines = text.split("\n")
    segments = []
    current_block = []
    current_type = None  # "numbered", "bullet", "heading", "prose"

    for line in lines:
        stripped = line.strip()

        # Detect line type
        if re.match(r"^\d+[\.\)]\s", stripped):
            line_type = "numbered"
        elif re.match(r"^[-*•]\s", stripped):
            line_type = "bullet"
        elif re.match(r"^#{1,6}\s+", stripped):
            line_type = "heading"
        elif stripped == "":
            # Blank line — may separate blocks
            if current_type in ("numbered", "bullet"):
                # Blank lines in the middle of a list? Keep accumulating.
                current_block.append(line)
                continue
            elif current_block:
                # End of a prose block
                current_block.append(line)
                continue
            else:
                continue
        else:
            # Continuation text (part of a numbered item, prose paragraph, etc.)
            if current_type in ("numbered", "bullet"):
                # Continuation of list item (wrapped text)
                current_block.append(line)
                continue
            line_type = "prose"

        # If type changed, flush the current block
        if line_type != current_type and current_block:
            block_text = "\n".join(current_block).strip()
            if block_text:
                segments.append(block_text)
            current_block = []

        current_type = line_type
        current_block.append(line)

    # Flush remaining
    if current_block:
        block_text = "\n".join(current_block).strip()
        if block_text:
            segments.append(block_text)

    return segments


def _split_large_segment(segment: str, max_tokens: int) -> list[str]:
    """
    Split an oversized segment at paragraph boundaries.
    Used when a single structural block (e.g., a very long numbered list)
    exceeds max_tokens.
    """
    paragraphs = re.split(r"\n\s*\n", segment)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    parts = []
    current = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current and (current_tokens + para_tokens) > max_tokens:
            parts.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        parts.append("\n\n".join(current))

    return parts


def _add_overlap(
    chunk_text: str,
    previous_tail: str,
    overlap_tokens: int,
    skip_if_chapter_start: bool = False,
) -> str:
    """Add trailing overlap from the previous chunk for continuity."""
    if not previous_tail or skip_if_chapter_start or overlap_tokens <= 0:
        return chunk_text
    overlap_chars = overlap_tokens * 4
    overlap_text = previous_tail[-overlap_chars:]
    return overlap_text + "\n\n" + chunk_text


def extract_glossary(chapters: list[dict], llm_client=None) -> dict:
    """
    Extract key terms, jargon, and domain-specific vocabulary from the book.

    Two strategies (tried in order):
      1. If llm_client is provided: send a sample of text to the LLM and ask
         it to identify and define key terms.
      2. Fallback: frequency-based heuristic — find words that appear often in
         the book but rarely in common English.

    Args:
        chapters:   List of chapter dicts.
        llm_client: Optional OllamaClient instance.

    Returns:
        Dict of {term: definition} pairs.
    """
    # Gather a representative sample: first ~2000 chars from each chapter
    samples = []
    for ch in chapters:
        text = ch.get("text", "")
        samples.append(text[:2000])

    combined_sample = "\n\n---\n\n".join(samples)

    # Trim to ~12000 chars (~3000 tokens) to stay within model context
    if len(combined_sample) > 12000:
        combined_sample = combined_sample[:12000]

    # Strategy 1: LLM-based extraction
    if llm_client:
        try:
            system_prompt, user_prompt = build_glossary_prompt(combined_sample)
            response = llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
            )

            # Parse the JSON response
            # The model might wrap it in ```json ... ``` blocks — strip those
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r"^```(?:json)?\s*", "", response)
                response = re.sub(r"\s*```$", "", response)

            glossary = json.loads(response)
            if isinstance(glossary, dict):
                return glossary

        except (json.JSONDecodeError, Exception) as e:
            print(f"⚠️  LLM glossary extraction failed ({e}), falling back to heuristic")

    # Strategy 2: Frequency-based heuristic fallback
    return _extract_glossary_heuristic(chapters)


def _extract_glossary_heuristic(chapters: list[dict]) -> dict:
    """
    Simple frequency-based jargon detection.

    Finds words that appear 3+ times in the book but aren't in the
    common English words list. Returns them with placeholder definitions.
    """
    common_words = _load_common_words()
    all_text = " ".join(ch.get("text", "") for ch in chapters).lower()

    # Extract words (alphanumeric, 4+ chars to skip articles/prepositions)
    words = re.findall(r"\b[a-z]{4,}\b", all_text)

    # Count frequencies
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Find uncommon words that appear frequently
    jargon = {}
    for word, count in sorted(freq.items(), key=lambda x: -x[1]):
        if word not in common_words and count >= 3:
            jargon[word] = "(definition needed — detected as domain-specific term)"

        if len(jargon) >= 30:  # cap at 30 terms
            break

    return jargon


def generate_book_summary(chapters: list[dict], llm_client=None) -> str:
    """
    Generate a concise (~500 token) overview of the book's topic, purpose,
    and main themes.

    Args:
        chapters:   List of chapter dicts.
        llm_client: Optional OllamaClient instance.

    Returns:
        A ~500-token summary string.
    """
    # Collect the first ~500 tokens from each chapter
    intro_parts = []
    for ch in chapters:
        text = ch.get("text", "")
        chapter_name = ch.get("chapter", "Untitled")

        # Take first ~2000 chars (~500 tokens)
        snippet = text[:2000].strip()
        if snippet:
            intro_parts.append(f"[{chapter_name}]\n{snippet}")

    chapter_intros = "\n\n---\n\n".join(intro_parts)

    # Trim to fit in model context (~16000 chars ≈ 4000 tokens)
    if len(chapter_intros) > 16000:
        chapter_intros = chapter_intros[:16000]

    if llm_client:
        try:
            system_prompt, user_prompt = build_summary_prompt(chapter_intros)
            summary = llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
            )
            return summary.strip()
        except Exception as e:
            print(f"⚠️  LLM summary generation failed ({e}), using fallback")

    # Fallback: just return the first chapter's opening
    first_text = chapters[0].get("text", "")[:2000] if chapters else ""
    return f"This book covers the following topics based on its opening:\n\n{first_text}"


def get_chunk_stats(chunks: list[dict]) -> dict:
    """
    Compute statistics about the chunks for review.

    Args:
        chunks: List of chunk dicts from create_chunks().

    Returns:
        Dict with descriptive statistics.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_tokens": 0,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "chapters_covered": 0,
            "estimated_minutes": 0,
        }

    token_counts = [ch["token_count"] for ch in chunks]
    unique_chapters = set(ch["chapter"] for ch in chunks)

    total_tokens = sum(token_counts)

    # Rough estimate: each chunk produces ~same number of output tokens
    # At ~30 tokens/sec for a 7B model on Apple Silicon, estimate total time
    estimated_output_tokens = total_tokens  # simplified output ≈ same length
    estimated_seconds = estimated_output_tokens / 30  # conservative 30 tok/s
    estimated_minutes = estimated_seconds / 60

    return {
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "avg_tokens": round(total_tokens / len(chunks), 1),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "chapters_covered": len(unique_chapters),
        "estimated_minutes": round(estimated_minutes, 1),
    }
