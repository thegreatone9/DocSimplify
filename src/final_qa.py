from __future__ import annotations

"""
Final Output QA
================
Post-assembly quality checks on the final markdown output.
Detects and fixes anomalies like duplicate headings, repeated paragraphs,
and other structural issues without any LLM calls.
"""

import re
from collections import Counter


def run_final_qa(markdown: str) -> dict:
    """
    Run a comprehensive QA pass on the final assembled markdown.

    Checks for and fixes:
      - Duplicate consecutive headings
      - Duplicate consecutive paragraphs
      - Repeated author name lines
      - Excessive blank lines
      - Orphaned footnote references

    Args:
        markdown: The assembled markdown string.

    Returns:
        Dict with:
          - fixed_markdown: str (cleaned output)
          - issues_found: list of issue descriptions
          - issues_fixed: int
    """
    issues = []
    fixed = markdown

    # ── 1. Remove duplicate consecutive headings ─────────────────────────────
    fixed, heading_fixes = _fix_duplicate_headings(fixed)
    issues.extend(heading_fixes)

    # ── 2. Remove duplicate consecutive paragraphs ───────────────────────────
    fixed, para_fixes = _fix_duplicate_paragraphs(fixed)
    issues.extend(para_fixes)

    # ── 3. Remove duplicate author bylines ───────────────────────────────────
    fixed, byline_fixes = _fix_duplicate_bylines(fixed)
    issues.extend(byline_fixes)

    # ── 4. Remove LLM meta-summary paragraphs ──────────────────────────────
    fixed, meta_fixes = _fix_meta_summaries(fixed)
    issues.extend(meta_fixes)

    # ── 5. Remove repeated sentences within paragraphs ───────────────────────
    fixed, repeat_fixes = _fix_repeated_sentences(fixed)
    issues.extend(repeat_fixes)

    # ── 6. Clean up excessive blank lines ────────────────────────────────────
    fixed = re.sub(r"\n{4,}", "\n\n\n", fixed)

    # ── 7. Remove orphaned horizontal rules (--- with nothing after) ─────────
    fixed = re.sub(r"\n---\n\n---\n", "\n---\n", fixed)

    return {
        "fixed_markdown": fixed.strip() + "\n",
        "issues_found": issues,
        "issues_fixed": len(issues),
    }


def _fix_duplicate_headings(text: str) -> tuple[str, list[str]]:
    """Remove duplicate consecutive headings (same text, possibly different levels)."""
    lines = text.split("\n")
    cleaned = []
    issues = []
    prev_heading_text = None

    for line in lines:
        stripped = line.strip()
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

        if heading_match:
            heading_text = heading_match.group(2).strip()
            if heading_text == prev_heading_text:
                issues.append(f"Removed duplicate heading: '{heading_text}'")
                continue  # Skip the duplicate
            prev_heading_text = heading_text
        else:
            if stripped:  # Reset heading tracker on non-empty non-heading line
                prev_heading_text = None

        cleaned.append(line)

    return "\n".join(cleaned), issues


def _fix_duplicate_paragraphs(text: str) -> tuple[str, list[str]]:
    """Remove consecutive paragraphs that are identical or near-identical."""
    paragraphs = text.split("\n\n")
    cleaned = []
    issues = []

    for i, para in enumerate(paragraphs):
        stripped = para.strip()
        if not stripped:
            cleaned.append(para)
            continue

        # Check against previous non-empty paragraph
        if cleaned:
            prev = cleaned[-1].strip()
            if prev and stripped == prev:
                issues.append(f"Removed duplicate paragraph: '{stripped[:80]}...'")
                continue

            # Near-duplicate: same first 100 chars
            if prev and len(stripped) > 100 and len(prev) > 100:
                if stripped[:100] == prev[:100]:
                    issues.append(f"Removed near-duplicate paragraph: '{stripped[:80]}...'")
                    continue

        cleaned.append(para)

    return "\n\n".join(cleaned), issues


def _fix_duplicate_bylines(text: str) -> tuple[str, list[str]]:
    """
    Remove duplicate author bylines.
    Catches patterns like:
      *By Author Name*
      ## Author Name
      ### Author Name
    Where the same name appears multiple times near the top.
    """
    lines = text.split("\n")
    issues = []

    # Extract all names that appear in heading/byline format in first 20 lines
    names_seen = Counter()
    byline_lines = {}  # line_index -> name

    for i, line in enumerate(lines[:30]):
        stripped = line.strip()

        # Check for byline patterns: *By Name*, **By Name**
        byline_match = re.match(r"^\*+\s*[Bb]y\s+(.+?)\s*\*+$", stripped)
        if byline_match:
            name = byline_match.group(1).strip()
            names_seen[name] += 1
            byline_lines[i] = name
            continue

        # Check for heading with just a name (no other content)
        heading_match = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if heading_match:
            name = heading_match.group(1).strip()
            # Only track short headings that look like names (1-4 words)
            if 1 <= len(name.split()) <= 4 and not any(c in name for c in ".,;:!?()[]"):
                names_seen[name] += 1
                byline_lines[i] = name

    # Remove duplicate occurrences (keep first)
    seen_names = set()
    remove_lines = set()
    for i in sorted(byline_lines.keys()):
        name = byline_lines[i]
        # Normalize for comparison (case-insensitive)
        name_lower = name.lower()
        if name_lower in seen_names:
            remove_lines.add(i)
            issues.append(f"Removed duplicate byline: '{lines[i].strip()}'")
        else:
            seen_names.add(name_lower)

    if remove_lines:
        lines = [l for i, l in enumerate(lines) if i not in remove_lines]

    return "\n".join(lines), issues


# Phrases that indicate the LLM wrote ABOUT the text instead of rewriting it
_META_PREFIXES = [
    "the passage argues",
    "the passage discusses",
    "the passage explains",
    "the passage describes",
    "the passage states",
    "the text argues",
    "the text discusses",
    "the text explains",
    "the text describes",
    "the text states",
    "the author argues",
    "the author discusses",
    "the author explains",
    "the author describes",
    "the author states",
    "this paragraph argues",
    "this paragraph discusses",
    "this paragraph explains",
    "this section argues",
    "this section discusses",
    "this section explains",
    "in this passage",
    "in this section",
    "in this paragraph",
    "the above passage",
    "the above text",
    "to summarize,",
    "in summary,",
    "overall, the passage",
    "overall, the text",
    "overall, the author",
]


def _fix_meta_summaries(text: str) -> tuple[str, list[str]]:
    """
    Remove paragraphs that are LLM meta-summaries (writing about the text
    rather than rewriting it).

    Detects paragraphs starting with phrases like:
      - "The passage argues that..."
      - "The text discusses..."
      - "The author states that..."
    """
    paragraphs = text.split("\n\n")
    cleaned = []
    issues = []

    for para in paragraphs:
        stripped = para.strip()

        # Skip empty, headings, footnotes, horizontal rules
        if not stripped or stripped.startswith("#") or stripped.startswith("[^") or stripped == "---":
            cleaned.append(para)
            continue

        # Check if paragraph starts with a meta-commentary phrase
        lower = stripped.lower()
        is_meta = False
        for prefix in _META_PREFIXES:
            if lower.startswith(prefix):
                is_meta = True
                break

        if is_meta:
            issues.append(f"Removed meta-summary: '{stripped[:80]}...'")
            continue  # Drop the paragraph

        cleaned.append(para)

    return "\n\n".join(cleaned), issues


def _sentence_word_set(sentence: str) -> set:
    """Extract a set of meaningful words from a sentence (lowercased, no punctuation)."""
    words = re.sub(r'[^\w\s]', '', sentence.lower()).split()
    # Remove very common words to focus on content overlap
    stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'about', 'that',
            'this', 'it', 'its', 'and', 'or', 'but', 'not', 'no', 'if', 'so',
            'than', 'too', 'very', 'just', 'also'}
    return {w for w in words if w not in stop and len(w) > 2}


def _fix_repeated_sentences(text: str) -> tuple[str, list[str]]:
    """
    Remove sentences within paragraphs that repeat information already stated
    in a nearby sentence (within 3 sentences).

    Uses word overlap to detect semantic repetition without LLM calls.
    Two sentences with >60% content word overlap are considered repetitive.
    """
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []
    issues = []

    for para in paragraphs:
        stripped = para.strip()

        # Skip headings, footnotes, short items
        if not stripped or stripped.startswith("#") or stripped.startswith("[^") or stripped == "---":
            cleaned_paragraphs.append(para)
            continue

        # Split into sentences (rough but effective)
        sentences = re.split(r'(?<=[.!?])\s+', stripped)

        if len(sentences) <= 1:
            cleaned_paragraphs.append(para)
            continue

        kept = [sentences[0]]
        kept_word_sets = [_sentence_word_set(sentences[0])]

        for i in range(1, len(sentences)):
            current_words = _sentence_word_set(sentences[i])

            if len(current_words) < 3:
                # Too short to judge, keep it
                kept.append(sentences[i])
                kept_word_sets.append(current_words)
                continue

            # Check against the last 3 kept sentences
            is_repeat = False
            for j in range(max(0, len(kept) - 3), len(kept)):
                prev_words = kept_word_sets[j]
                if len(prev_words) < 3:
                    continue

                # Calculate word overlap
                overlap = current_words & prev_words
                # Overlap relative to the smaller sentence
                smaller = min(len(current_words), len(prev_words))
                if smaller > 0 and len(overlap) / smaller >= 0.6:
                    is_repeat = True
                    issues.append(
                        f"Removed repeated sentence: '{sentences[i][:70]}...'"
                    )
                    break

            if not is_repeat:
                kept.append(sentences[i])
                kept_word_sets.append(current_words)

        cleaned_paragraphs.append(" ".join(kept))

    return "\n\n".join(cleaned_paragraphs), issues
