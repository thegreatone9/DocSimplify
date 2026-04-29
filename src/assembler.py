from __future__ import annotations

"""
Output Assembly Module
======================
Stitches simplified chunks back into a coherent Markdown book.
Handles overlap trimming, chapter structure, TOC generation, and
quality comparison metrics.
"""

import re
from pathlib import Path


def trim_overlaps(
    simplified_chunks: list[dict],
    overlap_tokens: int = 200,
) -> list[dict]:
    """
    Remove duplicate text from chunk boundaries caused by the overlap strategy.

    Each chunk (except the first in a chapter) starts with ~overlap_tokens of
    repeated text from the previous chunk. This function detects and removes
    those overlapping prefixes so the assembled book doesn't have duplicated
    paragraphs.

    Strategy: Use difflib to find the longest common prefix between the end
    of the previous simplified text and the start of the current one, then
    trim it.

    Args:
        simplified_chunks: List of chunk dicts with "simplified_text" key.
        overlap_tokens:    Number of overlap tokens that were used during chunking.

    Returns:
        List of chunk dicts with overlap-trimmed "simplified_text".
    """
    if not simplified_chunks:
        return []

    trimmed = []
    overlap_chars = overlap_tokens * 4  # approximate chars for the overlap region

    for i, chunk in enumerate(simplified_chunks):
        chunk_copy = dict(chunk)

        if i == 0 or chunk.get("is_chapter_start", False):
            # First chunk or chapter start — no overlap to trim
            trimmed.append(chunk_copy)
            continue

        current_text = chunk_copy.get("simplified_text", "")
        prev_text = trimmed[-1].get("simplified_text", "")

        if not prev_text or not current_text:
            trimmed.append(chunk_copy)
            continue

        # Get the tail of the previous chunk and try to find it at the
        # beginning of the current chunk
        prev_tail = prev_text[-overlap_chars:] if len(prev_text) > overlap_chars else prev_text

        # Split both into paragraphs for comparison
        prev_paragraphs = [p.strip() for p in prev_tail.split("\n\n") if p.strip()]
        curr_paragraphs = [p.strip() for p in current_text.split("\n\n") if p.strip()]

        if not prev_paragraphs or not curr_paragraphs:
            trimmed.append(chunk_copy)
            continue

        # Find how many leading paragraphs of current_text match trailing
        # paragraphs of prev_text (using fuzzy string matching)
        trim_count = 0
        for j, curr_para in enumerate(curr_paragraphs):
            # Check if this paragraph is similar to any of the last few
            # paragraphs of the previous chunk
            curr_words = set(curr_para.lower().split()[:15])  # first 15 words

            for prev_para in prev_paragraphs:
                prev_words = set(prev_para.lower().split()[:15])

                # If >60% of words overlap, it's likely a duplicate paragraph
                if curr_words and prev_words:
                    overlap = len(curr_words & prev_words)
                    max_len = max(len(curr_words), len(prev_words))
                    if overlap / max_len > 0.6:
                        trim_count = j + 1
                        break

            if trim_count <= j:
                # No match for this paragraph — stop trimming
                break

        # Remove the overlapping paragraphs from the beginning
        if trim_count > 0:
            remaining_paragraphs = curr_paragraphs[trim_count:]
            chunk_copy["simplified_text"] = "\n\n".join(remaining_paragraphs)
        
        trimmed.append(chunk_copy)

    return trimmed


def assemble_book(
    simplified_chunks: list[dict],
    chapters_metadata: list[dict],
    metadata: dict | None = None,
) -> str:
    """
    Stitch simplified chunks into a complete Markdown document.

    Fixed output format:
      1. Title + Author (if available from PDF metadata)
      2. Main content with chapter/section headings (if the original had them)
      3. (TOC and footnotes are added externally by the caller)

    The "Full Document" fallback label is suppressed — if there's only one
    section with no real heading, the content is output without a chapter heading.

    Args:
        simplified_chunks: Overlap-trimmed simplified chunk dicts.
        chapters_metadata: Original chapter structure from the ingestion step.
        metadata:          PDF metadata dict with "title", "author", etc.

    Returns:
        Complete simplified book as a Markdown string.
    """
    if not simplified_chunks:
        return ""

    metadata = metadata or {}
    parts = []

    # ── Title & Author header ────────────────────────────────────────────────
    # metadata["title"] and metadata["author"] come from the raw-page scan
    # which extracts them from visible content on actual PDF pages.
    title = metadata.get("title")
    author = metadata.get("author")

    if title:
        parts.append(f"# {title}\n")
        if author:
            parts.append(f"*By {author}*\n")
        # Insert reader-facing introduction if available
        intro = metadata.get("intro")
        if intro:
            parts.append(f"\n> {intro}\n")
        parts.append("---\n")

    # ── Validate chapter names against scan-derived sections ──────────────────
    # The scan extracted clean section headings from the document.
    # Use these to filter out broken chapter names from detect_chapters().
    scan_sections = set(metadata.get("sections", []))

    # ── Check if document has real structure ──────────────────────────────────
    # If all chunks share the same chapter name (especially "Full Document"),
    # the original had no headings. Don't insert fake chapter headings.
    unique_chapters = set(c.get("chapter", "") for c in simplified_chunks)
    has_real_chapters = not (
        len(unique_chapters) <= 1
        and any(ch in ("Full Document", "") for ch in unique_chapters)
    )

    # If the scan found section headings, use them to validate chapter names
    # A chapter name is valid if it matches a scan-detected heading
    def _is_valid_heading(name: str) -> bool:
        if not name or name == "Full Document":
            return False
        # If scan detected sections, only allow matching headings
        if scan_sections:
            return name in scan_sections
        # No scan data — allow all non-trivial headings but filter fragments
        # Fragments: start lowercase, end with '.', contain footnote markers
        if name[0].islower():
            return False
        if name.endswith("."):
            return False
        if "[^" in name or "[" in name:
            return False
        return True

    # ── Main content ─────────────────────────────────────────────────────────
    current_chapter = None

    for chunk in simplified_chunks:
        chapter_name = chunk.get("chapter", "")
        section_name = chunk.get("section", "")
        is_chapter_start = chunk.get("is_chapter_start", False)
        text = chunk.get("simplified_text", "")

        if not text.strip():
            continue

        # Skip FRONT_MATTER chunks when we already emitted title/author from metadata
        section_type = chunk.get("section_type", "")
        if section_type == "FRONT_MATTER" and title:
            continue

        # Only insert chapter/section headings if the original had them
        if has_real_chapters:
            if chapter_name != current_chapter:
                current_chapter = chapter_name
                # Only emit validated headings — skip broken fragments
                if _is_valid_heading(chapter_name) and chapter_name != title:
                    parts.append(f"\n\n## {chapter_name}\n")


                if section_name and _is_valid_heading(section_name):
                    parts.append(f"\n### {section_name}\n")

            elif section_name and is_chapter_start and _is_valid_heading(section_name):
                parts.append(f"\n### {section_name}\n")

        parts.append(text)

    book_text = "\n\n".join(parts)

    # Clean up excessive whitespace
    book_text = re.sub(r"\n{4,}", "\n\n\n", book_text)

    return book_text.strip()


def generate_toc(markdown_text: str, force: bool = False) -> str:
    """
    Auto-generate a Markdown Table of Contents from heading structure.

    Only generates a TOC if:
      - force=True, AND
      - The document has more than 5 section headings, AND
      - The document is longer than ~3 pages (~1500 words)

    Args:
        markdown_text: The assembled Markdown book.
        force:         If True, allow TOC generation (still gated by length/heading checks).

    Returns:
        A Markdown TOC string, or empty string if conditions aren't met.
    """
    if not force:
        return ""

    # Check document length (~500 words per page, need >3 pages)
    word_count = len(markdown_text.split())
    if word_count < 1500:
        return ""

    heading_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(markdown_text))

    # Need more than 5 section headings to justify a TOC
    if len(matches) <= 5:
        return ""

    toc_lines = ["## Table of Contents\n"]

    for match in matches:
        level = len(match.group(1))
        title = match.group(2).strip()

        # Strip footnote markers from display title and anchor
        clean_title = re.sub(r'\[\^\d+\]', '', title).strip()

        # Create anchor link (GitHub-style)
        anchor = re.sub(r"[^\w\s-]", "", clean_title.lower())
        anchor = re.sub(r"\s+", "-", anchor.strip())

        indent = "  " * (level - 2)
        toc_lines.append(f"{indent}- [{clean_title}](#{anchor})")

    return "\n".join(toc_lines) + "\n"



def save_output(markdown_text: str, output_path: str | Path) -> Path:
    """
    Save the final simplified book as a Markdown file.

    Creates parent directories if needed.

    Args:
        markdown_text: The complete simplified book Markdown.
        output_path:   Where to save the file.

    Returns:
        The Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    return output_path


def compare_lengths(
    original_chunks: list[dict],
    simplified_chunks: list[dict],
) -> dict:
    """
    Compare word counts between original and simplified chunks.

    A simplified version should be roughly the same length (90-120%) as the
    original. If it's significantly shorter, information is likely being dropped.
    If it's much longer, the model is being too verbose.

    Args:
        original_chunks:   List of chunk dicts with "text" key.
        simplified_chunks: List of chunk dicts with "simplified_text" key.

    Returns:
        Dict with comparison metrics and flagged chunks.
    """
    original_words = 0
    simplified_words = 0
    per_chunk_ratios = {}
    flagged_chunks = []

    for orig, simp in zip(original_chunks, simplified_chunks):
        chunk_id = orig.get("chunk_id", 0)

        orig_wc = len(orig.get("text", "").split())
        simp_wc = len(simp.get("simplified_text", "").split())

        original_words += orig_wc
        simplified_words += simp_wc

        ratio = simp_wc / orig_wc if orig_wc > 0 else 0
        per_chunk_ratios[chunk_id] = round(ratio, 2)

        # Flag chunks where the ratio is suspicious
        if ratio < 0.7 or ratio > 1.8:
            flagged_chunks.append(chunk_id)

    overall_ratio = simplified_words / original_words if original_words > 0 else 0

    return {
        "original_words": original_words,
        "simplified_words": simplified_words,
        "ratio": round(overall_ratio, 2),
        "per_chunk_ratios": per_chunk_ratios,
        "flagged_chunks": flagged_chunks,
    }


def insert_footnotes(
    markdown_text: str,
    footnotes: dict[str, str],
) -> str:
    """
    Insert Markdown footnote markers into the text and append definitions at the end.

    For each term in the footnotes dict, marks the FIRST occurrence in the text
    with [^N] and collects definitions into a "Notes" section at the bottom.

    Footnotes are numbered sequentially by their position in the text,
    so [^1] is always the first footnote the reader encounters.

    Uses case-insensitive matching with word boundaries so "Rawls" matches
    "rawls" in the lookup but preserves the original casing in the text.

    Args:
        markdown_text: The assembled Markdown book text.
        footnotes:     Dict of {term: concise explanation}.

    Returns:
        Markdown text with footnote markers and a "Notes" section appended.
    """
    if not footnotes:
        return markdown_text

    # Sort terms by length (longest first) to avoid partial match issues
    # e.g., "property-owning democracy" should match before "democracy"
    sorted_terms = sorted(footnotes.keys(), key=len, reverse=True)

    # First pass: find positions and insert temporary markers
    # Use unique temp markers to avoid numbering conflicts
    matches_found = []  # (position_in_text, term, definition, temp_marker)
    text = markdown_text
    temp_counter = 0

    for term in sorted_terms:
        definition = footnotes[term]

        # Build a regex that matches the term with word boundaries, case-insensitive
        escaped = re.escape(term)
        pattern = re.compile(r'(?<!\[)\b(' + escaped + r')\b(?!\])', re.IGNORECASE)

        # Only mark the FIRST occurrence
        match = pattern.search(text)
        if match:
            temp_counter += 1
            temp_marker = f"__FN_TEMP_{temp_counter}__"
            matched_text = match.group(1)
            replacement = f"{matched_text}[{temp_marker}]"
            matches_found.append((match.start(), term, definition, temp_marker))
            text = text[:match.start()] + replacement + text[match.end():]

    if not matches_found:
        return markdown_text

    # Second pass: sort by position in text and renumber sequentially
    matches_found.sort(key=lambda x: x[0])

    used_footnotes = []
    for final_num, (_, term, definition, temp_marker) in enumerate(matches_found, start=1):
        text = text.replace(f"[{temp_marker}]", f"[^{final_num}]")
        used_footnotes.append((final_num, term, definition))

    # Build the footnotes section
    footnote_lines = ["\n\n---\n\n## Notes\n"]
    for num, term, definition in used_footnotes:
        footnote_lines.append(f"[^{num}]: **{term}** — {definition}\n")

    text += "\n".join(footnote_lines)

    return text


# ── QA Layer ─────────────────────────────────────────────────────────────────

def qa_check_output(
    final_markdown: str,
    original_chapters: list[dict],
    doc_metadata: dict,
    doc_structure: dict | None = None,
) -> dict:
    """
    Validate the final output against the original document's structure.

    Checks:
      1. No TOC if the original didn't have one
      2. No invented headings if the original had none
      3. No stale "Full Document" labels
      4. Footnotes section only at the very end
      5. Word count sanity (output shouldn't be <50% or >200% of input)
      6. Enumeration integrity (numbered/bullet items match input)
      7. Structure map validation (if doc_structure provided)

    Args:
        final_markdown:    The assembled, footnoted Markdown output.
        original_chapters: Chapter metadata from the ingestion step.
        doc_metadata:      PDF metadata dict (title, author, has_toc, etc.).
        doc_structure:     Structure map from detect_document_structure().

    Returns:
        Dict with:
          - "passed": bool
          - "issues": list of {"severity": "ERROR"|"WARNING", "message": str}
          - "fixed_markdown": str (auto-corrected version, if fixable)
    """
    issues = []
    fixed = final_markdown
    has_toc = doc_metadata.get("has_toc", False)
    doc_structure = doc_structure or {}

    # ── Count headings in original ───────────────────────────────────────
    original_headings = set()
    for ch in original_chapters:
        if ch.get("chapter") and ch["chapter"] != "Full Document":
            original_headings.add(ch["chapter"])
        if ch.get("section"):
            original_headings.add(ch["section"])

    # Also count headings already present in the output (from VERBATIM paragraph_title blocks)
    # In paddle mode, original_chapters may be empty but real headings exist in the markdown
    output_md_headings = re.findall(r"^#{2,3}\s+(.+)$", final_markdown, re.MULTILINE)
    for h in output_md_headings:
        ht = h.strip()
        if ht not in ("Notes", "Table of Contents"):
            original_headings.add(ht)

    original_had_headings = len(original_headings) > 1

    # (TOC guard removed — generate_toc() now has built-in conditions)

    # ── Check 2: No invented headings if original had none ───────────────
    if not original_had_headings:
        output_headings = re.findall(r"^(#{2,6})\s+(.+)$", fixed, re.MULTILINE)
        # Filter out "Notes" heading (that's ours from footnotes)
        invented = [
            (h, title) for h, title in output_headings
            if title.strip() not in ("Notes", "Table of Contents")
        ]
        if invented:
            issues.append({
                "severity": "ERROR",
                "message": f"Output contains {len(invented)} heading(s) but the original document had none. "
                           f"Examples: {', '.join(t for _, t in invented[:3])}",
            })
            # Auto-fix: convert to bold
            for hashes, title in invented:
                pattern = re.escape(f"{hashes} {title}")
                fixed = re.sub(f"^{pattern}$", f"**{title.strip()}**", fixed, flags=re.MULTILINE)

    # ── Check 3: No "Full Document" label ────────────────────────────────
    if re.search(r"^#{1,3}\s+Full Document\s*$", fixed, re.MULTILINE):
        issues.append({
            "severity": "ERROR",
            "message": "Output contains the internal 'Full Document' fallback label",
        })
        fixed = re.sub(r"^#{1,3}\s+Full Document\s*\n*", "", fixed, flags=re.MULTILINE)

    # ── Check 4: Footnotes section at the end only ───────────────────────
    notes_matches = list(re.finditer(r"^## Notes\s*$", fixed, re.MULTILINE))
    if len(notes_matches) > 1:
        issues.append({
            "severity": "WARNING",
            "message": f"Multiple 'Notes' sections found ({len(notes_matches)}). Should be exactly one at the end.",
        })

    # ── Check 5: Word count sanity ───────────────────────────────────────
    original_words = sum(len(ch.get("text", "").split()) for ch in original_chapters)
    # Strip footnotes section for fair comparison
    content_for_count = re.split(r"^---\s*\n\s*## Notes", fixed, flags=re.MULTILINE)[0]
    output_words = len(content_for_count.split())

    if original_words > 0:
        ratio = output_words / original_words
        if ratio < 0.5:
            issues.append({
                "severity": "WARNING",
                "message": f"Output is very short ({ratio:.0%} of original). Possible excessive summarization.",
            })
        elif ratio > 2.0:
            issues.append({
                "severity": "WARNING",
                "message": f"Output is very long ({ratio:.0%} of original). Possible over-explanation.",
            })

    # ── Check 6: Enumeration integrity ───────────────────────────────────
    original_text = "\n".join(ch.get("text", "") for ch in original_chapters)
    orig_numbered = len(re.findall(r"^\d+[\.\)]\s", original_text, re.MULTILINE))
    out_numbered = len(re.findall(r"^\d+[\.\)]\s", content_for_count, re.MULTILINE))
    if orig_numbered > 0 or out_numbered > 0:
        if out_numbered > orig_numbered * 1.5 + 2:
            issues.append({
                "severity": "WARNING",
                "message": f"Numbered items inflated: input={orig_numbered}, output={out_numbered}",
            })
        elif out_numbered < orig_numbered * 0.5:
            issues.append({
                "severity": "WARNING",
                "message": f"Numbered items lost: input={orig_numbered}, output={out_numbered}",
            })

    orig_bullets = len(re.findall(r"^[-*•]\s", original_text, re.MULTILINE))
    out_bullets = len(re.findall(r"^[-*•]\s", content_for_count, re.MULTILINE))
    if orig_bullets > 0 or out_bullets > 0:
        if out_bullets > orig_bullets * 2 + 3:
            issues.append({
                "severity": "WARNING",
                "message": f"Bullet items inflated: input={orig_bullets}, output={out_bullets}",
            })

    # ── Check 7: Structure map validation ────────────────────────────────
    if doc_structure.get("segment_type") == "numbered":
        expected_major = doc_structure.get("total_major_segments", 0)
        expected_nums = [p["number"] for p in doc_structure.get("major_numbered_points", [])]

        if expected_major > 0:
            # Count major numbered items in output (items that start a paragraph)
            out_major = re.findall(r"^\d+[\.\)]\s", content_for_count, re.MULTILINE)
            out_count = len(out_major)

            if out_count > expected_major + 2:
                issues.append({
                    "severity": "WARNING",
                    "message": f"Structure mismatch: expected ~{expected_major} major numbered points "
                               f"(from structure map: {', '.join(expected_nums)}), "
                               f"but output has {out_count}",
                })
            elif out_count < expected_major - 2:
                issues.append({
                    "severity": "WARNING",
                    "message": f"Structure mismatch: expected ~{expected_major} major numbered points, "
                               f"but output only has {out_count} (possible merging)",
                })

    passed = not any(i["severity"] == "ERROR" for i in issues)

    return {
        "passed": passed,
        "issues": issues,
        "fixed_markdown": fixed,
    }
