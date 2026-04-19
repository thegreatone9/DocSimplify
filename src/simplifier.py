from __future__ import annotations

"""
Simplification Engine
======================
Orchestrates the LLM simplification pipeline with:
  - Auto-retry on short outputs (items 1-3)
  - Verification pass for info-loss detection (item 4)
  - Parallel chunk processing (item 9)
  - tqdm progress bars (item 10)
  - JSON checkpoint save/resume (item 11)
"""

import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.llm_client import OllamaClient
from src.prompts import (
    build_simplification_prompt,
    build_paragraph_prompt,
    build_bridge_summary_prompt,
    build_smoothing_prompt,
    build_verification_prompt,
    build_footnote_prompt,
)


# ═══════════════════════════════════════════════════════════════════════
# Non-prose element preservation (images, tables, equations)
# ═══════════════════════════════════════════════════════════════════════

# Regex patterns for non-prose elements in markdown
_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\([^\)]+\)')          # ![alt](path)
_TABLE_ROW_RE = re.compile(r'^\|.+\|$', re.MULTILINE)      # |col|col|
_LATEX_BLOCK_RE = re.compile(r'\$\$[^$]+\$\$', re.DOTALL)  # $$...$$
_LATEX_INLINE_RE = re.compile(r'(?<!\$)\$(?!\$)[^$]+\$(?!\$)')  # $...$


def strip_non_prose_elements(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace images, tables, and equations with placeholders.

    Returns:
        (text_with_placeholders, placeholder_map)
        where placeholder_map maps "<<IMG_1>>" -> original markdown.
    """
    placeholders = {}
    counter = {"IMG": 0, "TABLE": 0, "EQ": 0}

    # 1. Images: ![alt](path)
    def _replace_image(m):
        counter["IMG"] += 1
        key = f"<<IMG_{counter['IMG']}>>"
        placeholders[key] = m.group(0)
        return key
    text = _IMAGE_RE.sub(_replace_image, text)

    # 2. Tables: consecutive lines starting/ending with |
    # Find contiguous blocks of table rows (including header separator |---|---|)
    lines = text.split("\n")
    table_lines = []
    in_table = False
    table_start = -1

    for i, line in enumerate(lines):
        is_table_line = bool(re.match(r'^\s*\|.*\|\s*$', line))
        if is_table_line and not in_table:
            in_table = True
            table_start = i
        elif not is_table_line and in_table:
            # End of table block
            if i - table_start >= 2:  # At least header + separator
                table_lines.append((table_start, i))
            in_table = False

    if in_table and len(lines) - table_start >= 2:
        table_lines.append((table_start, len(lines)))

    # Replace table blocks from bottom to top (to keep indices valid)
    for start, end in reversed(table_lines):
        counter["TABLE"] += 1
        key = f"<<TABLE_{counter['TABLE']}>>"
        original = "\n".join(lines[start:end])
        placeholders[key] = original
        lines[start:end] = [key]

    text = "\n".join(lines)

    # 3. LaTeX block equations: $$...$$
    def _replace_block_eq(m):
        counter["EQ"] += 1
        key = f"<<EQ_{counter['EQ']}>>"
        placeholders[key] = m.group(0)
        return key
    text = _LATEX_BLOCK_RE.sub(_replace_block_eq, text)

    # 4. LaTeX inline equations: $...$
    def _replace_inline_eq(m):
        counter["EQ"] += 1
        key = f"<<EQ_{counter['EQ']}>>"
        placeholders[key] = m.group(0)
        return key
    text = _LATEX_INLINE_RE.sub(_replace_inline_eq, text)

    return text, placeholders


def reinsert_non_prose_elements(text: str, placeholders: dict[str, str]) -> str:
    """Swap placeholders back with original content."""
    for key, original in placeholders.items():
        text = text.replace(key, original)
    return text


def _is_reference_chunk(text: str) -> bool:
    """
    Detect if a chunk is primarily references, bibliography, or endnotes.

    Uses multiple heuristics:
    1. Citation entry patterns (author-year, page ranges, publishers)
    2. Line structure (bibliography lines are shorter, start with author names)
    3. Low ratio of connective/narrative words (references don't argue)
    """
    lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 10]
    if not lines:
        return False

    # Heuristic 1: Bibliography-style line starters
    # Reference entries typically start with author names or bullets/dashes
    bib_start_patterns = [
        r'^[A-Z][a-z]+,\s+[A-Z]\.',       # "Marx, K."
        r'^[A-Z][a-z]+,\s+[A-Z][a-z]+',   # "Cockshott, Paul"
        r'^-\s+',                           # "- footnote text"
        r'^\d+\.\s+[A-Z]',                 # "1. Reference"
    ]

    # Heuristic 2: Citation content patterns (need 2+ per line)
    citation_patterns = [
        r'\(\d{4}\)',           # (1997)
        r'\d{4}\.',            # 1997.
        r'\d{4}\)',            # 1997)
        r'pp\.\s*\d+',         # pp. 123
        r'vol\.\s*\d+',        # vol. 5
        r'[Ee]d[s]?\.',        # ed. / eds.
        r'[Pp]ress',           # University Press
        r'[Jj]ournal\s+of',    # Journal of
        r':\s*\d+[-\u2013]\d+',  # : 123-456 (page ranges)
    ]

    bib_line_count = 0
    for line in lines:
        # Check if line starts like a bibliography entry
        starts_like_bib = any(re.search(p, line) for p in bib_start_patterns)
        # Check citation content
        content_matches = sum(1 for p in citation_patterns if re.search(p, line))

        # A line is bibliographic if it starts like a reference AND has 2+ citation markers,
        # OR it has 4+ citation markers (definitely a reference, not body text)
        if (starts_like_bib and content_matches >= 2) or content_matches >= 4:
            bib_line_count += 1

    ratio = bib_line_count / len(lines)
    return ratio > 0.4



def _simplify_one_chunk(
    chunk: dict,
    llm: OllamaClient,
    book_summary: str,
    glossary: dict,
    temperature: float,
    min_ratio: float,
    max_ratio: float,
    max_retries: int,
    previous_context: str = "",
) -> dict:
    """
    Simplify a single chunk with automatic retry if the output is too short/long.

    Returns a dict with:
      - "simplified_text": str
      - "attempts": int  (how many attempts it took)
      - "final_ratio": float
      - "was_retried": bool
    """
    original_word_count = len(chunk["text"].split())

    # Skip near-empty chunks (e.g., "BY" from author bylines)
    if original_word_count < 5:
        return {
            "simplified_text": chunk["text"],
            "attempts": 1,
            "final_ratio": 1.0,
            "was_retried": False,
        }

    # Use LLM-based section labels to skip non-body sections
    section_type = chunk.get("section_type", "")
    passthrough_types = {"FRONT_MATTER", "VERBATIM", "REFERENCES", "FOOTNOTES", "APPENDIX"}
    if section_type in passthrough_types:
        print(f"\n     📚 Skipping {section_type} chunk (pass-through)")
        return {
            "simplified_text": chunk["text"],
            "attempts": 1,
            "final_ratio": 1.0,
            "was_retried": False,
        }

    # Fallback: regex-based reference detection when no label exists
    if not section_type and _is_reference_chunk(chunk["text"]):
        print(f"\n     📚 Skipping reference/bibliography chunk (regex fallback)")
        return {
            "simplified_text": chunk["text"],
            "attempts": 1,
            "final_ratio": 1.0,
            "was_retried": False,
        }

    # ── Strip non-prose elements (images, tables, equations) ─────────────
    chunk_text, non_prose_map = strip_non_prose_elements(chunk["text"])
    if non_prose_map:
        print(f"     📎 Preserved {len(non_prose_map)} non-prose element(s)")

    original_word_count = len(chunk_text.split())

    best_result = ""
    best_ratio = 0.0
    attempts = 0

    for attempt in range(1, max_retries + 2):  # +2: 1 initial + max_retries retries
        attempts = attempt
        is_retry = attempt > 1

        sys_prompt, usr_prompt = build_simplification_prompt(
            chunk_text=chunk_text,
            book_summary=book_summary,
            glossary=glossary,
            previous_context=previous_context,
            is_retry=is_retry,
        )

        try:
            result = llm.generate(
                prompt=usr_prompt,
                system_prompt=sys_prompt,
                temperature=temperature,
            )
        except Exception as e:
            # On error, keep any previous best result
            if best_result:
                break
            raise

        simplified_word_count = len(result.split())
        ratio = simplified_word_count / original_word_count if original_word_count > 0 else 1.0

        # Keep the best result so far (closest to 1.0 ratio)
        if abs(ratio - 1.0) < abs(best_ratio - 1.0) or not best_result:
            best_result = result
            best_ratio = ratio

        # Check if ratio is within acceptable bounds
        if min_ratio <= ratio <= max_ratio:
            break  # Acceptable — stop retrying

    # Strip any headings/bold labels the LLM invented
    best_result = _clean_llm_output(best_result, original_text=chunk_text)

    # ── Reinsert non-prose elements ──────────────────────────────────────
    if non_prose_map:
        best_result = reinsert_non_prose_elements(best_result, non_prose_map)

    return {
        "simplified_text": best_result,
        "attempts": attempts,
        "final_ratio": round(best_ratio, 2),
        "was_retried": attempts > 1,
    }


def _clean_llm_output(text: str, original_text: str) -> str:
    """
    Post-process LLM output to remove structural artifacts the model invented.

    Strips:
      - Markdown headings (### lines) that don't exist in the original
      - Standalone bold lines acting as fake section titles (e.g., **Some Title**)
        that don't exist in the original
      - Labels like "Simplified version:" or "Here is the simplified text:"

    The original text is checked to determine what's legitimate.
    """
    import re

    # Collect legitimate bold lines and headings from the original
    original_bold_lines = set()
    original_headings = set()
    for line in original_text.split("\n"):
        stripped = line.strip()
        if re.match(r"^\*\*.*\*\*$", stripped):
            original_bold_lines.add(stripped)
        heading_m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_m:
            original_headings.add(heading_m.group(2).strip())

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()

        # Strip invented Markdown headings
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            title = heading_match.group(2).strip()
            if title in original_headings:
                cleaned.append(line)  # Keep — it was in the original
            else:
                # Drop entirely — model invented it
                continue

        # Strip invented standalone bold labels (whole line is just **Title**)
        elif re.match(r"^\*\*[^*]+\*\*$", stripped) and len(stripped) < 120:
            if stripped in original_bold_lines:
                cleaned.append(line)  # Keep — it was in the original
            else:
                # Drop — model invented a section break
                continue

        # Strip meta-commentary labels
        elif re.match(r"^(Simplified|Here is|Below is|The simplified)", stripped, re.IGNORECASE):
            continue

        else:
            cleaned.append(line)

    result = "\n".join(cleaned)
    # Clean up excessive blank lines from removals
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def _inline_qa_check(result: dict, original_text: str) -> dict:
    """
    Run QA on a single chunk's output. Returns a dict of issues found.
    Called immediately after every chunk is simplified.
    """
    import re
    text = result.get("simplified_text", "")
    issues = []

    # Check for remaining headings
    headings = re.findall(r"^#{1,6}\s+.+$", text, re.MULTILINE)
    if headings:
        issues.append(f"Contains {len(headings)} heading(s)")

    # Check for invented bold labels not in original
    original_bolds = set()
    for line in original_text.split("\n"):
        s = line.strip()
        if re.match(r"^\*\*[^*]+\*\*$", s):
            original_bolds.add(s)

    invented_bolds = []
    for line in text.split("\n"):
        s = line.strip()
        if re.match(r"^\*\*[^*]+\*\*$", s) and len(s) < 120 and s not in original_bolds:
            invented_bolds.append(s)
    if invented_bolds:
        issues.append(f"Contains {len(invented_bolds)} invented bold label(s)")

    # Check for enumeration mismatch (numbered/bulleted items)
    orig_numbered = re.findall(r"^\d+[\.\)]\s", original_text, re.MULTILINE)
    out_numbered = re.findall(r"^\d+[\.\)]\s", text, re.MULTILINE)
    if orig_numbered or out_numbered:
        orig_count = len(orig_numbered)
        out_count = len(out_numbered)
        if out_count > orig_count + 1:  # +1 tolerance for sub-items
            issues.append(f"Numbered items: input={orig_count}, output={out_count} (inflated)")
        elif out_count < orig_count - 1:
            issues.append(f"Numbered items: input={orig_count}, output={out_count} (missing)")

    orig_bullets = re.findall(r"^[-*•]\s", original_text, re.MULTILINE)
    out_bullets = re.findall(r"^[-*•]\s", text, re.MULTILINE)
    if orig_bullets or out_bullets:
        orig_b = len(orig_bullets)
        out_b = len(out_bullets)
        if out_b > orig_b + 2:
            issues.append(f"Bullet items: input={orig_b}, output={out_b} (inflated)")
        elif out_b < orig_b - 2:
            issues.append(f"Bullet items: input={orig_b}, output={out_b} (missing)")

    # Check word ratio
    ratio = result.get("final_ratio", 1.0)
    if ratio < 0.5:
        issues.append(f"Severely shortened ({ratio:.0%} of original)")
    elif ratio > 2.0:
        issues.append(f"Severely bloated ({ratio:.0%} of original)")

    result["qa_issues"] = issues
    result["qa_passed"] = len(issues) == 0
    return result


def _qa_progress_report(checkpoint: dict, total_chunks: int, label: str = ""):
    """
    Print a QA progress report. Called every 10% of chunks processed.
    """
    done = len(checkpoint)
    pct = (done / total_chunks * 100) if total_chunks > 0 else 0

    qa_failures = sum(
        1 for entry in checkpoint.values()
        if entry.get("qa_issues")
    )
    avg_ratio = 0
    ratios = [e.get("final_ratio", 1.0) for e in checkpoint.values() if "final_ratio" in e]
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)

    print(f"\n   📊 QA checkpoint ({pct:.0f}% done — {done}/{total_chunks} chunks)")
    print(f"      Avg word ratio: {avg_ratio:.2f}x")
    print(f"      Chunks with QA issues: {qa_failures}/{done}")
    if qa_failures > 0:
        # Show the most common issue types
        all_issues = []
        for entry in checkpoint.values():
            all_issues.extend(entry.get("qa_issues", []))
        from collections import Counter
        top_issues = Counter(all_issues).most_common(3)
        for issue, count in top_issues:
            print(f"      • {issue} (×{count})")


def _merge_similar_paragraphs(paragraphs: list[str], threshold: float = 0.85) -> list[str]:
    """
    Merge consecutive paragraphs that are semantically similar.
    Uses sentence-transformers embeddings — no LLM calls needed.

    Args:
        paragraphs: List of paragraph strings.
        threshold:  Cosine similarity threshold for merging (0.85 = very similar).

    Returns:
        List of paragraphs with similar consecutive ones merged.
    """
    if len(paragraphs) <= 1:
        return paragraphs

    try:
        from src.embedding_qa import check_semantic_similarity, is_available
        if not is_available():
            return paragraphs
    except ImportError:
        return paragraphs

    merged = [paragraphs[0]]

    for i in range(1, len(paragraphs)):
        current = paragraphs[i]
        previous = merged[-1]

        # Skip very short paragraphs (headers, etc.) — don't merge
        if len(current.split()) < 5 or len(previous.split()) < 5:
            merged.append(current)
            continue

        similarity = check_semantic_similarity(previous, current)

        if similarity is not None and similarity >= threshold:
            # Merge: join with a space (they're about the same topic)
            merged[-1] = previous + " " + current
        else:
            merged.append(current)

    if len(merged) < len(paragraphs):
        print(f"     🔗 Merged {len(paragraphs)} → {len(merged)} paragraphs (similar content combined)")

    return merged


def _realign_paragraphs(original_text: str, simplified_text: str) -> str:
    """
    Realign simplified paragraph structure to match the original.

    If the LLM split one input paragraph into multiple output paragraphs,
    this merges output paragraphs to restore the original's structure.

    Strategy: Distribute output paragraphs proportionally across input
    paragraphs based on word count ratios, then merge.

    Args:
        original_text:   The original chunk text.
        simplified_text: The simplified chunk text (potentially paragraph-bloated).

    Returns:
        Simplified text with paragraph structure matching the original.
    """
    orig_paras = [p.strip() for p in original_text.split("\n\n") if p.strip()]
    simp_paras = [p.strip() for p in simplified_text.split("\n\n") if p.strip()]

    n_orig = len(orig_paras)
    n_simp = len(simp_paras)

    # If output has fewer or equal paragraphs, no realignment needed
    if n_simp <= n_orig or n_orig == 0:
        return simplified_text

    # Calculate word counts of original paragraphs to get proportions
    orig_word_counts = [len(p.split()) for p in orig_paras]
    total_orig_words = sum(orig_word_counts)

    if total_orig_words == 0:
        return simplified_text

    # Distribute output paragraphs proportionally
    merged = []
    simp_idx = 0

    for i, orig_wc in enumerate(orig_word_counts):
        # How many output paragraphs should map to this input paragraph?
        proportion = orig_wc / total_orig_words
        # At least 1, distribute remaining proportionally
        n_for_this = max(1, round(proportion * n_simp))

        # Last input paragraph gets all remaining output paragraphs
        if i == n_orig - 1:
            n_for_this = n_simp - simp_idx

        # Gather the output paragraphs for this slot
        slot = simp_paras[simp_idx: simp_idx + n_for_this]
        simp_idx += n_for_this

        # Merge them into one paragraph (join with space)
        merged.append(" ".join(slot))

    if len(merged) < n_simp:
        print(f"     📐 Realigned paragraphs: {n_simp} → {len(merged)} (matching input structure)")

    return "\n\n".join(merged)


def _simplify_one_chunk_paragraphs(
    chunk: dict,
    llm,
    book_summary: str,
    glossary: dict,
    temperature: float,
    previous_context: str = "",
) -> dict:
    """
    Simplify a chunk by processing each paragraph individually.
    Much simpler task per LLM call = better results from weaker models.
    """
    import re

    original_word_count = len(chunk["text"].split())

    # Skip near-empty chunks
    if original_word_count < 5:
        return {
            "simplified_text": chunk["text"],
            "attempts": 1,
            "final_ratio": 1.0,
            "was_retried": False,
        }

    # Use LLM-based section labels to skip non-body sections
    section_type = chunk.get("section_type", "")
    passthrough_types = {"FRONT_MATTER", "VERBATIM", "REFERENCES", "FOOTNOTES", "APPENDIX"}
    if section_type in passthrough_types:
        print(f"\n     📚 Skipping {section_type} chunk (pass-through)")
        return {
            "simplified_text": chunk["text"],
            "attempts": 1,
            "final_ratio": 1.0,
            "was_retried": False,
        }

    # Fallback: regex-based reference detection when no label exists
    if not section_type and _is_reference_chunk(chunk["text"]):
        print(f"\n     📚 Skipping reference/bibliography chunk (regex fallback)")
        return {
            "simplified_text": chunk["text"],
            "attempts": 1,
            "final_ratio": 1.0,
            "was_retried": False,
        }

    # Split into paragraphs
    paragraphs = [p.strip() for p in chunk["text"].split("\n\n") if p.strip()]

    # ── Pre-processing: LLM-based fragment merging ────────────────────────────
    # PDF extraction sometimes splits sentences across paragraph breaks.
    # Use LLM to detect broken boundaries, with regex as fast fallback.
    if len(paragraphs) >= 2:
        try:
            from src.doc_classifier import detect_fragment_boundaries
            boundary_labels = detect_fragment_boundaries(paragraphs, llm)

            merged = [paragraphs[0]]
            for j, label in enumerate(boundary_labels):
                if label == "MERGE":
                    merged[-1] = merged[-1] + " " + paragraphs[j + 1]
                else:
                    merged.append(paragraphs[j + 1])
            paragraphs = merged
        except Exception:
            # Fallback: regex-based merge (starts lowercase = continuation)
            merged = []
            for para in paragraphs:
                stripped = para.lstrip("- \u20220123456789.)")
                is_continuation = (stripped and stripped[0].islower())
                if merged and is_continuation:
                    merged[-1] = merged[-1] + " " + para
                else:
                    merged.append(para)
            paragraphs = merged

    simplified_paragraphs = []
    para_context = previous_context

    for i, para in enumerate(paragraphs):
        # Skip very short paragraphs (headings, bylines, orphans)
        if len(para.split()) < 3:
            simplified_paragraphs.append(para)
            continue

        # Strip non-prose elements before sending to LLM
        para_text, para_non_prose = strip_non_prose_elements(para)

        # If the paragraph is entirely non-prose (image-only, table-only), pass through
        if para_text.strip() == "" or all(c in " \n" for c in para_text.replace("<<", "").replace(">>", "")):
            simplified_paragraphs.append(para)
            continue

        if para_non_prose:
            print(f"       📎 Para {i+1}: preserved {len(para_non_prose)} non-prose element(s)")

        sys_prompt, usr_prompt = build_paragraph_prompt(
            paragraph_text=para_text,
            book_summary=book_summary,
            glossary=glossary,
            previous_context=para_context,
        )

        try:
            result = llm.generate(
                prompt=usr_prompt,
                system_prompt=sys_prompt,
                temperature=temperature,
            )
            # Clean the result
            result = result.strip()
            # Remove any meta-labels the model adds (lightweight regex first pass)
            result = re.sub(r"^(Rewritten|Simplified|Here is)[:\s].*?\n", "", result, flags=re.IGNORECASE)
            # Reinsert non-prose elements
            if para_non_prose:
                result = reinsert_non_prose_elements(result, para_non_prose)
            simplified_paragraphs.append(result)
            # Use last simplified paragraph as context for next
            para_context = result[-200:]
        except Exception as e:
            # Log the error so we can see what's failing
            print(f"\n     ⚠️  Paragraph failed: {str(e)[:120]}")
            simplified_paragraphs.append(para)

        # Throttle between paragraphs to stay under TPM limits
        import time
        time.sleep(5)

    # ── Post-processing: LLM-based meta-commentary detection ──────────────────
    # Check if any simplified paragraphs talk ABOUT the text instead of
    # rewriting it. Re-simplify those with a stricter prompt.
    try:
        from src.doc_classifier import detect_meta_commentary
        meta_flags = detect_meta_commentary(simplified_paragraphs, llm)
        meta_count = sum(meta_flags)
        if meta_count > 0:
            print(f"\n     🔍 Detected {meta_count} meta-commentary paragraph(s), re-simplifying...")
            for idx, is_meta in enumerate(meta_flags):
                if is_meta and idx < len(paragraphs):
                    # Re-simplify the original paragraph with stricter instruction
                    sys_prompt, usr_prompt = build_paragraph_prompt(
                        paragraph_text=paragraphs[min(idx, len(paragraphs) - 1)],
                        book_summary=book_summary,
                        glossary=glossary,
                        previous_context="",
                    )
                    # Prepend a strict anti-meta instruction
                    usr_prompt = (
                        "CRITICAL: Rewrite the content directly. Do NOT describe or summarize "
                        "what the text says. Do NOT use phrases like 'this passage discusses', "
                        "'the author argues', or 'this section examines'. Just restate the "
                        "actual ideas in simpler words.\n\n" + usr_prompt
                    )
                    try:
                        result = llm.generate(prompt=usr_prompt, system_prompt=sys_prompt, temperature=temperature)
                        simplified_paragraphs[idx] = result.strip()
                        time.sleep(5)
                    except Exception:
                        pass  # Keep the original if retry fails
    except Exception as e:
        print(f"\n     ⚠️  Meta-commentary check skipped: {str(e)[:80]}")

    # Merge similar consecutive paragraphs using embeddings
    simplified_paragraphs = _merge_similar_paragraphs(simplified_paragraphs)

    simplified_text = "\n\n".join(simplified_paragraphs)
    simplified_text = _clean_llm_output(simplified_text, original_text=chunk["text"])

    # Realign paragraph structure to match input
    simplified_text = _realign_paragraphs(chunk["text"], simplified_text)

    # Smoothing pass: harmonize tone AFTER realignment
    realigned_paras = [p.strip() for p in simplified_text.split("\n\n") if p.strip()]
    if len(realigned_paras) >= 1 and len(simplified_text.split()) > 20:
        try:
            sys_prompt, usr_prompt = build_smoothing_prompt(simplified_text)
            smoothed = llm.generate(
                prompt=usr_prompt,
                system_prompt=sys_prompt,
                temperature=0.2,
            )
            smoothed = smoothed.strip()
            # Only use smoothed version if it's roughly the same length
            smooth_ratio = len(smoothed.split()) / len(simplified_text.split())
            if 0.8 <= smooth_ratio <= 1.3:
                simplified_text = smoothed
        except Exception as e:
            print(f"\n     ℹ️  Smoothing skipped: {str(e)[:80]}")

    # Re-realign in case the smoothing LLM re-split paragraphs
    simplified_text = _realign_paragraphs(chunk["text"], simplified_text)

    simplified_word_count = len(simplified_text.split())
    ratio = simplified_word_count / original_word_count if original_word_count > 0 else 1.0

    return {
        "simplified_text": simplified_text,
        "attempts": 1,
        "final_ratio": round(ratio, 2),
        "was_retried": False,
    }


def _generate_bridge_summary(llm, simplified_text: str) -> str:
    """
    Generate a 2-sentence bridge summary for rolling context compression.
    Returns the summary string, or empty string on failure.
    """
    try:
        sys_prompt, usr_prompt = build_bridge_summary_prompt(simplified_text)
        summary = llm.generate(
            prompt=usr_prompt,
            system_prompt=sys_prompt,
            temperature=0.1,
            max_tokens=200,
        )
        return summary.strip()
    except Exception:
        # Non-critical — fall back to raw text tail
        return simplified_text[-300:]


def simplify_chunks(
    chunks: list[dict],
    llm,
    book_summary: str,
    glossary: dict,
    temperature: float = 0.3,
    min_ratio: float = 0.85,
    max_ratio: float = 1.5,
    max_retries: int = 2,
    max_workers: int = 2,
    checkpoint_path: Path | str | None = None,
    verify: bool = False,
    paragraph_mode: bool = False,
) -> list[dict]:
    """
    Simplify all chunks with retry, parallelism, progress bars, ongoing QA,
    and checkpointing.

    QA runs at three levels:
      1. Per-chunk:  _clean_llm_output strips artifacts after every LLM call
      2. Per-chunk:  _inline_qa_check validates structure after cleaning
      3. Every 10%:  _qa_progress_report prints aggregate QA stats

    Args:
        chunks:          List of chunk dicts with "text", "chunk_id", etc.
        llm:             OllamaClient instance.
        book_summary:    Book-level context string.
        glossary:        Term→definition dict for consistency.
        temperature:     LLM temperature.
        min_ratio:       Min word ratio before auto-retry.
        max_ratio:       Max word ratio before auto-retry.
        max_retries:     How many retry attempts per chunk.
        max_workers:     Concurrent LLM calls (1 = sequential with context).
        checkpoint_path: Path to save/load checkpoint JSON.
        verify:          Run a verification pass after simplification.

    Returns:
        List of chunk dicts with added "simplified_text" key.
    """
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    # ── Load checkpoint ──────────────────────────────────────────────────────
    checkpoint = {}
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        done_count = len(checkpoint)
        remaining = len(chunks) - done_count
        print(f"♻️  Resuming from checkpoint: {done_count}/{len(chunks)} done, {remaining} remaining")
    else:
        print(f"🆕 Starting fresh — {len(chunks)} chunks to process")

    # ── Choose processing mode ───────────────────────────────────────────────
    if paragraph_mode:
        print(f"   📝 Paragraph mode: simplifying paragraph-by-paragraph")
    if max_workers > 1:
        _simplify_parallel(chunks, llm, book_summary, glossary, temperature,
                           min_ratio, max_ratio, max_retries, max_workers,
                           checkpoint, checkpoint_path, paragraph_mode)
    else:
        _simplify_sequential(chunks, llm, book_summary, glossary, temperature,
                             min_ratio, max_ratio, max_retries,
                             checkpoint, checkpoint_path, paragraph_mode)

    # ── Verification pass ────────────────────────────────────────────────────
    if verify:
        _run_verification(chunks, llm, checkpoint)

    # ── Final QA report ──────────────────────────────────────────────────────
    _qa_progress_report(checkpoint, len(chunks), label="FINAL")

    # ── Merge results ────────────────────────────────────────────────────────
    output_chunks = []
    retried_count = 0
    for chunk in chunks:
        cid = str(chunk["chunk_id"])
        entry = checkpoint.get(cid, {})
        simplified_text = entry.get("simplified_text", chunk["text"])
        was_retried = entry.get("was_retried", False)
        if was_retried:
            retried_count += 1

        output_chunks.append({
            **chunk,
            "simplified_text": simplified_text,
        })

    # Print summary
    ratios = [checkpoint[str(c["chunk_id"])].get("final_ratio", 1.0)
              for c in chunks if str(c["chunk_id"]) in checkpoint]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0

    print(f"\n{'='*50}")
    print(f"✅ Simplification complete")
    print(f"   Chunks processed: {len(chunks)}")
    print(f"   Avg word ratio:   {avg_ratio:.2f}x")
    print(f"   Retried chunks:   {retried_count}")
    qa_failures = sum(1 for c in chunks
                      if checkpoint.get(str(c["chunk_id"]), {}).get("qa_issues"))
    print(f"   QA issues found:  {qa_failures} (auto-fixed)")
    if verify:
        flagged = sum(1 for c in chunks
                      if checkpoint.get(str(c["chunk_id"]), {}).get("verification_issues"))
        print(f"   Flagged (info loss): {flagged}")
    print(f"{'='*50}")

    return output_chunks


def _simplify_sequential(
    chunks, llm, book_summary, glossary, temperature,
    min_ratio, max_ratio, max_retries,
    checkpoint, checkpoint_path, paragraph_mode=False,
):
    """Process chunks sequentially with bridge summaries for context."""
    previous_context = ""
    total = len(chunks)
    last_report_pct = 0

    # If resuming, recover context from last completed chunk
    if checkpoint:
        last_done_id = max(int(cid) for cid in checkpoint.keys())
        prev_entry = checkpoint.get(str(last_done_id), {})
        previous_context = prev_entry.get("bridge_summary", prev_entry.get("simplified_text", "")[-400:])

    for chunk in tqdm(chunks, desc="Simplifying (sequential)", unit="chunk"):
        cid = str(chunk["chunk_id"])

        # Skip already-done chunks
        if cid in checkpoint:
            previous_context = checkpoint[cid].get("bridge_summary", checkpoint[cid].get("simplified_text", "")[-400:])
            continue

        if paragraph_mode:
            result = _simplify_one_chunk_paragraphs(
                chunk, llm, book_summary, glossary, temperature,
                previous_context=previous_context,
            )
        else:
            result = _simplify_one_chunk(
                chunk, llm, book_summary, glossary, temperature,
                min_ratio, max_ratio, max_retries,
                previous_context=previous_context,
            )

        # Inline QA: check for issues
        result = _inline_qa_check(result, chunk["text"])

        # Generate bridge summary for next chunk's context
        bridge = _generate_bridge_summary(llm, result["simplified_text"])
        result["bridge_summary"] = bridge
        previous_context = bridge

        checkpoint[cid] = result

        # Checkpoint after every chunk
        _save_checkpoint(checkpoint, checkpoint_path)

        # Periodic QA report every 10%
        done_pct = len(checkpoint) / total * 100 if total > 0 else 100
        if done_pct >= last_report_pct + 10:
            _qa_progress_report(checkpoint, total)
            last_report_pct = int(done_pct // 10) * 10


def _simplify_parallel(
    chunks, llm, book_summary, glossary, temperature,
    min_ratio, max_ratio, max_retries, max_workers,
    checkpoint, checkpoint_path, paragraph_mode=False,
):
    """Process chunks in parallel with ongoing QA."""
    # Filter to only chunks that need processing
    todo = [c for c in chunks if str(c["chunk_id"]) not in checkpoint]

    if not todo:
        print("   All chunks already done!")
        return

    total = len(chunks)
    last_report_pct = (len(checkpoint) / total * 100) if total > 0 else 0
    last_report_pct = int(last_report_pct // 10) * 10

    print(f"   Processing {len(todo)} chunks with {max_workers} workers...")

    # Build a lookup for original text by chunk_id
    original_texts = {str(c["chunk_id"]): c["text"] for c in chunks}

    # Choose the simplification function
    if paragraph_mode:
        simplify_fn = lambda chunk: _simplify_one_chunk_paragraphs(
            chunk, llm, book_summary, glossary, temperature,
            previous_context="",
        )
    else:
        simplify_fn = lambda chunk: _simplify_one_chunk(
            chunk, llm, book_summary, glossary, temperature,
            min_ratio, max_ratio, max_retries,
            previous_context="",
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for chunk in todo:
            future = executor.submit(simplify_fn, chunk)
            futures[future] = chunk

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Simplifying (parallel)", unit="chunk"):
            chunk = futures[future]
            cid = str(chunk["chunk_id"])

            try:
                result = future.result()
                # Inline QA
                result = _inline_qa_check(result, chunk["text"])
                checkpoint[cid] = result
            except Exception as e:
                print(f"\n   ❌ Chunk {cid} failed: {e}")
                checkpoint[cid] = {
                    "simplified_text": chunk["text"],
                    "attempts": 0,
                    "final_ratio": 1.0,
                    "was_retried": False,
                    "error": str(e),
                    "qa_issues": [f"Processing error: {e}"],
                    "qa_passed": False,
                }

            _save_checkpoint(checkpoint, checkpoint_path)

            # Periodic QA report every 10%
            done_pct = len(checkpoint) / total * 100 if total > 0 else 100
            if done_pct >= last_report_pct + 10:
                _qa_progress_report(checkpoint, total)
                last_report_pct = int(done_pct // 10) * 10


def _run_verification(chunks, llm, checkpoint):
    """Run verification pass — ask LLM to compare original vs simplified for info loss."""
    print("\n🔍 Running verification pass...")

    for chunk in tqdm(chunks, desc="Verifying", unit="chunk"):
        cid = str(chunk["chunk_id"])
        entry = checkpoint.get(cid, {})
        simplified = entry.get("simplified_text", "")

        if not simplified or len(chunk["text"].split()) < 5:
            continue

        sys_prompt, usr_prompt = build_verification_prompt(
            original_text=chunk["text"],
            simplified_text=simplified,
        )

        try:
            verdict = llm.generate(prompt=usr_prompt, system_prompt=sys_prompt, temperature=0.1)

            # Check if the model flagged issues
            is_clean = ("no issues" in verdict.lower() or "all preserved" in verdict.lower())

            if not is_clean:
                entry["verification_issues"] = verdict.strip()
                print(f"\n   ⚠️  Chunk {cid}: {verdict[:150]}...")
            else:
                entry["verification_issues"] = None

        except Exception as e:
            entry["verification_issues"] = f"Verification error: {e}"


# ── Footnote Generation ─────────────────────────────────────────────────────

def generate_footnotes(
    output_chunks: list[dict],
    llm: OllamaClient,
    max_workers: int = 2,
) -> dict[str, str]:
    """
    Scan all simplified chunks for terms a high-school graduate wouldn't know.
    Returns a single deduplicated glossary of {term: explanation} to be used as
    Markdown footnotes.

    Args:
        output_chunks: List of chunk dicts with "simplified_text" key.
        llm:           OllamaClient instance.
        max_workers:   Concurrent LLM calls for footnote extraction.

    Returns:
        Dict of {term: concise explanation}, deduplicated across all chunks.
    """
    import re as _re

    # Collect non-trivial chunks
    substantive = [
        c for c in output_chunks if len(c.get("simplified_text", "").split()) > 20
    ]

    if not substantive:
        return {}

    print(f"\n📝 Generating footnotes for {len(substantive)} chunks...")

    all_terms: dict[str, str] = {}

    def _extract_terms_from_chunk(chunk: dict) -> dict:
        """Ask the LLM to identify hard terms in one chunk."""
        text = chunk.get("simplified_text", "")
        sys_prompt, usr_prompt = build_footnote_prompt(text)

        try:
            response = llm.generate(
                prompt=usr_prompt,
                system_prompt=sys_prompt,
                temperature=0.1,
            )

            # Parse JSON — strip markdown fences if present
            response = response.strip()
            if response.startswith("```"):
                response = _re.sub(r"^```(?:json)?\s*", "", response)
                response = _re.sub(r"\s*```$", "", response)

            terms = json.loads(response)
            if isinstance(terms, dict):
                return terms
        except (json.JSONDecodeError, Exception):
            pass  # Silently skip – not critical

        return {}

    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_terms_from_chunk, c): c
            for c in substantive
        }

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Extracting footnotes", unit="chunk"):
            try:
                chunk_terms = future.result()
                # First definition wins (keeps explanations consistent)
                for term, definition in chunk_terms.items():
                    if term not in all_terms:
                        all_terms[term] = definition
            except Exception:
                pass

    print(f"   Found {len(all_terms)} unique terms for footnotes")
    return all_terms


def generate_concept_map(full_text: str, llm) -> str:
    """
    Generate a 'Connecting the Ideas' concept map from the full simplified text.

    Args:
        full_text: The assembled simplified document text.
        llm:       The LLM client instance.

    Returns:
        Markdown string with the concept map section, or empty string on failure.
    """
    from src.prompts import build_concept_map_prompt

    print("\n🗺️  Generating concept map...")

    sys_prompt, usr_prompt = build_concept_map_prompt(full_text)

    try:
        response = llm.generate(
            prompt=usr_prompt,
            system_prompt=sys_prompt,
            temperature=0.3,
        )
        concept_map = response.strip()

        # Clean any preamble the LLM might add
        concept_map = re.sub(
            r'^(Here is|Below is|The following|This concept map).*?\n',
            '', concept_map, flags=re.IGNORECASE
        )

        if len(concept_map) < 50:
            print("     ⚠️  Concept map too short, skipping")
            return ""

        section = f"\n\n---\n\n## Connecting the Ideas\n\n{concept_map.strip()}\n"
        print(f"     ✅ Concept map generated ({len(concept_map.split())} words)")
        return section

    except Exception as e:
        print(f"     ⚠️  Concept map generation failed: {str(e)[:80]}")
        return ""


def _save_checkpoint(checkpoint: dict, path: Path | None):
    """Save checkpoint to disk."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

