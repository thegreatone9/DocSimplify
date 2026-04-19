from __future__ import annotations

"""
Document Structure Classifier
==============================
Two-layer document intelligence:

1. **Raw-page scanning** — Scans actual PDF pages from both ends
   to extract title, author, sections, references, etc.
   Three targeted LLM calls on page-level text.

2. **Chunk classification** — Labels chunks as BODY / REFERENCES /
   FOOTNOTES for simplification decisions.

All detection is based on visible content only — no invisible PDF metadata.
"""

import json
import re
from pathlib import Path


# All recognized section types for chunk classification
SECTION_TYPES = [
    "BODY",         # Main content paragraphs (default)
    "REFERENCES",   # Bibliography, works cited
    "FOOTNOTES",    # Endnotes, footnotes section
    "APPENDIX",     # Appendices, supplementary material
    "FRONT_MATTER", # Title page, copyright, dedication, etc.
    "VERBATIM",     # Metadata-like content to preserve as-is (keywords, abstracts, bios)
]


# ═══════════════════════════════════════════════════════════════════════
# Raw-page document structure scanning
# ═══════════════════════════════════════════════════════════════════════

def scan_document_structure(pdf_path: str | Path, llm, extracted_text: str = "") -> dict:
    """
    Scan a PDF's actual pages to extract the full document structure.

    Three LLM calls:
      1. Forward scan (pages 1-5): title, author, TOC, abstract
      2. Backward scan (last 5 pages): references, appendix, footnotes
      3. Section heading extraction from paragraph starters

    Args:
        pdf_path:       Path to the PDF file.
        llm:            The LLM client instance.
        extracted_text: Optional markdown text from the extraction step.
                        If provided, used for better paragraph splitting
                        in section heading detection.

    Returns:
        Dict with document structure metadata.
    """
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    total_pages = doc.page_count

    # ── Call 1: Forward scan (front matter) ───────────────────────────────
    print("     Scan 1/3: Front matter (forward)...")
    front_result = _scan_forward(doc, llm, max_pages=5)

    # If all 5 pages were front matter, keep scanning
    body_start = front_result.get("body_starts_at_page")
    if body_start is not None and body_start >= 5 and total_pages > 5:
        print("     → Continuing forward scan (long front matter)...")
        extra = _scan_forward(doc, llm, max_pages=10, start_page=5)
        if extra.get("body_starts_at_page") is not None:
            front_result["body_starts_at_page"] = extra["body_starts_at_page"]

    # ── Call 2: Backward scan (back matter) ───────────────────────────────
    print("     Scan 2/3: Back matter (backward)...")
    back_result = _scan_backward(doc, llm, max_pages=5)

    # ── Call 3: Section heading extraction ────────────────────────────────
    print("     Scan 3/3: Section headings...")

    # Use extracted markdown text (better paragraph splits) if available,
    # otherwise fall back to raw PDF text
    if extracted_text:
        source_text = extracted_text
    else:
        source_text = ""
        for i in range(total_pages):
            source_text += doc[i].get_text("text") + "\n\n"

    paragraphs = [p.strip() for p in source_text.split("\n\n") if p.strip()]
    # Take first line (up to 100 chars) of each paragraph
    starters = []
    for p in paragraphs:
        first_line = p.split("\n")[0].strip()[:100]
        # Strip markdown heading markers for cleaner input
        first_line = re.sub(r"^#{1,6}\s+", "", first_line)
        if first_line and len(first_line) > 1:
            starters.append(first_line)

    sections = _extract_section_headings(starters, llm)

    doc.close()

    # ── Normalize author names ────────────────────────────────────────────
    raw_authors = front_result.get("authors", [])
    normalized_authors = []
    for name in raw_authors:
        name = str(name).strip()
        # Title-case ALL CAPS names (e.g., "ED QUISH" → "Ed Quish")
        if name == name.upper() and len(name) > 1:
            name = name.title()
        normalized_authors.append(name)

    # Assemble the complete structure
    result = {
        "title": front_result.get("title"),
        "subtitle": front_result.get("subtitle"),
        "authors": normalized_authors,
        "has_toc": front_result.get("has_toc", False),
        "has_abstract": front_result.get("has_abstract", False),
        "body_starts_at_page": front_result.get("body_starts_at_page", 0),
        "has_references": back_result.get("has_references", False),
        "has_endnotes": back_result.get("has_endnotes", False),
        "back_matter_starts_at_page": back_result.get("back_matter_starts_at_page"),
        "sections": sections,
        "total_pages": total_pages,
    }

    return result


def _scan_forward(doc, llm, max_pages: int = 5, start_page: int = 0) -> dict:
    """Scan pages from the front to extract front-matter metadata."""
    end_page = min(start_page + max_pages, doc.page_count)

    page_texts = []
    for i in range(start_page, end_page):
        text = doc[i].get_text("text").strip()
        if text:
            page_texts.append(f"=== PAGE {i + 1} ===\n{text[:800]}")
        else:
            page_texts.append(f"=== PAGE {i + 1} ===\n[BLANK PAGE]")

    pages_content = "\n\n".join(page_texts)

    system_prompt = (
        "You are a document structure analyst. You examine the first pages "
        "of a document to identify front-matter elements. You output JSON only."
    )

    user_prompt = f"""Examine these pages from the beginning of a document.
Your job is to find the document's TITLE, AUTHOR(S), and where the body begins.

IMPORTANT — Documents come in MANY formats:
- A book may have a separate title page (page with just the title and author)
- An academic paper may have the title as the FIRST LINE of page 1, followed immediately by the author and then body text, all on the same page
- A review/essay may start with the title, then "BY AUTHOR NAME", then body text
- A report may have a cover page with title, organization, and date
- The title could be in ALL CAPS, Title Case, or regular case
- There may be NO separate title page at all

For each page, classify it as one of:
- BLANK: Empty or nearly empty page
- COPYRIGHT: Copyright notice, publisher info, ISBN
- DEDICATION: Dedication or epigraph
- TITLE_PAGE: A page that is entirely or primarily the title/author/publication info
- TITLE_AND_BODY: Page that STARTS with the title/author but also contains body text
- TOC: Table of contents
- ABSTRACT: Abstract or summary
- PREFACE: Preface, foreword, or introduction by another author
- BODY: Purely main content

Extract the following from THE VISIBLE TEXT:
- title: The document's main title. This is the short, prominent text that names the work.
  It could be the very first line on page 1, or on a dedicated title page.
  Copy it EXACTLY as printed (but use Title Case if it's ALL CAPS).
- subtitle: Any secondary title line, publication info, or "Review of..." line
- authors: Author name(s). Look for lines like "BY [NAME]" or names printed prominently.
  Each author should be just a NAME (1-4 words), not a sentence or bio.
  If names are in ALL CAPS, convert to Title Case (ED QUISH → Ed Quish).
- has_toc: true if a table of contents is present
- has_abstract: true if an abstract section is present
- body_starts_at_page: Page number where substantive content (arguments/narrative) begins

CRITICAL RULES:
- The title is NEVER a full sentence. It's a short name/label for the document.
- If the first SHORT line of page 1 looks like a title (and isn't a sentence), it IS the title.
- Authors are just names — NOT bios, affiliations, or descriptions.
- If you see "BY [NAME]" or "[NAME]" under the title, that's the author.
- If you truly cannot find a title, use null. Do NOT guess or invent one.

Respond with ONLY this JSON:
{{
  "page_classifications": [{{"page": 1, "type": "TITLE_AND_BODY"}}, ...],
  "title": "exact title" or null,
  "subtitle": "exact subtitle" or null,
  "authors": ["Name One"] or [],
  "has_toc": false,
  "has_abstract": false,
  "body_starts_at_page": 1
}}

PAGES:

{pages_content}"""

    try:
        response = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )
        return _parse_json_object(response)
    except Exception as e:
        print(f"     ⚠️  Forward scan failed: {str(e)[:80]}")
        return {"body_starts_at_page": 0}


def _scan_backward(doc, llm, max_pages: int = 5) -> dict:
    """Scan pages from the back to find references/endnotes."""
    total = doc.page_count
    start = max(0, total - max_pages)

    page_texts = []
    for i in range(start, total):
        text = doc[i].get_text("text").strip()
        if text:
            page_texts.append(f"=== PAGE {i + 1} ===\n{text[:800]}")
        else:
            page_texts.append(f"=== PAGE {i + 1} ===\n[BLANK PAGE]")

    pages_content = "\n\n".join(page_texts)

    system_prompt = (
        "You are a document structure analyst. You examine the last pages "
        "of a document to identify back-matter elements. You output JSON only."
    )

    user_prompt = f"""Examine these pages from the END of a document.
Identify any back-matter sections by looking at the VISIBLE TEXT ONLY.

For each page, determine if it is:
- BODY: Regular main content
- REFERENCES: Bibliography, works cited, reference list
- ENDNOTES: Endnotes or footnotes section
- APPENDIX: Appendix or supplementary material
- INDEX: Subject or name index
- BLANK: Empty page

Respond with ONLY this JSON (no other text):
{{
  "page_classifications": [{{"page": 6, "type": "BODY"}}, ...],
  "has_references": false,
  "has_endnotes": false,
  "has_appendix": false,
  "back_matter_starts_at_page": null
}}

Set back_matter_starts_at_page to the page number where the FIRST non-body
section begins at the end. Set to null if the document ends with body text.

PAGES:

{pages_content}"""

    try:
        response = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )
        return _parse_json_object(response)
    except Exception as e:
        print(f"     ⚠️  Backward scan failed: {str(e)[:80]}")
        return {"has_references": False}


def _extract_section_headings(starters: list[str], llm) -> list[str]:
    """
    From a list of paragraph first-lines, identify which are section headings.

    Returns a list of heading strings in document order.
    """
    if not starters:
        return []

    # Sample evenly to keep prompt small — take every Nth starter
    max_starters = 80
    if len(starters) > max_starters:
        step = len(starters) // max_starters
        sampled = starters[::step]
    else:
        sampled = starters

    starters_text = "\n".join(f"{i}: {s}" for i, s in enumerate(sampled))

    system_prompt = (
        "You identify section headings in a document. You output JSON only. "
        "You NEVER invent or fabricate headings — you only select from the given list."
    )

    user_prompt = f"""Below are the first lines of consecutive paragraphs from a document.
Your job is to identify which of these lines are SECTION HEADINGS.

A section heading is typically:
- Short (1-8 words)
- A title or label, NOT a regular sentence
- May be numbered (1., I., A.) or unnumbered
- Does NOT end with a period (usually)
- Does NOT contain footnote markers like [1] or [^1]
- Is NOT a sentence fragment (must make sense as a standalone label)

CRITICAL ANTI-HALLUCINATION RULES:
- You MUST only return strings that appear EXACTLY in the list below
- Do NOT invent, rephrase, or fabricate any headings
- Do NOT return "Introduction", "Conclusion", "Chapter 1", etc. unless those
  EXACT strings appear in the numbered list below
- If NO lines look like section headings, return an empty array []
- When in doubt, EXCLUDE rather than include

Respond with ONLY a JSON array of heading strings copied EXACTLY from the list:
["Heading One", "Heading Two"]

PARAGRAPH FIRST LINES:

{starters_text}"""

    try:
        response = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )
        text = response.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            headings = json.loads(match.group())
            # Post-validate: only keep headings that actually exist in starters
            # (prevents hallucinated headings from slipping through)
            starters_set = set(sampled)
            validated = []
            for h in headings:
                if not isinstance(h, str):
                    continue
                h = h.strip()
                if h in starters_set:
                    validated.append(h)
                else:
                    # Fuzzy match: check if any starter starts with this heading
                    for s in sampled:
                        if s.startswith(h) or h.startswith(s):
                            validated.append(s)
                            break
            return validated
        return []
    except Exception as e:
        print(f"     ⚠️  Section heading extraction failed: {str(e)[:80]}")
        return []


def _parse_json_object(response: str) -> dict:
    """Parse a JSON object from an LLM response."""
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def print_scan_summary(scan_result: dict) -> None:
    """Print a readable summary of the document structure scan."""
    title = scan_result.get("title", "Unknown")
    authors = scan_result.get("authors", [])
    sections = scan_result.get("sections", [])

    print(f"     Title:      {title}")
    if scan_result.get("subtitle"):
        print(f"     Subtitle:   {scan_result['subtitle']}")
    if authors:
        print(f"     Author(s):  {', '.join(authors)}")
    print(f"     Body start: page {scan_result.get('body_starts_at_page', '?')}")
    print(f"     TOC: {'Yes' if scan_result.get('has_toc') else 'No'}")
    print(f"     Abstract: {'Yes' if scan_result.get('has_abstract') else 'No'}")
    print(f"     References: {'Yes' if scan_result.get('has_references') else 'No'}")
    if scan_result.get("back_matter_starts_at_page"):
        print(f"     Back matter: page {scan_result['back_matter_starts_at_page']}")
    if sections:
        print(f"     Sections ({len(sections)}): {', '.join(sections[:5])}" +
              (f"... +{len(sections)-5} more" if len(sections) > 5 else ""))



def classify_chunks(chunks: list[dict], llm) -> list[str]:
    """
    Classify every chunk in a document by its structural role.

    Three-step process:
      1. Classification pass — LLM labels each chunk from visible content
      2. QA validation pass — second LLM call reviews and flags anomalies
      3. Corrections — apply QA fixes

    Args:
        chunks:  List of chunk dicts with at least a 'text' key.
        llm:     The LLM client instance.

    Returns:
        List of section type strings, same length as chunks.
        Falls back to 'BODY' for any chunk the LLM fails to classify.
    """
    n = len(chunks)
    if n == 0:
        return []

    previews = _build_previews(chunks)
    previews_text = "\n\n".join(previews)

    # ── Pass 1: Classification ────────────────────────────────────────────
    print("     Pass 1: Classifying chunks...")
    labels = _classification_pass(previews_text, n, llm)

    # ── Pass 2: QA validation ─────────────────────────────────────────────
    print("     Pass 2: Validating classification...")
    corrections = _validation_pass(previews, labels, llm)

    if corrections:
        print(f"     QA flagged {len(corrections)} issue(s):")
        for idx, old_label, new_label, reason in corrections:
            print(f"       Chunk {idx}: {old_label} → {new_label} ({reason})")
            labels[idx] = new_label
    else:
        print("     ✅ QA validation passed — no issues found")

    return labels


def _build_previews(chunks: list[dict]) -> list[str]:
    """
    Build text previews for classification.
    Short chunks (<100 words) get full text; long chunks get first 300 chars.
    """
    previews = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"].strip()
        word_count = len(text.split())

        if word_count <= 100:
            # Short chunk — show full text for accurate classification
            preview = text.replace("\n", " | ")
        else:
            preview = text[:300].replace("\n", " ")

        previews.append(f"CHUNK {i} ({word_count} words): {preview}")
    return previews


def _classification_pass(previews_text: str, n: int, llm) -> list[str]:
    """Pass 1: Classify chunks from visible content."""
    system_prompt = (
        "You are a document structure analyst. You classify sections of "
        "a document by their visible structural role. You are precise."
    )

    user_prompt = f"""Below are previews of {n} consecutive chunks from a document.
Classify each chunk based ONLY on its visible text content.

CATEGORIES (only these 6):
- BODY: Main content — arguments, analysis, narrative, discussion (DEFAULT — most chunks are BODY)
- VERBATIM: Non-prose, structured content that must be preserved exactly as-is: keyword lists,
  abstracts, author bios, publication info, epigraphs, tables of data, numbered item lists,
  or any structured/factual content that would be RUINED if rewritten as prose.
  Use your judgment — if the content's value lies in its exact wording or structure, use VERBATIM.
- REFERENCES: Bibliography, works cited, or reference list entries
- FOOTNOTES: Endnotes, footnotes, or numbered annotations
- APPENDIX: Appendix, supplementary tables, or additional material
- FRONT_MATTER: Title page, copyright, dedication, preface (only at the very start)

CRITICAL RULES:
1. Respond with ONLY a JSON array of {n} strings, one per chunk
2. BODY is the default — when in doubt, use BODY
3. VERBATIM is for structured, non-prose content that should NOT be rewritten
4. FRONT_MATTER is only for the first 1-2 chunks — never in the middle
5. REFERENCES/FOOTNOTES are only at the end — never in the middle
6. Any chunk with argumentative prose, analysis, or narrative is BODY

EXAMPLE RESPONSE for 5 chunks:
["FRONT_MATTER", "VERBATIM", "BODY", "BODY", "REFERENCES"]

CHUNK PREVIEWS:

{previews_text}"""

    try:
        response = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )
        return _parse_classification_response(response, n)
    except Exception as e:
        print(f"     ⚠️  Classification pass failed: {str(e)[:80]}")
        return ["BODY"] * n


def _validation_pass(
    previews: list[str],
    labels: list[str],
    llm,
) -> list[tuple[int, str, str, str]]:
    """
    Pass 2: QA validation. A second LLM call reviews the classification
    and flags any that look wrong.

    Returns a list of (chunk_index, old_label, new_label, reason) tuples.
    """
    n = len(labels)

    # Build the labeled summary for review
    labeled_lines = []
    for i, (preview, label) in enumerate(zip(previews, labels)):
        labeled_lines.append(f"CHUNK {i} = {label}\n  {preview}")
    labeled_text = "\n\n".join(labeled_lines)

    system_prompt = (
        "You are a quality reviewer for document structure classification. "
        "You check if chunk labels make sense and flag errors."
    )

    user_prompt = f"""Review the following document structure classification.
Each chunk has been assigned a label. Check if ANY labels are wrong.

VALID LABELS: BODY, VERBATIM, REFERENCES, FOOTNOTES, APPENDIX, FRONT_MATTER

COMMON MISTAKES TO CHECK:
- FRONT_MATTER assigned to a chunk in the middle of the document (should be BODY)
- BODY assigned to a chunk that is clearly a bibliography/reference list (should be REFERENCES)
- BODY assigned to a chunk that is just a keyword list, abstract, or structured non-prose data (should be VERBATIM)
- REFERENCES or FOOTNOTES assigned to a chunk in the middle of the document (suspicious)
- Any chunk with argumentative prose labeled as non-BODY

RESPONSE FORMAT:
If ALL labels are correct, respond with: {{"corrections": []}}
If any labels are wrong, respond with:
{{"corrections": [{{"chunk": 0, "from": "BODY", "to": "VERBATIM", "reason": "Contains only keywords, not prose"}}]}}

CLASSIFICATION TO REVIEW:

{labeled_text}"""

    try:
        response = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )
        return _parse_validation_response(response, n, set(SECTION_TYPES))
    except Exception as e:
        print(f"     ⚠️  Validation pass failed: {str(e)[:80]}")
        return []


def _parse_validation_response(
    response: str, n: int, valid_types: set
) -> list[tuple[int, str, str, str]]:
    """Parse the QA validation response into correction tuples."""
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Find JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return []

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    corrections = []
    for item in data.get("corrections", []):
        try:
            idx = int(item["chunk"])
            old = str(item.get("from", "")).upper()
            new = str(item.get("to", "")).upper()
            reason = str(item.get("reason", ""))

            if 0 <= idx < n and new in valid_types:
                corrections.append((idx, old, new, reason))
        except (KeyError, ValueError, TypeError):
            continue

    return corrections


def _parse_classification_response(response: str, expected_count: int) -> list[str]:
    """
    Parse the LLM's JSON array response into a list of section types.

    Handles common LLM quirks: markdown code blocks, extra text, partial responses.
    Falls back to BODY for any unparseable or invalid labels.
    """
    text = response.strip()

    # Strip markdown code block wrapper if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Find the JSON array in the response
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        print(f"     ⚠️  Could not find JSON array in classification response")
        return ["BODY"] * expected_count

    try:
        labels = json.loads(match.group())
    except json.JSONDecodeError:
        print(f"     ⚠️  Could not parse classification JSON")
        return ["BODY"] * expected_count

    # Validate and normalize
    valid = set(SECTION_TYPES)
    normalized = []
    for label in labels:
        label_clean = str(label).strip().upper()
        if label_clean in valid:
            normalized.append(label_clean)
        else:
            normalized.append("BODY")

    # Pad or truncate to expected count
    while len(normalized) < expected_count:
        normalized.append("BODY")
    normalized = normalized[:expected_count]

    return normalized


def print_classification_summary(chunks: list[dict]) -> None:
    """Print a readable summary of chunk classifications."""
    print("     Final labels:")
    for i, chunk in enumerate(chunks):
        label = chunk.get("section_type", "UNKNOWN")
        preview = chunk["text"][:60].replace("\n", " ")
        marker = "  " if label == "BODY" else "→ "
        print(f"       {marker}Chunk {i:2d} [{label:10s}]: {preview}...")


# ═══════════════════════════════════════════════════════════════════════
# Common classification function
# ═══════════════════════════════════════════════════════════════════════

def classify_segments(
    segments: list[str],
    categories: list[str],
    category_descriptions: dict[str, str],
    llm,
    context: str = "",
) -> list[str]:
    """
    Universal segment classifier. One batched LLM call to label a list
    of text segments into the given categories.

    This is CLASSIFICATION ONLY — it never modifies, generates, or
    rewrites any text. It only returns labels.

    Args:
        segments:              List of text snippets to classify.
        categories:            List of valid category names.
        category_descriptions: Dict mapping category -> human description.
        llm:                   The LLM client instance.
        context:               Optional context about the document.

    Returns:
        List of category strings, same length as segments.
        Falls back to the first category for any failure.
    """
    n = len(segments)
    if n == 0:
        return []

    default = categories[0]

    # Build the numbered segment list
    seg_lines = []
    for i, seg in enumerate(segments):
        # Truncate each segment preview to keep prompt small
        preview = seg[:200].replace("\n", " ").strip()
        seg_lines.append(f"{i}: {preview}")
    segments_text = "\n".join(seg_lines)

    # Build category descriptions
    cat_text = "\n".join(f"- {cat}: {desc}" for cat, desc in category_descriptions.items())

    system_prompt = (
        "You are a precise text classifier. You label text segments into categories. "
        "You ONLY output a JSON array of labels. Nothing else."
    )

    context_line = f"\nDOCUMENT CONTEXT: {context}\n" if context else ""

    user_prompt = f"""Classify each of the {n} segments below into exactly ONE category.
{context_line}
CATEGORIES:
{cat_text}

RULES:
1. Respond with ONLY a JSON array of {n} strings
2. Each string must be one of: {json.dumps(categories)}
3. If unsure, use "{default}"

SEGMENTS:
{segments_text}"""

    try:
        response = llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )
        return _parse_json_labels(response, n, set(categories), default)

    except Exception as e:
        print(f"     ⚠️  classify_segments failed: {str(e)[:80]}")
        return [default] * n


def _parse_json_labels(response: str, expected: int, valid: set, default: str) -> list[str]:
    """Parse a JSON array of labels from an LLM response."""
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return [default] * expected

    try:
        labels = json.loads(match.group())
    except json.JSONDecodeError:
        return [default] * expected

    result = []
    for label in labels:
        clean = str(label).strip().upper()
        result.append(clean if clean in valid else default)

    # Pad or truncate
    while len(result) < expected:
        result.append(default)
    return result[:expected]


# ═══════════════════════════════════════════════════════════════════════
# Specialized classifiers (all use classify_segments under the hood)
# ═══════════════════════════════════════════════════════════════════════

def detect_fragment_boundaries(paragraphs: list[str], llm) -> list[str]:
    """
    For each gap between consecutive paragraphs, decide if they should
    be MERGED (broken sentence) or kept SEPARATE (natural paragraph break).

    Only examines the boundary — last ~80 chars of paragraph N and
    first ~80 chars of paragraph N+1.

    Returns a list of N-1 labels: "MERGE" or "KEEP".
    """
    if len(paragraphs) < 2:
        return []

    # Build boundary previews
    boundaries = []
    for i in range(len(paragraphs) - 1):
        tail = paragraphs[i].strip()[-80:]
        head = paragraphs[i + 1].strip()[:80]
        boundaries.append(f'...{tail} ||| {head}...')

    return classify_segments(
        segments=boundaries,
        categories=["KEEP", "MERGE"],
        category_descriptions={
            "KEEP": "Natural paragraph break. Two separate ideas or sentences.",
            "MERGE": "Broken sentence. The second part continues the first — they should be one paragraph.",
        },
        llm=llm,
        context="These are boundaries between consecutive paragraphs extracted from a PDF. "
                "PDF extraction sometimes breaks sentences across paragraphs at page or column boundaries.",
    )


def detect_paragraph_roles(paragraphs: list[str], llm) -> list[str]:
    """
    Classify each paragraph's structural role within a document chunk.

    Returns one label per paragraph:
    - SECTION_HEADING: A section title or numbered heading
    - BODY: Regular content paragraph
    - FOOTNOTE: An endnote, footnote, or citation annotation

    This replaces regex-based detection of numbered sections, headings,
    and footnotes.
    """
    return classify_segments(
        segments=paragraphs,
        categories=["BODY", "SECTION_HEADING", "FOOTNOTE"],
        category_descriptions={
            "BODY": "Regular content paragraph — arguments, analysis, narrative, or description.",
            "SECTION_HEADING": "A section title, chapter heading, numbered major section (e.g. '1. Introduction', 'Part II'), or subheading.",
            "FOOTNOTE": "An endnote, footnote, citation annotation, or bibliographic note (e.g. '1. See Marx (1867)').",
        },
        llm=llm,
        context="These are paragraphs from an academic or non-fiction document.",
    )


def detect_meta_commentary(simplified_paragraphs: list[str], llm) -> list[bool]:
    """
    Check each simplified paragraph for meta-commentary — text that
    DESCRIBES the original rather than REWRITING it.

    Returns a list of booleans: True = is meta-commentary (should be removed),
    False = is a valid rewrite (keep).

    Examples of meta-commentary:
      - "This passage discusses how Rawls viewed justice."
      - "The author argues that socialism is preferable."
      - "In this section, we examine the role of markets."

    Examples of valid rewrites:
      - "Rawls believed justice required two things."
      - "Socialism works better because markets fail the poor."
    """
    # Only check paragraphs that are long enough to judge
    indices_to_check = []
    segs_to_check = []
    for i, para in enumerate(simplified_paragraphs):
        # Only check first 1-2 sentences (meta-commentary is always at the start)
        first_sentence = para.split(". ")[0] + "." if ". " in para else para
        if len(first_sentence.split()) >= 5:
            indices_to_check.append(i)
            segs_to_check.append(first_sentence[:200])

    if not segs_to_check:
        return [False] * len(simplified_paragraphs)

    labels = classify_segments(
        segments=segs_to_check,
        categories=["REWRITE", "META"],
        category_descriptions={
            "REWRITE": "A direct rewrite — states facts, arguments, or ideas in simpler words. Written AS the content.",
            "META": "Meta-commentary — talks ABOUT the text, the author, or the passage instead of restating the content itself. Uses phrases like 'this text discusses', 'the author argues', 'this section examines'.",
        },
        llm=llm,
        context="These are the first sentences of paragraphs produced by a text simplification system. "
                "The system should REWRITE content in simpler language, NOT describe or summarize what the original says.",
    )

    # Map back to full list
    result = [False] * len(simplified_paragraphs)
    for idx, label in zip(indices_to_check, labels):
        result[idx] = (label == "META")

    return result
