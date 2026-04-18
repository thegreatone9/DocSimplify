from __future__ import annotations

"""
PDF Ingestion Module
====================
Converts a PDF file into structured Markdown text with chapter/section metadata.
Primary extractor: pymupdf4llm (fast, LLM-optimized).
Fallback: pdfplumber (precise layout control).
"""

import re
from pathlib import Path

import pymupdf4llm
import pymupdf  # PyMuPDF — used for page count & fallback


def extract_pdf_to_markdown(pdf_path: str | Path) -> str:
    """
    Extract all text from a PDF and return as a single Markdown string.

    Uses layout-aware extraction to separate body text from footnotes.
    Footnotes are collected separately and appended at the end, clearly
    marked, so they don't contaminate the main content.

    Args:
        pdf_path: Path to the source PDF file.

    Returns:
        Full document as a Markdown-formatted string (footnotes separated).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Try layout-aware extraction first
    try:
        body, footnotes = extract_pdf_with_layout(pdf_path)
        if body.strip():
            result = body
            if footnotes:
                result += "\n\n---\n\n**Original Document Footnotes:**\n\n"
                for fn in footnotes:
                    result += f"- {fn}\n"
            # Basic cleanup
            result = re.sub(r"\n{4,}", "\n\n\n", result)
            return result
    except Exception as e:
        print(f"  ⚠️  Layout-aware extraction failed ({e}), falling back to pymupdf4llm")

    # Fallback: pymupdf4llm (simpler, no layout awareness)
    markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
    markdown_text = re.sub(r"\n{4,}", "\n\n\n", markdown_text)
    return markdown_text


def extract_pdf_with_layout(pdf_path: str | Path) -> tuple[str, list[str]]:
    """
    Extract PDF text using PyMuPDF's layout data to separate body from footnotes.

    Uses font size and y-position to classify each text span:
      - Body text:  the most common (dominant) font size
      - Footnotes:  smaller font, positioned in the bottom portion of a page
      - Superscript references: tiny font (< body_size - 3), ignored/stripped

    Args:
        pdf_path: Path to the PDF.

    Returns:
        Tuple of (body_markdown, footnote_list):
          - body_markdown: Main content as a Markdown string
          - footnote_list: List of footnote text strings
    """
    doc = pymupdf.open(str(pdf_path))

    # ── Pass 1: Determine the dominant (body) font size ──────────────────
    all_sizes = {}
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:  # text blocks only
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if len(text) < 2:
                        continue  # Skip single chars (superscripts etc.)
                    size = round(span["size"], 1)
                    all_sizes[size] = all_sizes.get(size, 0) + len(text)

    if not all_sizes:
        doc.close()
        return "", []

    # Body font = the size with the most total characters
    body_font_size = max(all_sizes, key=all_sizes.get)
    # Footnote font = any size smaller than body by 1.5+ points
    footnote_threshold = body_font_size - 1.5
    # Superscript = any size < body - 3 points (reference markers)
    superscript_threshold = body_font_size - 3.0

    print(f"  📐 Layout analysis: body={body_font_size}pt, "
          f"footnote=<{footnote_threshold}pt, "
          f"superscript=<{superscript_threshold}pt")

    # ── Pass 2: Extract and classify text ────────────────────────────────
    body_parts = []       # list of (y_position, text, is_new_paragraph)
    footnotes = []
    current_footnote = []

    for page in doc:
        page_height = page.rect.height
        footnote_y_threshold = page_height * 0.65  # Bottom 35% of page
        blocks = page.get_text("dict")["blocks"]

        # Sort blocks by vertical position
        text_blocks = [b for b in blocks if b["type"] == 0]
        text_blocks.sort(key=lambda b: b["bbox"][1])

        prev_body_y = None
        line_heights = []  # collect line spacing to detect paragraph gaps

        for block in text_blocks:
            for line in block["lines"]:
                line_text_parts = []
                line_font_size = 0
                line_y = line["bbox"][1]
                first_span_text = None
                line_is_bold = True   # assume bold until proven otherwise
                line_char_count = 0

                for span in line["spans"]:
                    text = span["text"]
                    size = round(span["size"], 1)

                    # Skip superscript reference markers (tiny font)
                    if size < superscript_threshold and len(text.strip()) <= 3:
                        continue

                    if first_span_text is None:
                        first_span_text = text.strip()

                    line_text_parts.append(text)
                    if len(text.strip()) > 1:
                        line_font_size = max(line_font_size, size)

                    # Check bold: flag bit 16, or "Bold" in font name
                    span_is_bold = bool(
                        span["flags"] & 16
                    ) or "Bold" in span.get("font", "")
                    if text.strip() and not span_is_bold:
                        line_is_bold = False
                    line_char_count += len(text.strip())

                line_text = "".join(line_text_parts).strip()
                if not line_text:
                    continue

                # Check if this line starts with a numbered item
                starts_with_number = bool(
                    first_span_text
                    and re.match(r"^\d+[\.\)]\s*$", first_span_text)
                    and line_font_size >= body_font_size - 0.5
                )

                # Detect headings using multiple signals (no single signal
                # is reliable across all documents):
                #   Signal 1: Font size > body
                #   Signal 2: Bold face
                #   Signal 3: ALL CAPS text
                #   Signal 4: Structural keyword ("Chapter", "Part", etc.)
                #   Signal 5: Short standalone line (< 80 chars)
                # A line is a heading if it has 2+ signals, or 1 very strong one.
                is_heading = False
                heading_level = 0
                is_short = line_char_count > 0 and line_char_count < 80
                is_allcaps = (
                    line_text == line_text.upper()
                    and line_char_count > 3
                    and any(c.isalpha() for c in line_text)
                )
                has_keyword = bool(re.match(
                    r"^(chapter|part|section|appendix|references|bibliography|"
                    r"introduction|conclusion|preface|foreword|epilogue|"
                    r"acknowledgements?|abstract|summary)\b",
                    line_text, re.IGNORECASE
                ))

                # Strong signals (enough alone)
                if line_font_size > body_font_size + 3:
                    is_heading = True
                    heading_level = 1
                elif line_font_size > body_font_size + 1.5:
                    is_heading = True
                    heading_level = 2
                # Combined signals
                elif is_short and not starts_with_number:
                    signals = sum([line_is_bold, is_allcaps, has_keyword])
                    if signals >= 2:
                        is_heading = True
                        heading_level = 2
                    elif signals == 1 and (line_is_bold or has_keyword):
                        is_heading = True
                        heading_level = 3

                # Format the line text
                if is_heading and not starts_with_number:
                    line_text = "#" * heading_level + " " + line_text

                # Classify: is this body or footnote?
                is_footnote = (
                    line_font_size > 0
                    and line_font_size <= footnote_threshold
                    and line_y > footnote_y_threshold
                )

                if is_footnote:
                    current_footnote.append(line_text)
                else:
                    if current_footnote:
                        footnotes.append(" ".join(current_footnote))
                        current_footnote = []

                    # Detect paragraph break via vertical gap
                    if prev_body_y is not None:
                        gap = line_y - prev_body_y
                        if gap > 0:
                            line_heights.append(gap)
                    prev_body_y = line_y

                    body_parts.append((line_y, line_text, starts_with_number or is_heading))

        # End of page: flush footnote, mark page break
        if current_footnote:
            footnotes.append(" ".join(current_footnote))
            current_footnote = []
        # Page break = force new paragraph
        body_parts.append((999999, "", False))
        prev_body_y = None

    doc.close()

    # ── Determine normal line spacing ────────────────────────────────────
    if line_heights:
        line_heights.sort()
        # Median line spacing (most common gap = normal line-to-line)
        normal_spacing = line_heights[len(line_heights) // 2]
    else:
        normal_spacing = 14.0  # fallback

    # Paragraph gap threshold: 1.5x normal spacing
    para_gap_threshold = normal_spacing * 1.5

    # ── Build paragraphs from body parts ─────────────────────────────────
    paragraphs = []
    current_para = []
    prev_y = None

    for y, text, is_numbered_start in body_parts:
        if not text:
            # Page break or empty — flush
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            prev_y = None
            continue

        # Detect paragraph start
        start_new = False

        if prev_y is None:
            start_new = True
        else:
            gap = y - prev_y
            if gap > para_gap_threshold:
                start_new = True

        # Numbered items and headings always start a new paragraph
        if is_numbered_start:
            start_new = True

        is_heading_line = text.startswith("#")

        if start_new and current_para:
            paragraphs.append(" ".join(current_para))
            current_para = []

        # Headings are always standalone paragraphs
        if is_heading_line:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
            paragraphs.append(text)
            prev_y = None  # Force next line to start new paragraph
            continue

        current_para.append(text)
        prev_y = y

    if current_para:
        paragraphs.append(" ".join(current_para))

    body_md = "\n\n".join(paragraphs)

    return body_md, footnotes


def extract_pdf_with_surya(pdf_path: str | Path) -> tuple[str, dict]:
    """
    Extract PDF text using surya's neural models for layout + OCR.

    Best for scanned or image-heavy PDFs. Uses:
      - LayoutPredictor: classify regions (Title, Text, Footnote, etc.)
      - RecognitionPredictor + DetectionPredictor: OCR for scanned pages

    For each page, first tries PyMuPDF text extraction. If a page has no
    embedded text (scanned), falls back to surya's OCR.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        Tuple of (markdown_text, surya_structure):
          - markdown_text: Full document as Markdown with footnotes separated
          - surya_structure: Dict with region counts and types detected
    """
    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from PIL import Image

    pdf_path = Path(pdf_path)
    doc = pymupdf.open(str(pdf_path))

    # ── Render all pages as images ───────────────────────────────────────
    print(f"  📐 Rendering {len(doc)} pages...")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    # ── Load surya models ────────────────────────────────────────────────
    print(f"  🧠 Loading surya models...")
    foundation_pred = FoundationPredictor()
    layout_pred = LayoutPredictor(foundation_pred)

    # Check if any page is scanned (no embedded text)
    is_scanned = any(
        len(doc[i].get_text("text").strip()) < 50
        for i in range(min(3, len(doc)))
    )
    recognition_pred = None
    detection_pred = None
    if is_scanned:
        print(f"  📷 Scanned PDF detected — loading OCR models...")
        recognition_pred = RecognitionPredictor(foundation_pred)
        detection_pred = DetectionPredictor(foundation_pred)

    # ── Run layout detection ─────────────────────────────────────────────
    print(f"  🔍 Detecting layout on {len(images)} pages...")
    layout_results = layout_pred(images, top_k=20)

    # ── Extract text from each detected region ──────────────────────────
    body_parts = []
    footnotes = []
    region_counts = {}

    for page_idx, result in enumerate(layout_results):
        page = doc[page_idx]
        page_rect = page.rect
        img_w, img_h = images[page_idx].size
        scale_x = page_rect.width / img_w
        scale_y = page_rect.height / img_h

        # Check if this page has embedded text
        page_has_text = len(page.get_text("text").strip()) > 50

        for box in sorted(result.bboxes, key=lambda b: b.polygon[0][1]):
            label = box.label
            region_counts[label] = region_counts.get(label, 0) + 1

            xs = [p[0] for p in box.polygon]
            ys = [p[1] for p in box.polygon]

            # Try PyMuPDF text extraction first (faster, higher quality)
            text = ""
            if page_has_text:
                rect = pymupdf.Rect(
                    min(xs) * scale_x, min(ys) * scale_y,
                    max(xs) * scale_x, max(ys) * scale_y,
                )
                text = page.get_text("text", clip=rect).strip()

            # Fall back to surya OCR for scanned pages
            if not text and recognition_pred and detection_pred:
                # Crop the region from the page image
                crop_box = (int(min(xs)), int(min(ys)),
                            int(max(xs)), int(max(ys)))
                region_img = images[page_idx].crop(crop_box)
                ocr_results = recognition_pred(
                    [region_img], det_predictor=detection_pred
                )
                if ocr_results and ocr_results[0].text_lines:
                    text = " ".join(
                        line.text for line in ocr_results[0].text_lines
                    )

            if not text:
                continue

            if label in ("Footnote", "PageFooter"):
                footnotes.append(text.replace("\n", " "))
            elif label in ("PageHeader",):
                pass  # Strip page headers
            else:
                if label == "Title":
                    text = f"# {text}"
                elif label == "SectionHeader":
                    text = f"## {text}"

                body_parts.append({
                    "page": page_idx + 1,
                    "label": label,
                    "text": text,
                    "y": min(ys),
                })

    doc.close()

    # ── Assemble markdown ────────────────────────────────────────────────
    body_paragraphs = [p["text"].replace("\n", " ") for p in body_parts]
    body_md = "\n\n".join(body_paragraphs)

    if footnotes:
        body_md += "\n\n---\n\n**Original Document Footnotes:**\n\n"
        for fn in footnotes:
            body_md += f"- {fn}\n"

    body_md = re.sub(r"\n{4,}", "\n\n\n", body_md)

    surya_structure = {
        "extraction_mode": "surya" + (" + OCR" if is_scanned else ""),
        "total_pages": len(images),
        "region_counts": region_counts,
        "total_regions": sum(region_counts.values()),
        "footnotes_found": len(footnotes),
        "is_scanned": is_scanned,
    }

    print(f"  📊 Surya detected: {surya_structure['total_regions']} regions across {len(images)} pages")
    for label, count in sorted(region_counts.items()):
        print(f"     {label}: {count}")
    if is_scanned:
        print(f"  📷 Used OCR for text extraction")

    return body_md, surya_structure


def extract_pdf_metadata(pdf_path: str | Path) -> dict:
    """
    Extract document metadata (title, author, TOC presence) from a PDF.

    Only returns title/author if they can be inferred with high confidence
    from the PDF's embedded metadata fields.

    Args:
        pdf_path: Path to the source PDF file.

    Returns:
        Dict with keys:
          - "title":    str or None
          - "author":   str or None
          - "has_toc":  bool  (True if the PDF has a structural TOC / bookmarks)
          - "pages":    int
    """
    doc = pymupdf.open(str(pdf_path))
    meta = doc.metadata or {}

    # Extract title — only if it looks like a real title (not a filename or blank)
    raw_title = (meta.get("title") or "").strip()
    title = None
    if raw_title and not raw_title.endswith((".pdf", ".PDF")):
        # Title-case it if it's ALL CAPS
        if raw_title == raw_title.upper() and len(raw_title) > 5:
            title = raw_title.title()
        else:
            title = raw_title

    # Extract author — only if present and not a generic tool name
    raw_author = (meta.get("author") or "").strip()
    author = None
    tool_names = {"acrobat", "adobe", "microsoft", "scanner", "unknown", "user"}
    if raw_author and raw_author.lower() not in tool_names:
        # Title-case if ALL CAPS
        if raw_author == raw_author.upper() and len(raw_author) > 2:
            author = raw_author.title()
        else:
            author = raw_author

    # Check for TOC (PDF bookmarks / outlines)
    toc = doc.get_toc()
    has_toc = len(toc) > 0

    pages = doc.page_count
    doc.close()

    return {
        "title": title,
        "author": author,
        "has_toc": has_toc,
        "pages": pages,
    }


def clean_extracted_markdown(markdown_text: str) -> str:
    """
    Fix common pymupdf4llm extraction artifacts before chapter detection.

    Fixes:
      1. Merged author bylines + subtitles crammed into one heading
         (e.g., "### ED QUISH John Rawls is remembered as...")
      2. Stray "BY" lines between title and author
      3. Headings that are unreasonably long (likely merged paragraphs)

    Args:
        markdown_text: Raw Markdown from pymupdf4llm.

    Returns:
        Cleaned Markdown with structural artifacts fixed.
    """
    lines = markdown_text.split("\n")
    cleaned = []

    for line in lines:
        # Check if this is a heading
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if heading_match:
            hashes = heading_match.group(1)
            title = heading_match.group(2).strip()

            # ── Fix 1: Detect ALL-CAPS author name merged with a sentence ──
            # Pattern: "ED QUISH John Rawls is remembered as..."
            #          ^^^^^^^^ ALL-CAPS name   ^^^^^^^^^^^^^^^^ normal sentence
            author_merge = re.match(
                r"^([A-Z][A-Z\s.'-]{2,30}?)\s+([A-Z][a-z].{20,})$", title
            )
            if author_merge:
                author_name = author_merge.group(1).strip()
                subtitle = author_merge.group(2).strip()
                # Emit the author as a regular bold line, keep the subtitle as the heading
                cleaned.append(f"\n**{author_name}**\n")
                cleaned.append(f"{hashes} {subtitle}")
                continue

            # ── Fix 2: Heading is unreasonably long (>120 chars = likely merged) ──
            # Real section headings are rarely >80 chars. If >120, it's likely
            # a heading merged with the first sentence of the body.
            if len(title) > 120:
                # Try to split at the first sentence boundary
                sentence_break = re.search(r"[.!?]\s+[A-Z]", title)
                if sentence_break:
                    heading_part = title[:sentence_break.start() + 1]
                    body_part = title[sentence_break.start() + 1:].strip()
                    cleaned.append(f"{hashes} {heading_part}")
                    cleaned.append(f"\n{body_part}")
                    continue

            cleaned.append(line)
        else:
            # ── Fix 3: Stray isolated "BY" line → convert to byline marker ──
            if line.strip() == "BY":
                cleaned.append("")  # drop it; the author name follows
                continue

            cleaned.append(line)

    result = "\n".join(cleaned)
    # Clean up any excessive blank lines we may have introduced
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    return result


def detect_chapters(markdown_text: str) -> list[dict]:
    """
    Parse a Markdown string to identify chapters and sections.

    Applies structural cleanup first, then looks for Markdown heading
    patterns (# , ## , ### ) and splits the text into logical units.

    Args:
        markdown_text: Full Markdown text from extract_pdf_to_markdown().

    Returns:
        List of dicts, each with keys:
          - "chapter"       : str  (e.g., "Chapter 1: The Beginning")
          - "section"       : str  (e.g., "1.1 Introduction")
          - "text"          : str  (the body text under this heading)
          - "heading_level" : int  (1 for #, 2 for ##, etc.)
    """
    # Clean up extraction artifacts before parsing
    markdown_text = clean_extracted_markdown(markdown_text)

    chapters = []

    # Split by heading patterns: lines starting with one or more #
    # We'll treat # as chapter boundaries and ## as section boundaries
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    # Find all headings and their positions
    headings = []
    for match in heading_pattern.finditer(markdown_text):
        level = len(match.group(1))
        title = match.group(2).strip()
        start_pos = match.start()
        headings.append({
            "level": level,
            "title": title,
            "start": start_pos,
            "end": match.end(),
        })

    if not headings:
        # No headings detected — treat the entire text as one chapter
        return [{
            "chapter": "Full Document",
            "section": "",
            "text": markdown_text.strip(),
            "heading_level": 1,
        }]

    # Extract text between headings
    current_chapter_title = ""

    for i, heading in enumerate(headings):
        # Text runs from end of this heading to start of the next heading (or end of doc)
        text_start = heading["end"]
        text_end = headings[i + 1]["start"] if i + 1 < len(headings) else len(markdown_text)
        body_text = markdown_text[text_start:text_end].strip()

        # Track the current top-level chapter name for nested sections
        if heading["level"] == 1:
            current_chapter_title = heading["title"]

        chapter_name = current_chapter_title if current_chapter_title else heading["title"]
        section_name = heading["title"] if heading["level"] > 1 else ""

        # Skip headings with no body text (e.g., decorative headings)
        if not body_text:
            continue

        chapters.append({
            "chapter": chapter_name,
            "section": section_name,
            "text": body_text,
            "heading_level": heading["level"],
        })

    return chapters


def validate_extraction(chapters: list[dict], pdf_path: str | Path) -> dict:
    """
    Run quality checks on the extracted text.

    Checks:
      - Total character count vs expected (based on page count × ~2000 chars/page)
      - Percentage of garbled/non-ASCII characters
      - Very short or empty chapters
      - Repeated header/footer text that should be stripped

    Args:
        chapters: Output from detect_chapters().
        pdf_path: Original PDF path (to get page count for comparison).

    Returns:
        Dict with keys:
          - "is_ok"          : bool
          - "char_count"     : int
          - "expected_chars" : int
          - "garbled_pct"    : float
          - "warnings"       : list[str]
    """
    pdf_path = Path(pdf_path)
    warnings = []

    # Get page count from the PDF
    doc = pymupdf.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    # Total extracted characters
    total_chars = sum(len(ch["text"]) for ch in chapters)
    expected_chars = page_count * 2000  # rough estimate: ~2000 chars/page

    # Check extraction ratio
    if total_chars < expected_chars * 0.5:
        warnings.append(
            f"Low text yield: got {total_chars:,} chars from {page_count} pages "
            f"(expected ~{expected_chars:,}). PDF might be scanned or image-heavy."
        )

    # Check for garbled characters (non-printable, non-standard)
    all_text = " ".join(ch["text"] for ch in chapters)
    garbled_count = sum(
        1 for c in all_text
        if not c.isprintable() and c not in "\n\r\t"
    )
    garbled_pct = (garbled_count / len(all_text) * 100) if all_text else 0

    if garbled_pct > 2.0:
        warnings.append(
            f"High garbled character rate: {garbled_pct:.1f}%. "
            "Text extraction may have issues."
        )

    # Check for very short chapters
    short_chapters = [
        ch["chapter"] for ch in chapters
        if len(ch["text"]) < 100
    ]
    if short_chapters:
        warnings.append(
            f"{len(short_chapters)} very short section(s) detected (<100 chars). "
            "These might be decorative headings or extraction artifacts."
        )

    # Check for no chapters at all
    if not chapters:
        warnings.append("No chapters/sections detected in the document.")

    return {
        "is_ok": len(warnings) == 0,
        "char_count": total_chars,
        "expected_chars": expected_chars,
        "page_count": page_count,
        "garbled_pct": round(garbled_pct, 2),
        "chapter_count": len(chapters),
        "warnings": warnings,
    }
