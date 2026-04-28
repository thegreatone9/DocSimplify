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

    # Save images to the output directory so they sit alongside the .md file
    from config import OUTPUT_DIR
    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Primary extraction: pymupdf4llm with image and table extraction
    markdown_text = pymupdf4llm.to_markdown(
        str(pdf_path),
        write_images=True,
        image_path=str(images_dir),
        image_format="png",
        dpi=150,
    )
    markdown_text = re.sub(r"\n{4,}", "\n\n\n", markdown_text)

    # Convert image paths to relative (images/filename.png) for portability
    markdown_text = re.sub(
        r'!\[([^\]]*)\]\(([^)]*)\)',
        lambda m: f'![{m.group(1)}](images/{Path(m.group(2)).name})',
        markdown_text,
    )

    # Clean up pymupdf4llm's "picture text" markers into the image alt text
    # Pattern: ![](path)\n\n**----- Start of picture text -----**<br>\nCaption<br>\n**----- End of picture text -----**
    markdown_text = re.sub(
        r'(\!\[[^\]]*\]\([^\)]+\))\s*\n\s*\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*\s*(?:<br>)?',
        r'\1',
        markdown_text,
        flags=re.DOTALL,
    )

    # Post-processing: separate footnotes using layout analysis
    try:
        body, footnotes = extract_pdf_with_layout(pdf_path)
        if footnotes:
            # Append original document footnotes (if any) to the markdown
            markdown_text += "\n\n---\n\n**Original Document Footnotes:**\n\n"
            for fn in footnotes:
                markdown_text += f"- {fn}\n"
    except Exception:
        pass  # Layout analysis failed — proceed with full markdown as-is

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

        # ── Get ALL text spans with positions and font sizes (no clipping) ──
        page_spans = []
        if page_has_text:
            text_dict = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:  # text blocks only
                    continue
                for line in block.get("lines", []):
                    line_bbox = line["bbox"]  # (x0, y0, x1, y1) in PDF points
                    spans = line.get("spans", [])
                    line_text = "".join(span["text"] for span in spans)
                    # Weighted average font size for this line
                    total_chars = sum(len(span["text"]) for span in spans)
                    avg_size = (
                        sum(span["size"] * len(span["text"]) for span in spans)
                        / max(total_chars, 1)
                    )
                    if line_text.strip():
                        page_spans.append({
                            "text": line_text,
                            "y0": line_bbox[1],
                            "y1": line_bbox[3],
                            "x0": line_bbox[0],
                            "x1": line_bbox[2],
                            "font_size": avg_size,
                        })

        # Compute median body font size for this page
        all_sizes = [s["font_size"] for s in page_spans if s["font_size"] > 0]
        all_sizes.sort()
        median_font_size = all_sizes[len(all_sizes) // 2] if all_sizes else 12.0
        page_height = page_rect.height

        # ── Detect footnote separator line on this page ──────────────
        # Footnote separators are thin horizontal rules (lines or rects)
        # in the bottom half of the page, spanning > 20% of page width.
        footnote_separator_y = None
        page_width = page_rect.width
        for drawing in page.get_drawings():
            for item in drawing.get("items", []):
                if item[0] == "l":  # line
                    p1, p2 = item[1], item[2]
                    if (abs(p1.y - p2.y) < 3
                        and abs(p1.x - p2.x) > page_width * 0.2
                        and p1.y > page_height * 0.5):
                        footnote_separator_y = p1.y
                elif item[0] == "re":  # thin rectangle (used as rule)
                    rect = item[1]
                    if (rect.height < 3
                        and rect.width > page_width * 0.2
                        and rect.y0 > page_height * 0.5):
                        footnote_separator_y = rect.y0

        # ── Group spans into surya's detected regions ────────────────
        for box in sorted(result.bboxes, key=lambda b: b.polygon[0][1]):
            label = box.label
            region_counts[label] = region_counts.get(label, 0) + 1

            xs = [p[0] for p in box.polygon]
            ys = [p[1] for p in box.polygon]

            # Convert surya image coords → PDF point coords
            region_y0 = min(ys) * scale_y
            region_y1 = max(ys) * scale_y
            region_x0 = min(xs) * scale_x
            region_x1 = max(xs) * scale_x

            text = ""
            region_font_size = median_font_size
            if page_has_text and page_spans:
                # Collect spans whose vertical center falls within this region
                matching_lines = []
                for span in page_spans:
                    span_cy = (span["y0"] + span["y1"]) / 2
                    span_cx = (span["x0"] + span["x1"]) / 2
                    # Check Y overlap (primary) and X overlap (for multi-column)
                    if region_y0 - 5 <= span_cy <= region_y1 + 5:
                        if region_x0 - 20 <= span_cx <= region_x1 + 20:
                            matching_lines.append(span)

                # Sort by Y then X for reading order
                matching_lines.sort(key=lambda s: (s["y0"], s["x0"]))

                # ── Filter out footnote lines trapped inside this region ──
                # If surya drew one box spanning body + footnotes, strip
                # lines below the separator before paragraph splitting.
                if footnote_separator_y and matching_lines:
                    body_lines = []
                    for ml in matching_lines:
                        if ml["y0"] >= footnote_separator_y - 5:
                            # This line is below the separator → footnote
                            fn_text = ml["text"].strip()
                            if fn_text:
                                footnotes.append(fn_text)
                        else:
                            body_lines.append(ml)
                    matching_lines = body_lines

                # ── Coordinate-based paragraph splitting ──────────────
                # Even if surya drew one big box, detect paragraph breaks
                # using (a) vertical gaps and (b) first-line indentation.
                if len(matching_lines) >= 2:
                    # Compute median line gap and body left margin
                    gaps = []
                    left_xs = []
                    for k in range(len(matching_lines)):
                        left_xs.append(matching_lines[k]["x0"])
                        if k > 0:
                            gap = matching_lines[k]["y0"] - matching_lines[k - 1]["y1"]
                            if gap > 0:
                                gaps.append(gap)
                    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0
                    left_xs.sort()
                    body_left = left_xs[len(left_xs) // 4] if left_xs else 0  # 25th percentile

                    # Build text with paragraph breaks
                    parts = [matching_lines[0]["text"].strip()]
                    for k in range(1, len(matching_lines)):
                        cur = matching_lines[k]
                        prev = matching_lines[k - 1]
                        gap = cur["y0"] - prev["y1"]

                        # Signal 1: large vertical gap (> 1.5× median)
                        big_gap = median_gap > 0 and gap > median_gap * 1.5

                        # Signal 2: first-line indentation (> 8pt right of body margin)
                        indented = cur["x0"] > body_left + 8

                        # Require BOTH signals — gap alone or indent alone
                        # causes false splits (e.g. section numbers like "2."
                        # on their own line, or footnote superscripts).
                        if big_gap and indented:
                            parts.append("\n\n")
                        else:
                            parts.append(" ")
                        parts.append(cur["text"].strip())

                    text = "".join(parts)
                else:
                    text = " ".join(s["text"].strip() for s in matching_lines if s["text"].strip())

                # Average font size for this region
                if matching_lines:
                    region_font_size = sum(s["font_size"] for s in matching_lines) / len(matching_lines)

            # Fall back to surya OCR for scanned pages
            if not text and recognition_pred and detection_pred:
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

            # ── Footnote detection ────────────────────────────────────
            # Priority 1: surya labeled it as Footnote/PageFooter
            # Priority 2: text is below a detected separator line + starts with digit
            # Priority 3: fallback heuristic (bottom + small font + digit)
            is_footnote = label in ("Footnote", "PageFooter")
            if not is_footnote and label in ("Text", "TextBlock"):
                starts_with_digit = bool(re.match(r'^\d', text.strip()))
                if footnote_separator_y and region_y0 >= footnote_separator_y - 5 and starts_with_digit:
                    # Below the separator line and starts with a number → footnote
                    is_footnote = True
                elif not footnote_separator_y:
                    # No separator found — fall back to font-size heuristic
                    in_bottom = region_y1 > page_height * 0.70
                    small_font = region_font_size <= median_font_size * 0.85
                    if in_bottom and small_font and starts_with_digit:
                        is_footnote = True

            if is_footnote:
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
    # A single surya region may now contain \n\n paragraph breaks from
    # coordinate-based splitting. Expand those into separate paragraphs.
    body_paragraphs = []
    for p in body_parts:
        raw = p["text"]
        # Split on our inserted paragraph breaks
        sub_paras = re.split(r'\n{2,}', raw)
        for sp in sub_paras:
            cleaned = sp.replace("\n", " ").strip()
            if cleaned:
                body_paragraphs.append(cleaned)

    # ── Merge false paragraph splits ─────────────────────────────────────
    # Surya sometimes draws two bounding boxes for one paragraph.
    # Heuristic: if a paragraph starts lowercase and the previous one
    # doesn't end with terminal punctuation, merge them.
    merged = []
    for para in body_paragraphs:
        if (merged
            and para
            and para[0].islower()
            and not re.search(r'[.!?:]\s*$', merged[-1])):
            merged[-1] = merged[-1].rstrip() + " " + para
        else:
            merged.append(para)
    body_paragraphs = merged

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


def extract_pdf_with_paddle(pdf_path: str | Path) -> tuple[str, dict, list[dict]]:
    """
    Extract PDF text using PaddleOCR's LayoutDetection for layout analysis.

    Uses PaddleOCR's PP-DocLayout model which detects regions far more
    accurately than surya on single-column academic PDFs. Text is still
    extracted from PyMuPDF's span data (not OCR), so this is fast and
    accurate for native PDFs.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        Tuple of (markdown_text, structure_info, labeled_blocks):
          - markdown_text: Full document as Markdown with footnotes separated
          - structure_info: Dict with region counts and types detected
          - labeled_blocks: Ordered list of dicts with keys:
              "label": paddle label (text, title, reference, etc.)
              "text": extracted text for this region
              "page": 1-indexed page number
              "section_type": mapped type (BODY, FRONT_MATTER, REFERENCES, VERBATIM)
    """
    import numpy as np
    from PIL import Image
    from paddleocr import LayoutDetection

    pdf_path = Path(pdf_path)
    doc = pymupdf.open(str(pdf_path))

    # ── Render all pages as images ───────────────────────────────────────
    print(f"  📐 Rendering {len(doc)} pages...")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    # ── Load PaddleOCR layout model ──────────────────────────────────────
    print(f"  🧠 Loading PaddleOCR layout model...")
    layout_engine = LayoutDetection()

    # ── Run layout detection on each page ────────────────────────────────
    print(f"  🔍 Detecting layout on {len(images)} pages...")

    body_parts = []
    footnotes = []
    region_counts = {}

    skipped_pages = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # Skip blank pages
        page_text = page.get_text("text").strip()
        if not page_text:
            skipped_pages.append(page_idx + 1)
            continue

        page_rect = page.rect
        img_w, img_h = images[page_idx].size
        scale_x = page_rect.width / img_w
        scale_y = page_rect.height / img_h

        # Check if this page has embedded text
        page_has_text = len(page.get_text("text").strip()) > 50

        # ── Get ALL text spans with positions and font sizes ─────────
        page_spans = []
        if page_has_text:
            text_dict = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_bbox = line["bbox"]
                    spans = line.get("spans", [])
                    line_text = "".join(span["text"] for span in spans)
                    total_chars = sum(len(span["text"]) for span in spans)
                    avg_size = (
                        sum(span["size"] * len(span["text"]) for span in spans)
                        / max(total_chars, 1)
                    )
                    if line_text.strip():
                        page_spans.append({
                            "text": line_text,
                            "y0": line_bbox[1],
                            "y1": line_bbox[3],
                            "x0": line_bbox[0],
                            "x1": line_bbox[2],
                            "font_size": avg_size,
                        })

        # Compute median body font size for this page
        all_sizes = [s["font_size"] for s in page_spans if s["font_size"] > 0]
        all_sizes.sort()
        median_font_size = all_sizes[len(all_sizes) // 2] if all_sizes else 12.0
        page_height = page_rect.height

        # ── Detect footnote separator line ────────────────────────────
        footnote_separator_y = None
        page_width = page_rect.width
        for drawing in page.get_drawings():
            for item in drawing.get("items", []):
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    if (abs(p1.y - p2.y) < 3
                        and abs(p1.x - p2.x) > page_width * 0.2
                        and p1.y > page_height * 0.5):
                        footnote_separator_y = p1.y
                elif item[0] == "re":
                    rect = item[1]
                    if (rect.height < 3
                        and rect.width > page_width * 0.2
                        and rect.y0 > page_height * 0.5):
                        footnote_separator_y = rect.y0

        # ── Run PaddleOCR layout detection ────────────────────────────
        img_np = np.array(images[page_idx])
        raw_result = layout_engine.predict(img_np)
        det = raw_result[0]
        boxes = det['boxes']

        # ── Process each detected region ─────────────────────────────
        for box in sorted(boxes, key=lambda b: b['coordinate'][1]):
            label = box['label']
            region_counts[label] = region_counts.get(label, 0) + 1

            coord = box['coordinate']
            # Paddle coords are in image pixels
            region_x0 = float(coord[0]) * scale_x
            region_y0 = float(coord[1]) * scale_y
            region_x1 = float(coord[2]) * scale_x
            region_y1 = float(coord[3]) * scale_y

            text = ""
            region_font_size = median_font_size
            if page_has_text and page_spans:
                # Collect spans whose vertical center falls within this region
                matching_lines = []
                for span in page_spans:
                    span_cy = (span["y0"] + span["y1"]) / 2
                    span_cx = (span["x0"] + span["x1"]) / 2
                    if region_y0 - 5 <= span_cy <= region_y1 + 5:
                        if region_x0 - 20 <= span_cx <= region_x1 + 20:
                            matching_lines.append(span)

                # Sort by Y then X for reading order
                matching_lines.sort(key=lambda s: (s["y0"], s["x0"]))

                # ── Filter out footnote lines trapped inside this region ──
                if footnote_separator_y and matching_lines:
                    body_lines = []
                    for ml in matching_lines:
                        if ml["y0"] >= footnote_separator_y - 5:
                            fn_text = ml["text"].strip()
                            if fn_text:
                                footnotes.append(fn_text)
                        else:
                            body_lines.append(ml)
                    matching_lines = body_lines

                # ── Coordinate-based paragraph splitting ──────────────
                if len(matching_lines) >= 2:
                    gaps = []
                    left_xs = []
                    for k in range(len(matching_lines)):
                        left_xs.append(matching_lines[k]["x0"])
                        if k > 0:
                            gap = matching_lines[k]["y0"] - matching_lines[k - 1]["y1"]
                            if gap > 0:
                                gaps.append(gap)
                    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0
                    left_xs.sort()
                    body_left = left_xs[len(left_xs) // 4] if left_xs else 0

                    parts = [matching_lines[0]["text"].strip()]
                    for k in range(1, len(matching_lines)):
                        cur = matching_lines[k]
                        prev = matching_lines[k - 1]
                        gap = cur["y0"] - prev["y1"]

                        big_gap = median_gap > 0 and gap > median_gap * 1.5
                        indented = cur["x0"] > body_left + 8

                        if big_gap and indented:
                            parts.append("\n\n")
                        else:
                            parts.append(" ")
                        parts.append(cur["text"].strip())

                    text = "".join(parts)
                else:
                    text = " ".join(s["text"].strip() for s in matching_lines if s["text"].strip())

                # Average font size for this region
                if matching_lines:
                    region_font_size = sum(s["font_size"] for s in matching_lines) / len(matching_lines)

            if not text:
                continue

            # ── Footnote detection ────────────────────────────────────
            is_footnote = label in ("footnote", "footer")
            if not is_footnote and label in ("text",):
                starts_with_digit = bool(re.match(r'^\d', text.strip()))
                if footnote_separator_y and region_y0 >= footnote_separator_y - 5 and starts_with_digit:
                    is_footnote = True
                elif not footnote_separator_y:
                    in_bottom = region_y1 > page_height * 0.70
                    small_font = region_font_size <= median_font_size * 0.85
                    if in_bottom and small_font and starts_with_digit:
                        is_footnote = True

            if is_footnote:
                footnotes.append(text.replace("\n", " "))
            elif label in ("header",):
                pass  # Strip page headers
            else:
                if label == "title":
                    text = f"# {text}"

                body_parts.append({
                    "page": page_idx + 1,
                    "label": label,
                    "text": text,
                    "y": float(coord[1]),
                })

    doc.close()

    # ── Map paddle labels to section_type ─────────────────────────────────
    LABEL_TO_SECTION = {
        "text": "BODY",
        "title": "FRONT_MATTER",
        "doc_title": "FRONT_MATTER",
        "paragraph_title": "SECTION_HEADER",
        "reference": "REFERENCES",
        "reference_content": "REFERENCES",
        "table": "VERBATIM",
        "table_caption": "VERBATIM",
        "figure": "VERBATIM",
        "figure_caption": "VERBATIM",
        "figure_title": "VERBATIM",
        "image": "VERBATIM",
        "equation": "VERBATIM",
        "formula": "VERBATIM",
        "abstract": "BODY",
        "content": "VERBATIM",         # table of contents entries
        "seal": "VERBATIM",
        "number": "STRIP",         # page numbers — discard
        "header": "STRIP",         # page headers — discard
        "footer": "STRIP",         # page footers — discard
    }

    # ── Build labeled blocks with paragraph splitting applied ─────────────
    labeled_blocks = []
    for p in body_parts:
        raw = p["text"]
        label = p["label"]
        section_type = LABEL_TO_SECTION.get(label, "BODY")

        # Skip page numbers, headers, footers
        if section_type == "STRIP":
            continue

        # Format section headers as markdown headings
        if section_type == "SECTION_HEADER":
            heading_text = raw.replace("\n", " ").strip()
            # Remove leading "# " if the title processing already added it
            heading_text = re.sub(r'^#+\s*', '', heading_text)
            # Detect heading level from numbering: "4.2" = ###, "4" = ##
            if re.match(r'^\d+\.\d+', heading_text):
                heading_text = f"### {heading_text}"
            else:
                heading_text = f"## {heading_text}"
            labeled_blocks.append({
                "label": label,
                "text": heading_text,
                "page": p["page"],
                "section_type": "VERBATIM",  # Pass through as-is, don't simplify
            })
            continue

        # Split text regions into paragraphs (coordinate-based splits
        # already inserted \n\n during extraction)
        sub_paras = re.split(r'\n{2,}', raw)
        cleaned_parts = []
        for sp in sub_paras:
            cleaned = sp.replace("\n", " ").strip()
            if cleaned:
                cleaned_parts.append(cleaned)

        if cleaned_parts:
            labeled_blocks.append({
                "label": label,
                "text": "\n\n".join(cleaned_parts),
                "page": p["page"],
                "section_type": section_type,
            })

    # ── Reclassify text blocks after "References" heading ────────────────
    # Paddle sometimes labels reference entries as "text" on the first page
    # of references. Once we see a heading called "References", everything
    # after it (except other headings) should be REFERENCES.
    in_references = False
    for block in labeled_blocks:
        if block["section_type"] == "VERBATIM" and block["label"] == "paragraph_title":
            heading_text = block["text"].lstrip("#").strip().lower()
            if heading_text in ("references", "bibliography", "works cited"):
                in_references = True
                continue
            elif in_references and heading_text:
                # A new non-reference section heading ends the references zone
                in_references = False
        if in_references and block["section_type"] == "BODY":
            block["section_type"] = "REFERENCES"

    # ── Absorb frontmatter fragments ─────────────────────────────────────
    # Short text blocks (<50 words) before the first substantial paragraph
    # are frontmatter (bylines, citations, blurbs, dates, affiliations).
    # Rule: on pages T and T+1 (T = title page), absorb short text blocks
    # until we hit a text block with ≥50 words.
    # Fallback (no title): absorb leading short text blocks on any page.
    title_page = None
    for block in labeled_blocks:
        if block["label"] in ("doc_title", "title"):
            title_page = block["page"]
            break

    absorbed = 0
    for block in labeled_blocks:
        if block["section_type"] != "BODY":
            continue

        word_count = len(block["text"].split())

        if word_count >= 50:
            # First substantial paragraph — stop absorbing
            break

        # Check page constraint: title page and next page, or any page if no title
        if title_page is not None:
            if block["page"] > title_page + 1:
                break  # Past the frontmatter zone
        
        block["section_type"] = "FRONT_MATTER"
        absorbed += 1

    if absorbed:
        print(f"  📋 Absorbed {absorbed} frontmatter fragment(s) (short blocks before first body paragraph)")

    # ── Merge false paragraph splits within text blocks ──────────────────
    for block in labeled_blocks:
        if block["section_type"] != "BODY":
            continue
        paras = block["text"].split("\n\n")
        merged = []
        for para in paras:
            if (merged
                and para
                and para[0].islower()
                and not re.search(r'[.!?:]\s*$', merged[-1])):
                merged[-1] = merged[-1].rstrip() + " " + para
            else:
                merged.append(para)
        block["text"] = "\n\n".join(merged)

    # ── Assemble markdown (backward compat) ──────────────────────────────
    body_paragraphs = []
    for p in body_parts:
        raw = p["text"]
        sub_paras = re.split(r'\n{2,}', raw)
        for sp in sub_paras:
            cleaned = sp.replace("\n", " ").strip()
            if cleaned:
                body_paragraphs.append(cleaned)

    # ── Merge false paragraph splits ─────────────────────────────────────
    merged = []
    for para in body_paragraphs:
        if (merged
            and para
            and para[0].islower()
            and not re.search(r'[.!?:]\s*$', merged[-1])):
            merged[-1] = merged[-1].rstrip() + " " + para
        else:
            merged.append(para)
    body_paragraphs = merged

    body_md = "\n\n".join(body_paragraphs)

    if footnotes:
        body_md += "\n\n---\n\n**Original Document Footnotes:**\n\n"
        for fn in footnotes:
            body_md += f"- {fn}\n"

    body_md = re.sub(r"\n{4,}", "\n\n\n", body_md)

    structure = {
        "extraction_mode": "paddle",
        "total_pages": len(images),
        "region_counts": region_counts,
        "total_regions": sum(region_counts.values()),
        "footnotes_found": len(footnotes),
        "is_scanned": False,
    }

    print(f"  📊 Paddle detected: {structure['total_regions']} regions across {len(images)} pages")
    if skipped_pages:
        print(f"  🗑️  Skipped {len(skipped_pages)} empty pages: {skipped_pages}")
    for label, count in sorted(region_counts.items()):
        print(f"     {label}: {count}")

    # Summarize labeled blocks
    block_types = {}
    for b in labeled_blocks:
        block_types[b["section_type"]] = block_types.get(b["section_type"], 0) + 1
    print(f"  📦 Labeled blocks: {len(labeled_blocks)} total — {dict(block_types)}")

    return body_md, structure, labeled_blocks


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

    # Detect visible TOC by scanning first pages for heading text
    has_toc = _detect_toc_in_pages(doc)

    # Detect where references/bibliography start (from the back)
    ref_start_page = _detect_references_start_page(doc)

    pages = doc.page_count
    doc.close()

    return {
        "title": title,
        "author": author,
        "has_toc": has_toc,
        "pages": pages,
        "reference_start_page": ref_start_page,  # None if no references found
    }


def _detect_toc_in_pages(doc, max_pages: int = 4) -> bool:
    """
    Scan the first few pages of a PDF for a visible Table of Contents.

    Looks for heading text like 'Table of Contents', 'Contents', etc.
    and dotted leader + page number patterns (e.g., 'Chapter 1 ....... 5').
    """
    toc_headings = [
        r'\btable\s+of\s+contents\b',
        r'\bcontents\b',
        r'\btoc\b',
        r'\blist\s+of\s+(chapters|sections|figures|tables)\b',
        r'\b(chapter|section)\s+index\b',
    ]

    # Dotted leader pattern: text followed by dots and a page number
    dotted_leader = r'\.{3,}\s*\d+'

    pages_to_check = min(max_pages, doc.page_count)
    for page_num in range(pages_to_check):
        page = doc[page_num]
        text = page.get_text("text").lower()

        # Check for TOC heading
        has_heading = any(re.search(p, text) for p in toc_headings)

        # Check for dotted leaders (strong indicator)
        leader_count = len(re.findall(dotted_leader, text))

        if has_heading and leader_count >= 2:
            return True
        if leader_count >= 5:
            # Many dotted leaders = definitely a TOC page even without heading
            return True

    return False


def _detect_references_start_page(doc, max_pages_from_end: int = 5) -> int | None:
    """
    Scan the last few pages of a PDF for a References/Bibliography section.

    Returns the 0-indexed page number where references start, or None
    if no reference section is detected.
    """
    ref_headings = [
        r'\breferences\b',
        r'\bbibliography\b',
        r'\bworks?\s+cited\b',
        r'\bliterature\s+cited\b',
        r'\bcited\s+works?\b',
        r'\bsources\b',
        r'\bendnotes\b',
        r'\bnotes\s+and\s+references\b',
        r'\bfurther\s+reading\b',
    ]

    total_pages = doc.page_count
    start_from = max(0, total_pages - max_pages_from_end)

    for page_num in range(start_from, total_pages):
        page = doc[page_num]
        text = page.get_text("text")

        # Check each line — headings are usually standalone lines
        for line in text.split("\n"):
            stripped = line.strip().lower()
            # Only check short lines (headings, not body text)
            if len(stripped) > 40:
                continue
            if any(re.search(p, stripped) for p in ref_headings):
                return page_num

    return None


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
