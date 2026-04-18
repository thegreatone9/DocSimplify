from __future__ import annotations

"""
EPUB Ingestion Module
=====================
Converts an EPUB file into structured Markdown text with chapter metadata.
Uses ebooklib for file handling + BeautifulSoup for HTML→text conversion.
EPUBs are HTML under the hood, so parsing is far more reliable than PDFs.
"""

import re
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, Tag


def _html_to_markdown(html_content: bytes | str) -> str:
    """
    Convert HTML body content to Markdown-like text, preserving headings.

    This is a lightweight converter that handles:
    - Headings (h1-h6) → Markdown # headings
    - Paragraphs → double newlines
    - Bold/italic → Markdown **bold** / *italic*
    - Lists → Markdown bullet points
    - Everything else → plain text

    Args:
        html_content: Raw HTML string or bytes.

    Returns:
        Markdown-formatted text string.
    """
    if isinstance(html_content, bytes):
        html_content = html_content.decode("utf-8", errors="replace")

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.decompose()

    lines = []

    for element in soup.body.children if soup.body else soup.children:
        if isinstance(element, Tag):
            tag_name = element.name.lower()

            # Headings → Markdown
            if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(tag_name[1])
                heading_text = element.get_text(strip=True)
                if heading_text:
                    lines.append(f"\n{'#' * level} {heading_text}\n")

            # Paragraphs
            elif tag_name == "p":
                text = element.get_text(separator=" ", strip=True)
                if text:
                    lines.append(f"\n{text}\n")

            # Lists
            elif tag_name in ("ul", "ol"):
                for li in element.find_all("li", recursive=False):
                    li_text = li.get_text(separator=" ", strip=True)
                    if li_text:
                        lines.append(f"- {li_text}")

            # Blockquotes
            elif tag_name == "blockquote":
                text = element.get_text(separator=" ", strip=True)
                if text:
                    lines.append(f"> {text}\n")

            # Divs and other block elements — recurse
            elif tag_name in ("div", "section", "article"):
                inner_html = str(element)
                inner_md = _html_to_markdown(inner_html)
                if inner_md.strip():
                    lines.append(inner_md)

            # Catch-all for other elements
            else:
                text = element.get_text(separator=" ", strip=True)
                if text:
                    lines.append(text)

        elif hasattr(element, "strip"):
            # NavigableString (raw text)
            text = element.strip()
            if text:
                lines.append(text)

    result = "\n".join(lines)

    # Clean up excessive whitespace
    result = re.sub(r"\n{4,}", "\n\n\n", result)

    return result.strip()


def extract_epub_to_markdown(epub_path: str | Path) -> str:
    """
    Extract all text from an EPUB and return as a single Markdown string.

    Reads the EPUB spine (reading order), extracts body content from each
    document item, converts HTML to Markdown while preserving heading hierarchy.

    Args:
        epub_path: Path to the source EPUB file.

    Returns:
        Full document as a Markdown-formatted string.
    """
    epub_path = Path(epub_path)
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB not found: {epub_path}")

    book = epub.read_epub(str(epub_path))
    sections = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_body_content()
        if content:
            md_text = _html_to_markdown(content)
            if md_text.strip():
                sections.append(md_text)

    return "\n\n---\n\n".join(sections)


def parse_epub_chapters(epub_path: str | Path) -> list[dict]:
    """
    Parse an EPUB into ordered chapters with metadata.

    Reads the EPUB's spine/TOC to determine chapter order, then extracts
    text for each chapter item, filtering out non-content items like
    title pages, copyright pages, and TOC pages.

    Args:
        epub_path: Path to the source EPUB file.

    Returns:
        List of dicts, each with keys:
          - "chapter"       : str  (chapter title or item name)
          - "section"       : str  (sub-section if detectable)
          - "text"          : str  (the body text)
          - "heading_level" : int
          - "spine_index"   : int  (position in reading order)
    """
    epub_path = Path(epub_path)
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB not found: {epub_path}")

    book = epub.read_epub(str(epub_path))

    # Build a map of item ID → item for spine lookup
    item_map = {item.get_id(): item for item in book.get_items()}

    # Get spine order (reading sequence)
    spine_ids = [item_id for item_id, _ in book.spine]

    # Non-content filename patterns to skip
    skip_patterns = re.compile(
        r"(title|copyright|cover|toc|nav|colophon|dedication|contents)",
        re.IGNORECASE,
    )

    chapters = []
    chapter_counter = 0

    for spine_idx, item_id in enumerate(spine_ids):
        item = item_map.get(item_id)
        if item is None:
            continue

        # Skip non-document items
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        item_name = item.get_name()

        # Skip likely non-content pages
        if skip_patterns.search(item_name):
            continue

        content = item.get_body_content()
        if not content:
            continue

        md_text = _html_to_markdown(content)
        if not md_text.strip():
            continue

        # Try to extract the first heading as the chapter title
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", md_text, re.MULTILINE)
        if heading_match:
            heading_level = len(heading_match.group(1))
            chapter_title = heading_match.group(2).strip()
        else:
            heading_level = 1
            chapter_title = f"Chapter {chapter_counter + 1}"

        chapter_counter += 1

        chapters.append({
            "chapter": chapter_title,
            "section": "",
            "text": md_text,
            "heading_level": heading_level,
            "spine_index": spine_idx,
        })

    # If we got nothing after filtering, try again without filters
    if not chapters:
        for spine_idx, item_id in enumerate(spine_ids):
            item = item_map.get(item_id)
            if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            content = item.get_body_content()
            if not content:
                continue
            md_text = _html_to_markdown(content)
            if not md_text.strip() or len(md_text.strip()) < 50:
                continue

            chapter_counter += 1
            chapters.append({
                "chapter": f"Section {chapter_counter}",
                "section": "",
                "text": md_text,
                "heading_level": 1,
                "spine_index": spine_idx,
            })

    return chapters


def validate_epub_extraction(chapters: list[dict]) -> dict:
    """
    Run quality checks on the extracted EPUB text.

    Checks:
      - Empty or very short chapters
      - Leftover HTML artifacts
      - Encoding issues

    Args:
        chapters: Output from parse_epub_chapters().

    Returns:
        Dict with keys:
          - "is_ok"          : bool
          - "warnings"       : list[str]
          - "total_chars"    : int
          - "chapter_count"  : int
    """
    warnings = []

    if not chapters:
        warnings.append("No chapters extracted from the EPUB.")
        return {
            "is_ok": False,
            "warnings": warnings,
            "total_chars": 0,
            "chapter_count": 0,
        }

    total_chars = sum(len(ch["text"]) for ch in chapters)

    # Check for very short chapters
    short = [ch["chapter"] for ch in chapters if len(ch["text"]) < 100]
    if short:
        warnings.append(
            f"{len(short)} very short section(s) detected (<100 chars): "
            f"{', '.join(short[:5])}"
        )

    # Check for leftover HTML tags
    all_text = " ".join(ch["text"] for ch in chapters)
    html_remnants = re.findall(r"<[^>]+>", all_text)
    if html_remnants:
        warnings.append(
            f"Found {len(html_remnants)} leftover HTML tag(s) in extracted text. "
            "Extraction may be incomplete."
        )

    # Check overall length
    if total_chars < 5000:
        warnings.append(
            f"Very low text yield ({total_chars:,} chars). "
            "The EPUB may be image-heavy or DRM-protected."
        )

    return {
        "is_ok": len(warnings) == 0,
        "warnings": warnings,
        "total_chars": total_chars,
        "chapter_count": len(chapters),
    }
