from __future__ import annotations

"""
Document Sanity Checker
========================
Detects documents from the "20% we defer" category — the kinds of inputs
where our pipeline will produce poor results. Fails fast with a clear
explanation rather than wasting compute on a bad output.

Checks for:
  1. Scanned / image-only PDFs (need OCR — we don't do that)
  2. Math-heavy documents (LaTeX, Unicode math symbols)
  3. Image-heavy documents (diagrams/charts dominate over text)
  4. Non-English / mixed-language content
  5. DRM-protected EPUBs (extraction returns nothing)
  6. Extremely short documents (not worth the pipeline overhead)
  7. Table-heavy / data-heavy documents (spreadsheet-like)
  8. Low prose density (infographics, slide decks, forms, certificates)
  9. Encrypted / password-protected PDFs
 10. Extremely large PDFs (>500 pages)
 11. Source code / programming documents
 12. Duplicate-page PDFs (print spooler errors)
 13. Corrupted / malformed PDFs
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass, field

import pymupdf  # PyMuPDF


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class SanityResult:
    """Result of a sanity check run."""
    passed: bool
    warnings: list[str] = field(default_factory=list)   # non-fatal, proceed with caution
    failures: list[str] = field(default_factory=list)    # fatal, do not proceed
    stats: dict = field(default_factory=dict)            # raw numbers for debugging

    def report(self) -> str:
        """Print a human-readable report."""
        lines = []

        if self.passed:
            lines.append("✅ SANITY CHECK PASSED — document is suitable for simplification.\n")
        else:
            lines.append("❌ SANITY CHECK FAILED — this document is NOT suitable for our pipeline.\n")

        if self.failures:
            lines.append("BLOCKING ISSUES (pipeline will produce poor results):")
            for f in self.failures:
                lines.append(f"  🚫 {f}")

        if self.warnings:
            lines.append("\nWARNINGS (may affect quality, but worth trying):")
            for w in self.warnings:
                lines.append(f"  ⚠️  {w}")

        if self.stats:
            lines.append("\nDETAILED STATS:")
            for k, v in self.stats.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)


# ── Thresholds (tunable) ────────────────────────────────────────────────────

# If more than this % of content is math symbols / LaTeX commands, it's math-heavy
MATH_HEAVY_THRESHOLD_PCT = 5

# If more than this % of characters are non-Latin, it's likely non-English
NON_ENGLISH_THRESHOLD_PCT = 15

# Minimum total extracted text (chars) to be worth processing
MIN_VIABLE_CHARS = 3_000

# If more than this % of lines look like table rows, it's table-heavy
TABLE_HEAVY_THRESHOLD_PCT = 25

# Maximum allowed page count
MAX_PAGE_COUNT = 500

# If more than this % of lines look like code, it's a code document
CODE_HEAVY_THRESHOLD_PCT = 30

# If more than this % of pages are duplicates of another page, it's a spooler error
DUPLICATE_PAGE_THRESHOLD_PCT = 40


# ── LaTeX / math patterns ───────────────────────────────────────────────────

LATEX_PATTERNS = re.compile(
    r"\\(?:frac|int|sum|prod|lim|sqrt|alpha|beta|gamma|delta|epsilon|theta|lambda|"
    r"sigma|omega|partial|nabla|infty|forall|exists|mathbb|mathcal|mathbf|begin\{|"
    r"end\{|left[(\[{]|right[)\]}]|cdot|times|leq|geq|neq|approx|equiv|subset|"
    r"supset|cup|cap|in\b|notin|rightarrow|leftarrow|Rightarrow|Leftarrow)"
)

UNICODE_MATH_CHARS = set(
    "∀∃∄∅∈∉∊∋∌∍∎∏∐∑−∓∔∕∖∗∘∙√∛∜∝∞∟∠∡∢∣∤∥∦∧∨∩∪∫∬∭∮∯∰∱∲∳"
    "∴∵∶∷∸∹∺∻∼∽∾∿≀≁≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗≘≙≚≛≜≝≞≟"
    "≠≡≢≣≤≥≦≧≨≩≪≫≬≭≮≯≰≱≲≳≴≵≶≷≸≹≺≻≼≽≾≿⊀⊁⊂⊃⊄⊅⊆⊇⊈⊉"
    "⊊⊋⊌⊍⊎⊏⊐⊑⊒⊓⊔⊕⊖⊗⊘⊙⊚⊛⊜⊝⊞⊟⊠⊡⊢⊣⊤⊥⊦⊧⊨⊩⊪⊫⊬⊭⊮⊯"
    "⊰⊱⊲⊳⊴⊵⊶⊷⊸⊹⊺⊻⊼⊽⊾⊿⋀⋁⋂⋃⋄⋅⋆⋇⋈⋉⋊⋋⋌⋍⋎⋏⋐⋑⋒⋓⋔⋕⋖⋗"
    "⋘⋙⋚⋛⋜⋝⋞⋟⋠⋡⋢⋣⋤⋥⋦⋧⋨⋩⋪⋫⋬⋭⋮⋯⋰⋱"
    "αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"
)

# Inline math delimiters
INLINE_MATH_PATTERN = re.compile(r"\$[^$]+\$")
DISPLAY_MATH_PATTERN = re.compile(r"\$\$[^$]+\$\$", re.DOTALL)

# ── Table detection ──────────────────────────────────────────────────────────

TABLE_ROW_PATTERN = re.compile(
    r"^[\s]*\|.*\|[\s]*$|"        # Markdown tables: | col1 | col2 |
    r"^\s*\S+\t\S+",              # Tab-separated data
    re.MULTILINE
)


# ── Main check functions ────────────────────────────────────────────────────

def check_pdf(pdf_path: str | Path) -> SanityResult:
    """
    Phase 1: Pre-extraction sanity check on a PDF file.

    Only checks structural/file-level properties that don't need text content:
      - Corrupted / unreadable
      - Encrypted / password-protected
      - Too many pages
      - Duplicate pages

    Content-based checks (prose density, math, language, etc.) run separately
    via check_extracted_content() after OCR extraction.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        SanityResult with pass/fail, warnings, failures, and stats.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        return SanityResult(
            passed=False,
            failures=[f"File not found: {pdf_path}"],
        )

    # ── Check 0: Corrupted / unreadable PDF ──
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        return SanityResult(
            passed=False,
            failures=[
                f"CORRUPTED or MALFORMED PDF. Cannot open file: {str(e)[:150]}. "
                "The file may be damaged, truncated, or not a valid PDF."
            ],
        )

    page_count = len(doc)

    if page_count == 0:
        doc.close()
        return SanityResult(
            passed=False,
            failures=["PDF has 0 pages."],
        )

    # ── Check 1: Encrypted / password-protected ──
    if doc.is_encrypted:
        doc.close()
        return SanityResult(
            passed=False,
            failures=[
                "ENCRYPTED / PASSWORD-PROTECTED PDF. The document requires a password "
                "to read its content. Remove the password protection first (e.g., using "
                "qpdf or an online tool) before processing."
            ],
            stats={"pages": page_count},
        )

    # ── Check 2: Too many pages ──
    if page_count > MAX_PAGE_COUNT:
        doc.close()
        return SanityResult(
            passed=False,
            failures=[
                f"DOCUMENT TOO LARGE. {page_count:,} pages exceeds the maximum of "
                f"{MAX_PAGE_COUNT:,}. Processing would take too long and consume "
                "excessive API credits. Split the document into smaller parts first."
            ],
            stats={"pages": page_count},
        )

    # ── Check 3: Duplicate pages ──
    full_text_parts = []
    for page in doc:
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        full_text_parts.append(text)

    doc.close()

    result = SanityResult(passed=True)
    result.stats = {"pages": page_count}

    _check_duplicate_pages(full_text_parts, result)

    return result


def check_extracted_content(
    text: str,
    page_count: int,
    source_label: str = "document",
) -> SanityResult:
    """
    Phase 2: Post-extraction sanity check on the actual text content.

    Runs AFTER OCR/extraction so it works for both embedded-text and scanned PDFs.
    Checks the extracted text for suitability:
      - Minimum viable content length
      - Prose density (rejects forms, certificates, slide decks)
      - Math-heavy content
      - Non-English / mixed-language
      - Table/data-heavy
      - Source code

    Args:
        text:         The full extracted text (from PaddleOCR or PyMuPDF).
        page_count:   Number of pages in the document.
        source_label: Label for error messages (e.g., "PDF" or "EPUB").

    Returns:
        SanityResult with pass/fail, warnings, failures, and stats.
    """
    result = SanityResult(passed=True)
    total_chars = len(text)
    result.stats = {
        "total_chars": total_chars,
        "pages": page_count,
        "avg_chars_per_page": round(total_chars / page_count) if page_count > 0 else 0,
    }

    # ── Check 1: Minimum viable content ──
    if total_chars < MIN_VIABLE_CHARS:
        result.passed = False
        result.failures.append(
            f"TOO LITTLE TEXT. Only {total_chars:,} characters extracted from {page_count} pages. "
            f"Minimum is {MIN_VIABLE_CHARS:,}. The {source_label} may be mostly images, "
            "a certificate, a form, or corrupted."
        )

    # ── Check 2: Low prose density ──
    _check_prose_density(text, page_count, result)

    # ── Check 3: Math-heavy ──
    _check_math_heavy(text, result)

    # ── Check 4: Non-English ──
    _check_non_english(text, result)

    # ── Check 5: Table-heavy ──
    _check_table_heavy(text, result)

    # ── Check 6: Source code ──
    _check_source_code(text, result)

    return result


def check_epub(epub_path: str | Path, chapters: list[dict] = None) -> SanityResult:
    """
    Run all sanity checks on an EPUB file.

    Can be called with pre-extracted chapters (to avoid re-parsing) or
    with just the file path (will do a lightweight check).

    Args:
        epub_path:  Path to the EPUB file.
        chapters:   Optional pre-extracted chapters from parse_epub_chapters().

    Returns:
        SanityResult with pass/fail, warnings, failures, and stats.
    """
    epub_path = Path(epub_path)

    if not epub_path.exists():
        return SanityResult(
            passed=False,
            failures=[f"File not found: {epub_path}"],
        )

    result = SanityResult(passed=True)

    # If chapters were provided, use them; otherwise do a quick content check
    if chapters is not None:
        full_text = "\n".join(ch.get("text", "") for ch in chapters)
        total_chars = len(full_text)
        chapter_count = len(chapters)
    else:
        # Quick extraction attempt — if ebooklib fails, likely DRM
        try:
            from ebooklib import epub
            book = epub.read_epub(str(epub_path))
            items = list(book.get_items())
            chapter_count = len(items)
            total_chars = 0
            text_parts = []

            from ebooklib import ITEM_DOCUMENT
            for item in book.get_items_of_type(ITEM_DOCUMENT):
                content = item.get_body_content()
                if content:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(content, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                    text_parts.append(text)
                    total_chars += len(text)

            full_text = "\n".join(text_parts)

        except Exception as e:
            result.passed = False
            result.failures.append(
                f"FAILED TO READ EPUB: {e}. This may be a DRM-protected file. "
                "Remove DRM first (e.g., with Calibre + DeDRM plugin) before processing."
            )
            return result

    result.stats = {
        "total_chars": total_chars,
        "chapter_count": chapter_count,
    }

    # ── Check 1: DRM / empty content ──
    if total_chars < MIN_VIABLE_CHARS:
        result.passed = False
        result.failures.append(
            f"TOO LITTLE TEXT. Only {total_chars:,} characters extracted. "
            "The EPUB may be DRM-protected, image-based, or corrupted. "
            "Try removing DRM with Calibre + DeDRM plugin."
        )

    if chapter_count == 0:
        result.passed = False
        result.failures.append(
            "NO CHAPTERS detected. The EPUB may be malformed or empty."
        )

    # ── Remaining checks on extracted text ──
    if total_chars > 0:
        _check_math_heavy(full_text, result)
        _check_non_english(full_text, result)
        _check_table_heavy(full_text, result)

    return result


# ── Shared check subroutines ────────────────────────────────────────────────

def _check_math_heavy(text: str, result: SanityResult) -> None:
    """Detect math-heavy content (LaTeX, Unicode math symbols)."""
    if not text:
        return

    # Count LaTeX commands
    latex_matches = LATEX_PATTERNS.findall(text)
    inline_math = INLINE_MATH_PATTERN.findall(text)
    display_math = DISPLAY_MATH_PATTERN.findall(text)

    # Count Unicode math characters
    math_char_count = sum(1 for c in text if c in UNICODE_MATH_CHARS)

    total_math_indicators = len(latex_matches) + len(inline_math) + len(display_math)
    math_char_pct = (math_char_count / len(text) * 100) if text else 0

    result.stats["latex_commands_found"] = len(latex_matches)
    result.stats["inline_math_expressions"] = len(inline_math)
    result.stats["display_math_expressions"] = len(display_math)
    result.stats["unicode_math_char_pct"] = round(math_char_pct, 2)

    # Heuristic: if there are a LOT of math indicators relative to text length
    math_density = (total_math_indicators / (len(text) / 1000)) if text else 0  # per 1000 chars

    if math_density > 5 or math_char_pct > MATH_HEAVY_THRESHOLD_PCT:
        result.passed = False
        result.failures.append(
            f"MATH-HEAVY document detected. Found {len(latex_matches)} LaTeX commands, "
            f"{len(inline_math)} inline math expressions, and {math_char_pct:.1f}% Unicode "
            "math characters. Our pipeline cannot faithfully simplify mathematical notation — "
            "equations would be garbled or dropped. Use a math-aware tool instead."
        )
    elif math_density > 2 or math_char_pct > 2:
        result.warnings.append(
            f"Moderate math content detected ({len(latex_matches)} LaTeX commands, "
            f"{len(inline_math)} inline math). Mathematical expressions may not be "
            "preserved accurately in the simplified output."
        )


def _check_non_english(text: str, result: SanityResult) -> None:
    """Detect non-English or mixed-language content."""
    if not text:
        return

    total_alpha = 0
    non_latin = 0

    for c in text:
        if c.isalpha():
            total_alpha += 1
            # Check if character is outside Basic Latin + Latin Extended ranges
            cp = ord(c)
            if cp > 0x024F:  # Beyond Latin Extended-B
                non_latin += 1

    non_latin_pct = (non_latin / total_alpha * 100) if total_alpha > 0 else 0
    result.stats["non_latin_char_pct"] = round(non_latin_pct, 2)

    if non_latin_pct > NON_ENGLISH_THRESHOLD_PCT:
        result.passed = False
        result.failures.append(
            f"NON-ENGLISH or MIXED-LANGUAGE content detected. {non_latin_pct:.1f}% of "
            "alphabetic characters are non-Latin script. Our pipeline is designed for "
            "English-only text. Mixed-language content will produce inconsistent or "
            "garbled results."
        )
    elif non_latin_pct > 5:
        result.warnings.append(
            f"Some non-Latin characters detected ({non_latin_pct:.1f}%). The document "
            "may contain foreign language passages, transliterations, or special characters. "
            "These sections may not simplify well."
        )


def _check_table_heavy(text: str, result: SanityResult) -> None:
    """Detect documents that are mostly tabular data."""
    if not text:
        return

    lines = text.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]

    if not non_empty_lines:
        return

    table_rows = sum(1 for l in non_empty_lines if TABLE_ROW_PATTERN.match(l))
    table_pct = (table_rows / len(non_empty_lines) * 100)
    result.stats["table_row_pct"] = round(table_pct, 1)

    if table_pct > TABLE_HEAVY_THRESHOLD_PCT:
        result.passed = False
        result.failures.append(
            f"TABLE/DATA-HEAVY document. {table_pct:.0f}% of lines appear to be tabular data. "
            "Our pipeline is designed for prose text. Simplifying tables, spreadsheet data, "
            "or reference material with heavy tabular structure will produce poor results."
        )
    elif table_pct > 15:
        result.warnings.append(
            f"Moderate tabular content detected ({table_pct:.0f}% of lines). "
            "Tables may not be preserved well in the simplified output."
        )


def _check_prose_density(text: str, page_count: int, result: SanityResult) -> None:
    """
    Detect documents with low prose density — infographics, timelines, posters,
    slide decks, or visual documents that have text but no real paragraphs.

    Two signals:
      1. Low words per page (infographics have very sparse text)
      2. Low prose ratio (most text is short labels, not multi-sentence paragraphs)
    """
    if not text or page_count == 0:
        return

    total_words = len(text.split())
    avg_words_per_page = total_words / page_count

    result.stats["avg_words_per_page"] = round(avg_words_per_page)

    # Count prose: paragraphs with 3+ sentences (text blocks separated by \n\n)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    prose_chars = 0
    for para in paragraphs:
        # Count sentences (rough: split on . ! ?)
        sentences = re.split(r'[.!?]+', para)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) >= 2:
            prose_chars += len(para)

    prose_pct = (prose_chars / len(text) * 100) if text else 0
    result.stats["prose_density_pct"] = round(prose_pct, 1)

    # Check 1: Very low words per page (infographic/poster territory)
    if avg_words_per_page < 50:
        result.passed = False
        result.failures.append(
            f"VISUAL / INFOGRAPHIC document detected. Only {avg_words_per_page:.0f} words/page "
            f"on average (threshold: 50). This document is primarily visual content — "
            "timelines, infographics, posters, or slide decks — with too little prose "
            "to simplify meaningfully."
        )

    # Check 2: Low prose density — has text but it's all short labels/captions
    elif prose_pct < 20 and total_words > 100:
        result.passed = False
        result.failures.append(
            f"LOW PROSE DENSITY. Only {prose_pct:.0f}% of text is in actual paragraphs. "
            f"The document appears to be mostly short labels, captions, or structured data "
            "(infographics, slide decks, forms). Our pipeline needs continuous prose to "
            "simplify effectively."
        )
    elif prose_pct < 40:
        result.warnings.append(
            f"Moderate prose density ({prose_pct:.0f}%). The document has significant "
            "non-prose content (labels, captions, structured elements). Some sections "
            "may not simplify well."
        )


def _check_source_code(text: str, result: SanityResult) -> None:
    """
    Detect documents that are primarily source code or programming content.

    Looks for code-specific patterns: braces, semicolons, import statements,
    function/class definitions, common language keywords.
    """
    if not text:
        return

    lines = text.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]

    if len(non_empty_lines) < 20:
        return  # Too few lines to judge

    # Code indicators per line
    code_patterns = re.compile(
        r"^\s*(?:"
        r"import\s+\w|from\s+\w+\s+import|"       # Python imports
        r"#include\s*<|#define\s+|"                 # C/C++ preprocessor
        r"(?:public|private|protected)\s+(?:class|void|static|int|String)|"  # Java/C#
        r"(?:def|class|function|const|let|var)\s+\w+|"  # Python/JS definitions
        r"(?:if|else|for|while|switch|case|return)\s*[\({]|"  # Control flow with braces
        r"\}\s*(?:else|catch|finally)?|"            # Closing braces
        r";\s*$|"                                    # Semicolon-terminated lines
        r"^\s*//|^\s*/\*|^\s*\*"                    # Comments (// or /* or *)
        r")",
        re.MULTILINE
    )

    code_lines = sum(1 for l in non_empty_lines if code_patterns.match(l))
    code_pct = (code_lines / len(non_empty_lines) * 100)
    result.stats["code_line_pct"] = round(code_pct, 1)

    if code_pct > CODE_HEAVY_THRESHOLD_PCT:
        result.passed = False
        result.failures.append(
            f"SOURCE CODE document detected. {code_pct:.0f}% of lines appear to be "
            "programming code (imports, function definitions, braces, semicolons). "
            "Our pipeline is designed for prose text, not code. Source code cannot be "
            "meaningfully 'simplified' — use a code documentation tool instead."
        )
    elif code_pct > 15:
        result.warnings.append(
            f"Some code-like content detected ({code_pct:.0f}% of lines). "
            "Code blocks may not be preserved well in the simplified output."
        )


def _check_duplicate_pages(page_texts: list[str], result: SanityResult) -> None:
    """
    Detect PDFs with many duplicate pages — typically caused by print spooler
    errors or copy-paste accidents. These waste API credits processing the
    same content repeatedly.
    """
    if len(page_texts) < 3:
        return  # Too few pages to have meaningful duplicates

    import hashlib

    # Hash each page's text (stripped and lowered for normalization)
    page_hashes = []
    for text in page_texts:
        normalized = text.strip().lower()
        if len(normalized) < 50:
            # Very short pages (blank/near-blank) — don't count as duplicates
            page_hashes.append(None)
        else:
            page_hashes.append(hashlib.md5(normalized.encode()).hexdigest())

    # Count unique vs total (excluding blank pages)
    valid_hashes = [h for h in page_hashes if h is not None]
    if not valid_hashes:
        return

    unique_hashes = set(valid_hashes)
    duplicate_count = len(valid_hashes) - len(unique_hashes)
    duplicate_pct = (duplicate_count / len(valid_hashes) * 100)

    result.stats["duplicate_page_pct"] = round(duplicate_pct, 1)

    if duplicate_pct > DUPLICATE_PAGE_THRESHOLD_PCT:
        result.passed = False
        result.failures.append(
            f"DUPLICATE PAGE document. {duplicate_pct:.0f}% of pages ({duplicate_count} of "
            f"{len(valid_hashes)}) are exact duplicates of other pages. This is likely a "
            "print spooler error or copy-paste accident. Processing would waste API credits "
            "on repeated content. Remove duplicate pages first."
        )
    elif duplicate_pct > 15:
        result.warnings.append(
            f"Some duplicate pages detected ({duplicate_pct:.0f}%). "
            "The document may have repeated content or boilerplate pages."
        )


# ── Convenience function ────────────────────────────────────────────────────

def run_sanity_check(
    file_path: str | Path,
    chapters: list[dict] = None,
    exit_on_fail: bool = True,
) -> SanityResult:
    """
    Run the full sanity check for a document. Routes to PDF or EPUB check
    based on file extension.

    Args:
        file_path:     Path to the input document.
        chapters:      Optional pre-extracted chapters (for EPUB, avoids re-parsing).
        exit_on_fail:  If True, print the report and sys.exit(1) on failure.

    Returns:
        SanityResult (only if exit_on_fail is False, or check passed).
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        result = check_pdf(file_path)
    elif ext == ".epub":
        result = check_epub(file_path, chapters=chapters)
    else:
        result = SanityResult(
            passed=False,
            failures=[f"Unsupported file type: '{ext}'. Only .pdf and .epub are supported."],
        )

    # Print the report
    print(result.report())

    if not result.passed and exit_on_fail:
        print("\n🛑 Aborting. Fix the issues above or use a different document.")
        # In a notebook context, raising an exception is better than sys.exit()
        raise SystemExit(
            "Document failed sanity check. See report above for details."
        )

    return result
