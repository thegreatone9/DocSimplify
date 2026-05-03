#!/usr/bin/env python3
"""
Book Simplifier — CLI Entry Point
===================================
Usage:
    1. Drop a PDF or EPUB into data/input/
    2. Run:  python3 run.py

The script auto-detects the file in data/input/ — no filename needed.
Output goes to data/output/<book_name>_simplified.md

Options:
    python3 run.py --workers 1       # sequential (better context)
    python3 run.py --no-verify       # skip verification (faster)
    python3 run.py --resume          # resume from last checkpoint
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Check dependencies before anything else ──────────────────────────────────
_REQUIRED = ["pymupdf", "pymupdf4llm", "ebooklib", "bs4", "lxml", "tqdm", "requests"]
_missing = []
for _mod in _REQUIRED:
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_mod)

if _missing:
    print(f"❌ Missing packages: {', '.join(_missing)}")
    print(f"\n   Fix with:\n     python3 -m pip install -r requirements.txt\n")
    sys.exit(1)

from config import (
    INPUT_DIR, INTERMEDIATE_DIR, OUTPUT_DIR,
    BOOK_STRUCTURE_FILE, CHUNKS_FILE, GLOSSARY_FILE,
    BOOK_SUMMARY_FILE, SIMPLIFIED_CHUNKS_FILE, CHECKPOINT_FILE,
    MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE,
    MAX_CHUNK_TOKENS, OVERLAP_TOKENS,
    MIN_ACCEPTABLE_RATIO, MAX_ACCEPTABLE_RATIO,
    MAX_RETRIES, MAX_WORKERS,
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
)

SUPPORTED_EXTENSIONS = {".pdf", ".epub"}


# ── Logging: tee all output to a log file ─────────────────────────────────────
class _TeeWriter:
    """Write to both a file and the original stream (stdout/stderr)."""
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()
    def flush(self):
        self.stream.flush()
        self.log_file.flush()

_log_handle = None

def _setup_logging():
    """Set up tee logging to data/output/run_log.txt."""
    global _log_handle
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUTPUT_DIR / f"run_log_{timestamp}.txt"
    _log_handle = open(log_path, "w", encoding="utf-8")
    sys.stdout = _TeeWriter(sys.__stdout__, _log_handle)
    sys.stderr = _TeeWriter(sys.__stderr__, _log_handle)
    print(f"  📝 Log file: {log_path}")


def _find_input_file() -> Path:
    """
    Auto-detect a single PDF/EPUB in data/input/.
    Exits with a clear message if 0 or 2+ files are found.
    """
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if len(candidates) == 0:
        print(f"❌ No PDF or EPUB found in {INPUT_DIR}/")
        print(f"   Drop your book into that folder and re-run.")
        sys.exit(1)

    if len(candidates) > 1:
        print(f"❌ Multiple files found in {INPUT_DIR}/:")
        for f in sorted(candidates):
            print(f"   • {f.name}")
        print(f"\n   Please keep only ONE file in the input folder and re-run.")
        sys.exit(1)

    return candidates[0]


def _parse_skip_pages(spec: str) -> set[int]:
    """Parse a skip-pages specification like '1-5,8,10' into a set of ints."""
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            pages.update(range(int(lo), int(hi) + 1))
        elif part:
            pages.add(int(part))
    return pages


def main():
    _setup_logging()
    parser = argparse.ArgumentParser(
        description="Simplify a PDF or EPUB book into plain-English Markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
How to use:
  1. Drop a PDF or EPUB into data/input/
  2. Run:  python3 run.py

Options:
  python3 run.py --workers 1       # sequential (better quality)
  python3 run.py --no-verify       # skip verification pass (faster)
  python3 run.py --resume          # resume from last checkpoint
  python3 run.py --model gemma2:9b # use a different model
  python3 run.py --surya            # use neural layout model (for scanned PDFs)
  python3 run.py --gemini           # use Gemini API
  python3 run.py --groq             # use Groq API (Llama 3.3 70B, fast + free)
        """,
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Parallel LLM workers (default: {MAX_WORKERS}, use 1 for sequential)")

    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint (don't re-extract)")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"Ollama model name (default: {MODEL_NAME})")
    parser.add_argument("--surya", action="store_true",
                        help="Use surya neural layout model (for scanned/image-heavy PDFs)")
    parser.add_argument("--heuristic", action="store_true",
                        help="Use heuristic parser (PyMuPDF) instead of PaddleOCR")
    parser.add_argument("--gemini", action="store_true",
                        help="Use Gemini API (requires GEMINI_API_KEY in config)")
    parser.add_argument("--groq", action="store_true",
                        help="Use Groq API with Llama 3.3 70B (fast, free, generous limits)")
    parser.add_argument("--chunk", action="store_true",
                        help="Simplify whole chunks instead of paragraph-by-paragraph (faster but lower quality)")
    parser.add_argument("--title", type=str, default=None,
                        help="Override document title (otherwise auto-detected from PDF)")
    parser.add_argument("--author", type=str, default=None,
                        help="Override author name (otherwise auto-detected from PDF)")
    parser.add_argument("--skip-pages", type=str, default=None,
                        help="Pages to skip as front matter, e.g. '1-5,8,10' (1-indexed)")

    args = parser.parse_args()

    # ── Auto-detect input file ───────────────────────────────────────────────
    input_path = _find_input_file()
    ext = input_path.suffix.lower()

    if args.skip_pages and ext == ".epub":
        print("  ⚠️  --skip-pages is not supported for EPUB inputs (ignored)")

    workers = args.workers
    model = args.model
    use_surya = args.surya
    use_paddle = not args.heuristic
    use_gemini = args.gemini
    use_groq = args.groq

    # Override settings for cloud LLMs
    if use_gemini:
        model = GEMINI_MODEL
        if "2.5" in model:
            workers = 1  # 2.5 models: 5 req/min
    elif use_groq:
        model = GROQ_MODEL
        if "8b" in model.lower():
            workers = 1  # 8B has tight 6K TPM limit

    # Paragraph mode (default) must be sequential (many sub-requests per chunk)
    if not args.chunk:
        workers = 1

    # Check layout engine availability
    if use_paddle:
        try:
            from paddleocr import LayoutDetection
            print("  ✅ PaddleOCR layout model available")
        except ImportError:
            print("  ❌ PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")
            print("     Falling back to heuristic parser.")
            use_paddle = False
    elif use_surya:
        try:
            from surya.layout import FoundationPredictor, LayoutPredictor
            print("  ✅ Surya neural layout model available")
        except ImportError:
            print("  ❌ Surya not installed. Install with: pip install surya-ocr")
            print("     Falling back to heuristic parser.")
            use_surya = False

    # Auto-name output after the input file
    output_name = input_path.stem + "_simplified.md"
    output_path = OUTPUT_DIR / output_name

    extraction_mode = "PaddleOCR (neural)" if use_paddle else ("surya (neural)" if use_surya else "heuristic (font/position)")
    llm_mode = "Gemini API" if use_gemini else ("Groq API" if use_groq else f"Ollama ({model})")
    print(f"""
╔══════════════════════════════════════════════════╗
║           📚 Book Simplifier                     ║
╚══════════════════════════════════════════════════╝
  Input:    {input_path.name}
  LLM:     {llm_mode}
  Parser:   {extraction_mode}
  Workers:  {workers} ({'parallel' if workers > 1 else 'sequential + context'})

  Output:   {output_path.name}
""")

    total_start = time.time()

    # ── Step 1: Sanity Check ─────────────────────────────────────────────────
    _step("1/10", "Sanity Check")
    from src.sanity_check import run_sanity_check
    run_sanity_check(input_path, exit_on_fail=True)

    # ── Clean previous run (unless resuming) ─────────────────────────────────
    if not args.resume:
        import shutil
        for d in [INTERMEDIATE_DIR, OUTPUT_DIR]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        print("  🧹 Cleared previous output and intermediate files")

    # ── Step 2: LLM Setup ─────────────────────────────────────────────────────
    # Moved early since the raw-page scan (Step 3) needs the LLM
    _step("2/10", "LLM Setup")

    if use_gemini:
        from src.gemini_client import GeminiClient
        try:
            llm = GeminiClient(model_name=model, api_key=GEMINI_API_KEY)
        except ValueError as e:
            print(f"  ❌ {e}")
            sys.exit(1)

        if not llm.is_available():
            print(f"  ❌ Gemini API not reachable. Check your API key.")
            sys.exit(1)
        print(f"  ✅ Connected to Gemini API ({model})")
    elif use_groq:
        from src.groq_client import GroqClient
        try:
            llm = GroqClient(model_name=model, api_key=GROQ_API_KEY)
        except ValueError as e:
            print(f"  ❌ {e}")
            sys.exit(1)

        if not llm.is_available():
            print(f"  ❌ Groq API not reachable. Check your API key.")
            sys.exit(1)
        print(f"  ✅ Connected to Groq API ({model})")
    else:
        from src.llm_client import OllamaClient
        llm = OllamaClient(model_name=model, base_url=OLLAMA_BASE_URL)

        if not llm.is_available():
            print(f"  ❌ Ollama not reachable at {OLLAMA_BASE_URL}")
            print(f"     Start it with: ollama serve")
            print(f"     Pull the model: ollama pull {model}")
            sys.exit(1)

    speed = llm.estimate_speed()
    print(f"  {speed['model_name']} @ {speed['tokens_per_second']:.0f} tok/s")

    # ── Step 3: Extract & Chunk ──────────────────────────────────────────────
    if args.resume and CHUNKS_FILE.exists():
        _step("3/10", "Loading cached extraction")
        with open(BOOK_STRUCTURE_FILE) as f:
            chapters = json.load(f)
        with open(CHUNKS_FILE) as f:
            chunks = json.load(f)
        print(f"  Loaded {len(chapters)} sections, {len(chunks)} chunks")
    else:
        _step("3/10", "Extraction & Chunking")

        if ext == ".pdf" and use_paddle:
            from src.pdf_parser import extract_pdf_with_paddle, detect_chapters, validate_extraction
            print("  🧠 Running PaddleOCR layout detection...")
            markdown, paddle_structure, labeled_blocks = extract_pdf_with_paddle(input_path)

            # ── Post-extraction content sanity check ──
            # Now that we have actual OCR'd text, verify it's prose-worthy
            from src.sanity_check import check_extracted_content
            extracted_text = "\n".join(b["text"] for b in labeled_blocks if b.get("text"))
            page_count = paddle_structure.get("pages", len(set(b["page"] for b in labeled_blocks)))
            content_check = check_extracted_content(extracted_text, page_count, source_label="PDF")
            print(content_check.report())
            if not content_check.passed:
                print("\n🛑 Aborting. The extracted content is not suitable for simplification.")
                raise SystemExit("Content sanity check failed. See report above.")

            # ── Page-level front-matter detection ────────────────────────────
            # Stage 0: user override  →  Stage 1: group by page  →
            # Stage 2: heuristic pre-filter  →  Stage 3: LLM classification  →
            # Stage 4: lookahead boundary  →  Stage 5: apply to blocks
            print("  🔎 Detecting front-matter pages...")
            doc_boundary_author = None
            doc_boundary_title = None
            try:
                from collections import defaultdict
                from src.prompts import build_page_classification_prompt, build_metadata_extraction_prompt
                import re as _re

                # ── Stage 0: User override (--skip-pages) ──
                user_skip = set()
                if args.skip_pages:
                    user_skip = _parse_skip_pages(args.skip_pages)
                    print(f"     User skip pages: {sorted(user_skip)}")

                # ── Stage 1: Group blocks by page ──
                page_blocks = defaultdict(list)
                for i, block in enumerate(labeled_blocks):
                    page_blocks[block["page"]].append(i)

                page_texts = {}
                for pg, indices in page_blocks.items():
                    page_texts[pg] = "\n".join(
                        labeled_blocks[i]["text"] for i in indices
                    )

                sorted_pages = sorted(page_blocks.keys())
                total_pages = len(sorted_pages)

                # page_verdicts: "frontmatter", "body", or None (unresolved)
                page_verdicts = {}

                # Mark user-skipped pages immediately
                for pg in user_skip:
                    if pg in page_blocks:
                        page_verdicts[pg] = "frontmatter"

                # ── Stage 2: Heuristic pre-filter ──
                # First 20 pages: full heuristic checks
                # ALL pages: low-content check (catches mid-document chapter dividers)
                _FM_PATTERNS = [
                    r"(?i)\ball\s+rights\s+reserved\b",
                    r"(?i)\bpublished\s+by\b",
                    r"(?i)\balso\s+available\b",
                    r"(?i)\bisbn\b",
                ]
                _FM_PATTERNS_COMPILED = [_re.compile(p) for p in _FM_PATTERNS]

                for pg in sorted_pages:
                    if pg in page_verdicts:
                        continue  # already resolved

                    text = page_texts[pg]
                    word_count = len(text.split())

                    # Low-content check — applies to ALL pages
                    if word_count < 50:
                        page_verdicts[pg] = "frontmatter"
                        continue

                    # Full heuristic checks — first 20 pages only
                    if pg <= 20:
                        # Copyright symbol
                        if "©" in text:
                            page_verdicts[pg] = "frontmatter"
                            continue

                        # Pattern-based checks
                        if any(pat.search(text) for pat in _FM_PATTERNS_COMPILED):
                            page_verdicts[pg] = "frontmatter"
                            continue

                        # TOC detection: PaddleOCR's 'content' label specifically
                        # means table-of-contents entries — deterministic signal
                        has_toc = any(
                            labeled_blocks[idx]["label"] == "content"
                            for idx in page_blocks[pg]
                        )
                        if has_toc:
                            page_verdicts[pg] = "frontmatter"
                            continue

                heuristic_fm = sum(1 for v in page_verdicts.values() if v == "frontmatter")
                print(f"     Heuristic: {heuristic_fm} page(s) flagged as front matter")

                # ── Stage 3: LLM classification on ambiguous pages ──
                # Only classify unresolved pages within the first 20
                ambiguous = [pg for pg in sorted_pages if pg <= 20 and pg not in page_verdicts]
                llm_calls = 0

                for pg in ambiguous:
                    text = page_texts[pg]
                    # Send first ~400 words (roughly ~2000 chars)
                    page_preview = " ".join(text.split()[:400])

                    sys_p, usr_p = build_page_classification_prompt(page_preview)
                    response = llm.generate(prompt=usr_p, system_prompt=sys_p, temperature=0.0)
                    llm_calls += 1

                    verdict_match = _re.search(r'"verdict"\s*:\s*"(body|frontmatter)"', response)
                    if verdict_match:
                        page_verdicts[pg] = verdict_match.group(1)
                    else:
                        # If LLM response is unparseable, default to body (safe)
                        page_verdicts[pg] = "body"

                    reason_match = _re.search(r'"reason"\s*:\s*"([^"]*)"', response)
                    reason = reason_match.group(1) if reason_match else ""
                    print(f"     Page {pg}: {page_verdicts[pg]}"
                          f"{' — ' + reason if reason else ''}")

                if llm_calls:
                    print(f"     LLM classified {llm_calls} ambiguous page(s)")

                # ── Stage 4: Lookahead boundary confirmation ──
                # Walk pages: when we see "body", confirm with lookahead.
                # If lookahead fails → preface (mark VERBATIM, keep scanning).
                lookahead_depth = 1 if total_pages < 10 else 2
                confirmed_body_start = None
                preface_pages = set()

                for i, pg in enumerate(sorted_pages):
                    verdict = page_verdicts.get(pg)
                    if verdict != "body":
                        continue

                    # Check next N pages
                    lookahead_ok = True
                    for offset in range(1, lookahead_depth + 1):
                        if i + offset < len(sorted_pages):
                            next_pg = sorted_pages[i + offset]
                            next_v = page_verdicts.get(next_pg, "body")
                            if next_v == "frontmatter":
                                lookahead_ok = False
                                break
                        # If we're at the end of pages, count as ok
                        # (document might just be short)

                    if lookahead_ok:
                        confirmed_body_start = pg
                        break
                    else:
                        # False positive — likely a preface
                        preface_pages.add(pg)

                if confirmed_body_start is not None:
                    print(f"  ✅ Body confirmed at page {confirmed_body_start} "
                          f"(lookahead={lookahead_depth})")
                else:
                    # No confirmed body start — treat everything as body
                    confirmed_body_start = sorted_pages[0] if sorted_pages else 1
                    print(f"  ⚠️  No confirmed body start, treating all pages as body")

                if preface_pages:
                    print(f"     Preface pages (kept as verbatim): {sorted(preface_pages)}")

                # ── Stage 5: Apply verdicts to blocks ──
                discarded = 0
                verbatim_kept = 0
                for pg in sorted_pages:
                    if pg >= confirmed_body_start:
                        break  # everything from here onward keeps default section_type
                    for idx in page_blocks[pg]:
                        if pg in preface_pages:
                            labeled_blocks[idx]["section_type"] = "VERBATIM"
                            verbatim_kept += 1
                        else:
                            labeled_blocks[idx]["section_type"] = "FRONT_MATTER"
                            discarded += 1

                # Also mark low-content pages AFTER the body start (mid-doc dividers)
                mid_doc_flagged = 0
                for pg in sorted_pages:
                    if pg <= confirmed_body_start:
                        continue
                    if page_verdicts.get(pg) == "frontmatter":
                        for idx in page_blocks[pg]:
                            labeled_blocks[idx]["section_type"] = "FRONT_MATTER"
                            mid_doc_flagged += 1

                print(f"     Discarded {discarded} block(s), "
                      f"kept {verbatim_kept} preface block(s) as verbatim")
                if mid_doc_flagged:
                    print(f"     Flagged {mid_doc_flagged} mid-document block(s) "
                          f"(low-content pages)")

                # ── Title/author extraction ──
                # Combine text from first 3 pages for metadata extraction
                meta_pages_text = "\n\n---\n\n".join(
                    page_texts[pg] for pg in sorted_pages[:3] if pg in page_texts
                )
                if meta_pages_text:
                    meta_sys, meta_usr = build_metadata_extraction_prompt(
                        meta_pages_text[:3000]  # cap at ~750 words
                    )
                    meta_response = llm.generate(
                        prompt=meta_usr, system_prompt=meta_sys, temperature=0.0,
                    )
                    author_match = _re.search(r'"author"\s*:\s*"([^"]*)"', meta_response)
                    if author_match and author_match.group(1).strip():
                        doc_boundary_author = author_match.group(1).strip()
                        print(f"  📝 LLM detected author: {doc_boundary_author}")

                    title_match = _re.search(r'"title"\s*:\s*"([^"]*)"', meta_response)
                    if title_match and title_match.group(1).strip():
                        doc_boundary_title = title_match.group(1).strip()
                        print(f"  📝 LLM detected title: {doc_boundary_title}")

            except Exception as e:
                print(f"  ⚠️  Front-matter detection failed ({str(e)[:80]}), "
                      f"using all blocks as-is")

            # ── Hybrid block-level metadata post-filter ──────────────────────
            # Layer 1: Comprehensive regex for known metadata identifiers
            #          (runs on ALL blocks — zero false-positive risk)
            # Layer 2: LLM classification for short blocks on early pages
            #          (catches novel metadata the regex doesn't know about)
            import re as _re_blk
            from src.prompts import build_block_metadata_prompt

            _METADATA_PATTERNS = [
                _re_blk.compile(r"(?i)\b[pel]-?ISSN\b"),
                _re_blk.compile(r"(?i)\bDOI\s*:\s*10\."),
                _re_blk.compile(r"©"),
                _re_blk.compile(r"(?i)\bJEL\s+Classification\b"),
                _re_blk.compile(r"(?i)\bKey\s*words?\s*:"),
                _re_blk.compile(r"(?i)\bReceived\s*:.*Accepted\b"),
                _re_blk.compile(r"(?i)\bCorresponding\s+author\b"),
            ]

            # Layer 1: Regex pass (all blocks)
            regex_filtered = 0
            for block in labeled_blocks:
                if block["section_type"] == "FRONT_MATTER":
                    continue
                if any(pat.search(block["text"]) for pat in _METADATA_PATTERNS):
                    block["section_type"] = "FRONT_MATTER"
                    regex_filtered += 1
            if regex_filtered:
                print(f"  🔍 Post-filter regex: {regex_filtered} metadata block(s) removed")

            # Layer 2: LLM pass on short blocks in early pages
            # Early pages = first 10% or first 3 pages, whichever is larger
            all_pages = sorted(set(b["page"] for b in labeled_blocks))
            early_page_cutoff = max(3, int(len(all_pages) * 0.10))
            early_pages = set(all_pages[:early_page_cutoff])

            llm_candidates = []
            for i, block in enumerate(labeled_blocks):
                if block["section_type"] == "FRONT_MATTER":
                    continue
                if block["page"] not in early_pages:
                    continue
                word_count = len(block["text"].split())
                if word_count < 50:
                    llm_candidates.append(i)

            llm_filtered = 0
            if llm_candidates:
                print(f"  🔍 Post-filter LLM: checking {len(llm_candidates)} "
                      f"short block(s) on pages {sorted(early_pages)}")
                for idx in llm_candidates:
                    block = labeled_blocks[idx]
                    sys_p, usr_p = build_block_metadata_prompt(block["text"])
                    response = llm.generate(
                        prompt=usr_p, system_prompt=sys_p, temperature=0.0,
                    )
                    verdict_match = _re_blk.search(
                        r'"verdict"\s*:\s*"(metadata|content)"', response
                    )
                    if verdict_match and verdict_match.group(1) == "metadata":
                        block["section_type"] = "FRONT_MATTER"
                        llm_filtered += 1
                        print(f"       Block {idx} (page {block['page']}): "
                              f"metadata → removed")

            total_filtered = regex_filtered + llm_filtered
            if total_filtered:
                print(f"  ✅ Post-filter total: {total_filtered} metadata block(s) "
                      f"removed ({regex_filtered} regex, {llm_filtered} LLM)")

            # Paddle path: use labeled blocks directly — no LLM classification needed
            from src.chunker import create_chunks_from_blocks, get_chunk_stats
            chunks = create_chunks_from_blocks(labeled_blocks, max_tokens=MAX_CHUNK_TOKENS)
            stats = get_chunk_stats(chunks)

            # Chapter structure not used in paddle path but needed for assembly
            chapters = [{"chapter": "Full Document", "text": markdown}]

            # Validation
            report = validate_extraction(chapters, input_path)

            if report.get("warnings"):
                for w in report["warnings"]:
                    print(f"  ⚠️  {w}")

            # Print chunk summary
            body_chunks = [c for c in chunks if c.get("section_type") == "BODY"]
            other_chunks = [c for c in chunks if c.get("section_type") != "BODY"]
            print(f"  📦 {len(chunks)} chunks: {len(body_chunks)} BODY (to simplify), {len(other_chunks)} pass-through")
            print(f"  {stats['total_tokens']:,} tokens, est. ~{stats['estimated_minutes']:.0f} min")

            # Paragraph audit: after chunking
            _input_para_count = sum(
                len([p.strip() for p in c['text'].split('\n\n') if p.strip()])
                for c in chunks if c.get('section_type') == 'BODY'
            )
            print(f"  📊 Body paragraph count after chunking: {_input_para_count}")

            # No doc_structure needed for paddle
            doc_structure = {"segment_type": "paddle", "total_major_segments": 0,
                             "major_numbered_points": [], "minor_references": []}

            # Save intermediate files
            INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(BOOK_STRUCTURE_FILE, "w") as f:
                json.dump(chapters, f, ensure_ascii=False, indent=2)
            with open(CHUNKS_FILE, "w") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            with open(INTERMEDIATE_DIR / "doc_structure.json", "w") as f:
                json.dump(doc_structure, f, ensure_ascii=False, indent=2)
            with open(INTERMEDIATE_DIR / "labeled_blocks.json", "w") as f:
                json.dump(labeled_blocks, f, ensure_ascii=False, indent=2)

        elif ext == ".pdf" and use_surya:
            from src.pdf_parser import extract_pdf_with_surya, detect_chapters, validate_extraction
            print("  🧠 Running surya neural layout detection (this may take a few minutes)...")
            markdown, surya_structure = extract_pdf_with_surya(input_path)
            chapters = detect_chapters(markdown)
            report = validate_extraction(chapters, input_path)
        elif ext == ".pdf":
            from src.pdf_parser import extract_pdf_to_markdown, detect_chapters, validate_extraction
            markdown = extract_pdf_to_markdown(input_path)
            chapters = detect_chapters(markdown)
            report = validate_extraction(chapters, input_path)
        else:
            from src.epub_parser import parse_epub_chapters, validate_epub_extraction
            chapters = parse_epub_chapters(input_path)
            report = validate_epub_extraction(chapters)

        # ── Shared post-extraction for non-paddle paths ─────────────────
        if not use_paddle:
            if report.get("warnings"):
                for w in report["warnings"]:
                    print(f"  ⚠️  {w}")

            # Detect document structure before chunking
            from src.chunker import create_chunks, get_chunk_stats, detect_document_structure
            doc_structure = detect_document_structure(chapters)

            print(f"  Structure: {doc_structure['segment_type']} ({doc_structure['total_major_segments']} major segments)")
            if doc_structure["major_numbered_points"]:
                nums = [p["number"] for p in doc_structure["major_numbered_points"]]
                print(f"  Major numbered points: {', '.join(nums)}")
            if doc_structure["minor_references"]:
                refs = [r["number"] for r in doc_structure["minor_references"]]
                print(f"  Minor references: {', '.join(refs)} (will not be treated as major segments)")

            chunks = create_chunks(chapters, max_tokens=MAX_CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS)
            stats = get_chunk_stats(chunks)

            print(f"  {len(chapters)} sections → {stats['total_chunks']} chunks ")
            print(f"  {stats['total_tokens']:,} tokens, est. ~{stats['estimated_minutes']:.0f} min")

            # Paragraph audit: after chunking
            _input_para_count = sum(
                len([p.strip() for p in c['text'].split('\n\n') if p.strip()])
                for c in chunks
            )
            print(f"  📊 Paragraph count after chunking: {_input_para_count}")

            # Save intermediate files
            INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(BOOK_STRUCTURE_FILE, "w") as f:
                json.dump(chapters, f, ensure_ascii=False, indent=2)
            with open(CHUNKS_FILE, "w") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            with open(INTERMEDIATE_DIR / "doc_structure.json", "w") as f:
                json.dump(doc_structure, f, ensure_ascii=False, indent=2)

    # ── Step 4: Raw-Page Document Structure Scan ─────────────────────────────
    scan_path = INTERMEDIATE_DIR / "doc_scan.json"
    if args.resume and scan_path.exists():
        _step("4/10", "Loading cached document scan")
        with open(scan_path) as f:
            doc_scan = json.load(f)
    else:
        _step("4/10", "Document Structure Scan")
        if ext == ".pdf" and use_paddle:
            # Paddle path: derive structure from paddle labels + boundary detection
            # instead of re-opening the PDF and making 3-12 LLM calls
            try:
                _title = doc_boundary_title
            except NameError:
                _title = None
            try:
                _author = doc_boundary_author
            except NameError:
                _author = None

            doc_scan = {
                "title": _title,
                "subtitle": None,
                "authors": [_author] if _author else [],
                "has_toc": any(b.get("label") == "content" for b in labeled_blocks),
                "has_abstract": False,
                "has_references": any(b.get("label") == "reference" for b in labeled_blocks),
                "has_endnotes": False,
                "sections": [
                    b["text"].split("\n")[0].strip()[:80]
                    for b in labeled_blocks
                    if b.get("label") == "paragraph_title" and b.get("text", "").strip()
                ],
                "total_pages": paddle_structure.get("total_pages", 0),
                "body_starts_at_page": labeled_blocks[0]["page"] if labeled_blocks else 1,
                "back_matter_starts_at_page": None,
            }
            print(f"     Title:      {doc_scan['title']}")
            if doc_scan['authors']:
                print(f"     Author(s):  {', '.join(doc_scan['authors'])}")
            print(f"     Sections:   {len(doc_scan['sections'])} headings from paddle labels")
            print(f"     TOC: {'Yes' if doc_scan['has_toc'] else 'No'}")
            print(f"     References: {'Yes' if doc_scan['has_references'] else 'No'}")

        elif ext == ".pdf":
            from src.doc_classifier import scan_document_structure, print_scan_summary
            # Build extracted text for section heading detection
            # Use markdown if available (fresh run), else reconstruct from chapters
            try:
                extracted_text = markdown  # type: ignore[possibly-undefined]
            except NameError:
                extracted_text = "\n\n".join(ch.get("text", "") for ch in chapters)
            doc_scan = scan_document_structure(input_path, llm, extracted_text=extracted_text)
            print_scan_summary(doc_scan)
        else:
            # EPUB — no raw-page scanning; use barebones metadata
            doc_scan = {
                "title": None, "subtitle": None, "authors": [],
                "has_toc": False, "has_abstract": False,
                "has_references": False, "has_endnotes": False,
                "sections": [], "total_pages": 0,
                "body_starts_at_page": 0, "back_matter_starts_at_page": None,
            }

    # Build doc_metadata from the scan (replaces invisible PDF metadata)
    doc_metadata = {
        "title": doc_scan.get("title"),
        "author": ", ".join(doc_scan.get("authors", [])) or None,
        "has_toc": doc_scan.get("has_toc", False),
        "sections": doc_scan.get("sections", []),
    }

    # For paddle: use LLM boundary detection for title + author
    # (more reliable than paddle labels which can pick up series titles)
    if use_paddle:
        try:
            if doc_boundary_title:
                doc_metadata["title"] = doc_boundary_title
        except NameError:
            pass

        try:
            if doc_boundary_author:
                doc_metadata["author"] = doc_boundary_author
        except NameError:
            pass

    # CLI overrides take highest priority
    if args.title:
        doc_metadata["title"] = args.title
    if args.author:
        doc_metadata["author"] = args.author

    # ── Hallucination check: verify title/author against source text ──────
    # Only for scanned PDFs where OCR noise causes the LLM to fabricate names.
    # Text-based PDFs get clean input, so the LLM reliably echoes real metadata.
    is_scanned = False
    try:
        is_scanned = paddle_structure.get("is_scanned", False)
    except NameError:
        pass

    if is_scanned and not args.author and not args.title:
        try:
            source_text = " ".join(
                b.get("text", "") for b in labeled_blocks
            ).lower()

            def _verify_against_source(value: str, label: str) -> str | None:
                """Return value if it appears in source, else None."""
                if not value:
                    return None
                # Split into significant words (skip short/common ones)
                words = [w for w in value.split() if len(w) > 2]
                if not words:
                    return value  # Nothing to check
                found = sum(1 for w in words if w.lower() in source_text)
                ratio = found / len(words)
                if ratio < 0.8:
                    print(f"  ⚠️  Dropping hallucinated {label}: \"{value}\" "
                          f"({found}/{len(words)} words found in source)")
                    return None
                return value

            doc_metadata["title"] = _verify_against_source(
                doc_metadata.get("title"), "title"
            )
            doc_metadata["author"] = _verify_against_source(
                doc_metadata.get("author"), "author"
            )
        except NameError:
            pass  # labeled_blocks not available (non-paddle path)

    print(f"  📌 Title: {doc_metadata['title']}")
    print(f"  📌 Author: {doc_metadata['author']}")

    # Save scan results
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(scan_path, "w") as f:
        json.dump(doc_scan, f, ensure_ascii=False, indent=2)
    with open(INTERMEDIATE_DIR / "doc_metadata.json", "w") as f:
        json.dump(doc_metadata, f, ensure_ascii=False, indent=2)

    # ── Step 5: Chunk Classification ─────────────────────────────────────────
    if use_paddle:
        # Paddle path: section_type already set from paddle labels in create_chunks_from_blocks
        _step("5/10", "Chunk Classification (skipped — using paddle labels)")
        body_count = sum(1 for c in chunks if c.get("section_type") == "BODY")
        other_count = len(chunks) - body_count
        print(f"  ✅ Using paddle labels: {body_count} BODY, {other_count} pass-through")
    else:
        _step("5/10", "Chunk Classification")
        from src.doc_classifier import classify_chunks, print_classification_summary

        section_labels = classify_chunks(chunks, llm)
        for i, label in enumerate(section_labels):
            chunks[i]["section_type"] = label
        print_classification_summary(chunks)

    # Save updated chunks with labels
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # ── Step 6: Glossary & Summary ───────────────────────────────────────────
    if args.resume and GLOSSARY_FILE.exists() and BOOK_SUMMARY_FILE.exists():
        _step("6/10", "Loading cached glossary & summary")
        with open(GLOSSARY_FILE) as f:
            glossary = json.load(f)
        with open(BOOK_SUMMARY_FILE) as f:
            book_summary = f.read()
        print(f"  {len(glossary)} terms, {len(book_summary)} char summary")
    else:
        _step("6/10", "Glossary & Book Summary")
        from src.chunker import extract_glossary, generate_book_summary
        glossary = extract_glossary(chapters, llm_client=llm)
        book_summary = generate_book_summary(chapters, llm_client=llm)
        print(f"  {len(glossary)} glossary terms extracted")

        with open(GLOSSARY_FILE, "w") as f:
            json.dump(glossary, f, ensure_ascii=False, indent=2)
        with open(BOOK_SUMMARY_FILE, "w") as f:
            f.write(book_summary)

    # ── Step 7: Simplification ───────────────────────────────────────────────
    _step("7/10", "Simplification")
    from src.simplifier import simplify_chunks, generate_footnotes

    output_chunks = simplify_chunks(
        chunks=chunks,
        llm=llm,
        book_summary=book_summary,
        glossary=glossary,
        temperature=TEMPERATURE,
        min_ratio=MIN_ACCEPTABLE_RATIO,
        max_ratio=MAX_ACCEPTABLE_RATIO,
        max_retries=MAX_RETRIES,
        max_workers=workers,
        checkpoint_path=CHECKPOINT_FILE,

        paragraph_mode=not args.chunk,
        strict_paragraphs=use_paddle,
    )

    with open(SIMPLIFIED_CHUNKS_FILE, "w") as f:
        json.dump(output_chunks, f, ensure_ascii=False, indent=2)

    # Paragraph audit: after simplification
    _simp_para_count = sum(
        len([p.strip() for p in c.get('simplified_text', '').split('\n\n') if p.strip()])
        for c in output_chunks
    )
    print(f"\n  📊 Paragraph count after simplification: {_simp_para_count} (input was {_input_para_count})")
    if _simp_para_count != _input_para_count:
        print(f"     ⚠️  Drift: {_simp_para_count - _input_para_count:+d} paragraphs")

    # ── Step 5b: Embedding QA ────────────────────────────────────────────────
    from src.embedding_qa import is_available as embedding_available, run_embedding_qa
    if embedding_available():
        print("\n  🔬 Running embedding-based quality check...")
        # Reload checkpoint for embedding QA
        with open(CHECKPOINT_FILE, "r") as f:
            eq_checkpoint = json.load(f)
        eq_result = run_embedding_qa(chunks, eq_checkpoint)
        print(f"     Checked: {eq_result['total_checked']} chunks")
        print(f"     Avg semantic similarity: {eq_result['avg_similarity']:.2%}")
        if eq_result["flagged_chunks"]:
            print(f"     ⚠️  {len(eq_result['flagged_chunks'])} chunks flagged (low similarity):")
            for cid, score in eq_result["flagged_chunks"][:5]:
                print(f"        Chunk {cid}: {score:.2%}")

            # Run targeted correction pass on flagged chunks
            from src.simplifier import run_correction_pass
            correction_threshold = 0.75
            needs_correction = [(cid, s) for cid, s in eq_result["flagged_chunks"] if s < correction_threshold]
            if needs_correction:
                print(f"\n  🔧 Running correction pass on {len(needs_correction)} chunk(s) below {correction_threshold:.0%}...")
                improved = run_correction_pass(
                    chunks=chunks,
                    output_chunks=output_chunks,
                    flagged_chunks=needs_correction,
                    llm=llm,
                    checkpoint_path=CHECKPOINT_FILE,
                    similarity_threshold=correction_threshold,
                )
                if improved > 0:
                    print(f"     ✅ Improved {improved} chunk(s)")
                    # Save updated chunks
                    with open(SIMPLIFIED_CHUNKS_FILE, "w") as f:
                        json.dump(output_chunks, f, ensure_ascii=False, indent=2)
                else:
                    print(f"     ℹ️  No chunks could be improved")
        else:
            print(f"     ✅ All chunks pass semantic similarity check")
    else:
        print("\n  ℹ️  Embedding QA skipped (install sentence-transformers for semantic checks)")

    # ── Step 8: Footnotes ─────────────────────────────────────────────────────
    _step("8/10", "Footnote Generation")
    footnotes = generate_footnotes(
        output_chunks=output_chunks,
        llm=llm,
        max_workers=workers,
    )

    # ── Step 9: Assembly & Concept Map ───────────────────────────────────────
    _step("9/10", "Assembly")
    from src.assembler import (
        trim_overlaps, assemble_book, generate_toc,
        save_output, compare_lengths, insert_footnotes, qa_check_output,
    )

    # Generate reader-facing intro (paddle only)
    if use_paddle:
        print("  📝 Generating reader introduction...")
        from src.prompts import build_intro_prompt
        try:
            intro_sys, intro_usr = build_intro_prompt(
                doc_summary=book_summary,
                title=doc_metadata.get("title", ""),
                author=doc_metadata.get("author", ""),
            )
            intro_text = llm.generate(
                prompt=intro_usr,
                system_prompt=intro_sys,
                temperature=0.3,
            )
            if intro_text:
                doc_metadata["intro"] = intro_text.strip()
                print(f"  ✅ Intro: {intro_text.strip()[:100]}...")
        except Exception as e:
            print(f"  ⚠️  Intro generation failed: {str(e)[:100]}")

    # doc_metadata was built from the scan in Step 4 — no file reload needed
    if OVERLAP_TOKENS > 0:
        trimmed = trim_overlaps(output_chunks, overlap_tokens=OVERLAP_TOKENS)
    else:
        trimmed = output_chunks  # No overlap to trim
    book_md = assemble_book(trimmed, chapters, metadata=doc_metadata)

    # Paragraph audit: after assembly
    _output_para_count = len([p.strip() for p in book_md.split('\n\n') if p.strip()])
    print(f"\n  📊 Paragraph count after assembly: {_output_para_count} (input was {_input_para_count})")
    if _output_para_count != _input_para_count:
        print(f"     ⚠️  Drift: {_output_para_count - _input_para_count:+d} paragraphs")

    # Concept map
    from src.simplifier import generate_concept_map
    concept_map_md = generate_concept_map(book_md, llm)

    # Insert footnotes
    book_md = insert_footnotes(book_md, footnotes)

    # Insert concept map before footnotes section
    if concept_map_md:
        for marker in ["\n## Notes", "\n## Footnotes"]:
            if marker in book_md:
                pos = book_md.find(marker)
                divider_pos = book_md.rfind("\n---\n", max(0, pos - 10), pos)
                insert_at = divider_pos if divider_pos >= 0 else pos
                book_md = book_md[:insert_at] + concept_map_md + book_md[insert_at:]
                break
        else:
            book_md = book_md.rstrip() + concept_map_md

    # TOC — use the scan-derived signal only (no stale metadata)
    # Always generate TOC from output headings (original TOC page numbers are stale)
    toc = generate_toc(book_md, force=True)
    if toc:
        divider_pos = book_md.find("---")
        if divider_pos > 0:
            insert_pos = book_md.find("\n", divider_pos) + 1
            book_md = book_md[:insert_pos] + "\n" + toc + "\n" + book_md[insert_pos:]
        else:
            book_md = toc + "\n\n" + book_md

    final = book_md

    # ── Step 10: QA Check ────────────────────────────────────────────────────
    _step("10/10", "Quality Assurance")

    # 10a: Pattern-based QA — fix duplicate headings, paragraphs, bylines
    from src.final_qa import run_final_qa, run_landmark_qa
    final_qa_result = run_final_qa(final)
    if final_qa_result["issues_fixed"] > 0:
        final = final_qa_result["fixed_markdown"]
        print(f"  🔧 Fixed {final_qa_result['issues_fixed']} output anomalies:")
        for issue in final_qa_result["issues_found"]:
            print(f"     • {issue}")
    else:
        print("  ✅ No output anomalies found")

    # 10b: LLM-based landmark QA — fix broken headings, bylines, fragments
    print("\n  🔬 Reviewing structural landmarks...")
    landmark_result = run_landmark_qa(final, llm)
    if landmark_result["issues_fixed"] > 0:
        final = landmark_result["fixed_markdown"]
        print(f"  🔧 Fixed {landmark_result['issues_fixed']} landmark issue(s):")
        for issue in landmark_result["issues_found"]:
            print(f"     • {issue}")
    else:
        print("  ✅ All landmarks OK")

    # Load structure map for validation
    struct_path = INTERMEDIATE_DIR / "doc_structure.json"
    if struct_path.exists():
        with open(struct_path) as f:
            doc_structure = json.load(f)
    else:
        doc_structure = {}

    qa_result = qa_check_output(final, chapters, doc_metadata, doc_structure)

    if qa_result["issues"]:
        for issue in qa_result["issues"]:
            icon = "❌" if issue["severity"] == "ERROR" else "⚠️"
            print(f"  {icon} [{issue['severity']}] {issue['message']}")

        if qa_result["fixed_markdown"] != final:
            final = qa_result["fixed_markdown"]
            print(f"  🔧 Auto-fixed {sum(1 for i in qa_result['issues'] if i['severity'] == 'ERROR')} issue(s)")
    else:
        print("  ✅ All checks passed")

    length_report = compare_lengths(
        [c for c in output_chunks],
        [c for c in output_chunks],
    )

    saved = save_output(final, output_path)

    # ── Done ─────────────────────────────────────────────────────────────────
    total_time = time.time() - total_start

    print(f"""
╔══════════════════════════════════════════════════╗
║  ✅ Done!                                        ║
╠══════════════════════════════════════════════════╣
║  Output:     {str(saved):<36s}║
║  Words:      {length_report['original_words']:,} → {length_report['simplified_words']:,} ({length_report['ratio']}x){' ' * max(0, 20 - len(f"{length_report['original_words']:,} → {length_report['simplified_words']:,} ({length_report['ratio']}x)"))}║
║  Time:       {total_time:.0f}s ({total_time/60:.1f} min){' ' * max(0, 24 - len(f"{total_time:.0f}s ({total_time/60:.1f} min)"))}║
╚══════════════════════════════════════════════════╝
""")


def _step(number: str, title: str):
    """Print a formatted step header."""
    print(f"\n{'─'*50}")
    print(f"  [{number}] {title}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
