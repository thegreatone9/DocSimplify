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
    parser.add_argument("--paddle", action="store_true",
                        help="Use PaddleOCR layout detection (recommended for academic PDFs)")
    parser.add_argument("--gemini", action="store_true",
                        help="Use Gemini API (requires GEMINI_API_KEY in config)")
    parser.add_argument("--groq", action="store_true",
                        help="Use Groq API with Llama 3.3 70B (fast, free, generous limits)")
    parser.add_argument("--paragraph", action="store_true",
                        help="Simplify paragraph-by-paragraph (better quality for weaker models)")

    args = parser.parse_args()

    # ── Auto-detect input file ───────────────────────────────────────────────
    input_path = _find_input_file()
    ext = input_path.suffix.lower()

    workers = args.workers
    model = args.model
    use_surya = args.surya
    use_paddle = args.paddle
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

    # Paragraph mode must be sequential (many sub-requests per chunk)
    if args.paragraph:
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

    # ── Default: start fresh. Only keep intermediate files if --resume is passed ──
    if not args.resume:
        # Clear ALL intermediate and output files to ensure clean state
        import glob
        for stale_file in glob.glob(str(INTERMEDIATE_DIR / "*")):
            Path(stale_file).unlink()
        if output_path.exists():
            output_path.unlink()
        print("  🆕 Starting fresh (use --resume to continue a previous run)")

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
        _step("4/10", "Document Structure Scan (raw pages)")
        if ext == ".pdf":
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

    # For paddle: override title/author from paddle's exact labels
    # (the LLM scan can truncate titles)
    if use_paddle:
        try:
            for b in labeled_blocks:
                if b["label"] in ("doc_title", "title"):
                    paddle_title = b["text"].replace("\n", " ").strip()
                    # Remove markdown heading prefix if present
                    paddle_title = paddle_title.lstrip("# ").strip()
                    if paddle_title:
                        doc_metadata["title"] = paddle_title
                        print(f"  📌 Using paddle title: {paddle_title}")
                    break
        except NameError:
            pass  # labeled_blocks not available (resume mode)

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

        paragraph_mode=args.paragraph,
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
    # Skip in strict paddle mode — paragraph integrity is guaranteed by construction
    if use_paddle:
        print("\n  ℹ️  Embedding QA skipped (strict paddle mode — paragraph integrity guaranteed)")
    else:
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
                correction_threshold = 0.60
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
    had_toc = doc_scan.get("has_toc", False)
    toc = generate_toc(book_md, force=had_toc)
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
