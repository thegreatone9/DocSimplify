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
    VERIFY_CHUNKS, MIN_ACCEPTABLE_RATIO, MAX_ACCEPTABLE_RATIO,
    MAX_RETRIES, MAX_WORKERS,
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
)

SUPPORTED_EXTENSIONS = {".pdf", ".epub"}


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
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the verification pass (faster)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint (don't re-extract)")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"Ollama model name (default: {MODEL_NAME})")
    parser.add_argument("--surya", action="store_true",
                        help="Use surya neural layout model (for scanned/image-heavy PDFs)")
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
    verify = VERIFY_CHUNKS and not args.no_verify
    workers = args.workers
    model = args.model
    use_surya = args.surya
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

    # Check surya availability if requested
    if use_surya:
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

    extraction_mode = "surya (neural)" if use_surya else "heuristic (font/position)"
    llm_mode = "Gemini API" if use_gemini else ("Groq API" if use_groq else f"Ollama ({model})")
    print(f"""
╔══════════════════════════════════════════════════╗
║           📚 Book Simplifier                     ║
╚══════════════════════════════════════════════════╝
  Input:    {input_path.name}
  LLM:     {llm_mode}
  Parser:   {extraction_mode}
  Workers:  {workers} ({'parallel' if workers > 1 else 'sequential + context'})
  Verify:   {'ON' if verify else 'OFF'}
  Output:   {output_path.name}
""")

    total_start = time.time()

    # ── Step 1: Sanity Check ─────────────────────────────────────────────────
    _step("1/8", "Sanity Check")
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

    # ── Step 2: Extract & Chunk ──────────────────────────────────────────────
    if args.resume and CHUNKS_FILE.exists():
        _step("2/8", "Loading cached extraction")
        with open(BOOK_STRUCTURE_FILE) as f:
            chapters = json.load(f)
        with open(CHUNKS_FILE) as f:
            chunks = json.load(f)
        print(f"  Loaded {len(chapters)} sections, {len(chunks)} chunks")
    else:
        _step("2/8", "Extraction & Chunking")

        if ext == ".pdf" and use_surya:
            from src.pdf_parser import extract_pdf_with_surya, detect_chapters, validate_extraction, extract_pdf_metadata
            print("  🧠 Running surya neural layout detection (this may take a few minutes)...")
            markdown, surya_structure = extract_pdf_with_surya(input_path)
            chapters = detect_chapters(markdown)
            report = validate_extraction(chapters, input_path)
            doc_metadata = extract_pdf_metadata(input_path)
        elif ext == ".pdf":
            from src.pdf_parser import extract_pdf_to_markdown, detect_chapters, validate_extraction, extract_pdf_metadata
            markdown = extract_pdf_to_markdown(input_path)
            chapters = detect_chapters(markdown)
            report = validate_extraction(chapters, input_path)
            doc_metadata = extract_pdf_metadata(input_path)
        else:
            from src.epub_parser import parse_epub_chapters, validate_epub_extraction
            chapters = parse_epub_chapters(input_path)
            report = validate_epub_extraction(chapters)
            doc_metadata = {"title": None, "author": None, "has_toc": False}

        if doc_metadata.get("title"):
            print(f"  Title:  {doc_metadata['title']}")
        if doc_metadata.get("author"):
            print(f"  Author: {doc_metadata['author']}")

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

        # Save intermediate files
        INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(BOOK_STRUCTURE_FILE, "w") as f:
            json.dump(chapters, f, ensure_ascii=False, indent=2)
        with open(CHUNKS_FILE, "w") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        with open(INTERMEDIATE_DIR / "doc_metadata.json", "w") as f:
            json.dump(doc_metadata, f, ensure_ascii=False, indent=2)
        with open(INTERMEDIATE_DIR / "doc_structure.json", "w") as f:
            json.dump(doc_structure, f, ensure_ascii=False, indent=2)

    # ── Step 3: LLM Setup ────────────────────────────────────────────────────
    _step("3/8", "LLM Setup")

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

    # ── Step 4: Glossary & Summary ───────────────────────────────────────────
    if args.resume and GLOSSARY_FILE.exists() and BOOK_SUMMARY_FILE.exists():
        _step("4/8", "Loading cached glossary & summary")
        with open(GLOSSARY_FILE) as f:
            glossary = json.load(f)
        with open(BOOK_SUMMARY_FILE) as f:
            book_summary = f.read()
        print(f"  {len(glossary)} terms, {len(book_summary)} char summary")
    else:
        _step("4/8", "Glossary & Book Summary")
        from src.chunker import extract_glossary, generate_book_summary
        glossary = extract_glossary(chapters, llm_client=llm)
        book_summary = generate_book_summary(chapters, llm_client=llm)
        print(f"  {len(glossary)} glossary terms extracted")

        with open(GLOSSARY_FILE, "w") as f:
            json.dump(glossary, f, ensure_ascii=False, indent=2)
        with open(BOOK_SUMMARY_FILE, "w") as f:
            f.write(book_summary)

    # ── Step 5: Simplification ───────────────────────────────────────────────
    _step("5/8", "Simplification")
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
        verify=verify,
        paragraph_mode=args.paragraph,
    )

    with open(SIMPLIFIED_CHUNKS_FILE, "w") as f:
        json.dump(output_chunks, f, ensure_ascii=False, indent=2)

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
        else:
            print(f"     ✅ All chunks pass semantic similarity check")
    else:
        print("\n  ℹ️  Embedding QA skipped (install sentence-transformers for semantic checks)")

    # ── Step 6: Footnotes ────────────────────────────────────────────────────
    _step("6/8", "Footnote Generation")
    footnotes = generate_footnotes(
        output_chunks=output_chunks,
        llm=llm,
        max_workers=workers,
    )

    # ── Step 7: Assembly ─────────────────────────────────────────────────────
    _step("7/8", "Assembly")
    from src.assembler import (
        trim_overlaps, assemble_book, generate_toc,
        save_output, compare_lengths, insert_footnotes, qa_check_output,
    )

    # Load metadata (may have been saved during extraction or resume)
    meta_path = INTERMEDIATE_DIR / "doc_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            doc_metadata = json.load(f)
    else:
        doc_metadata = {}

    trimmed = trim_overlaps(output_chunks, overlap_tokens=OVERLAP_TOKENS)
    book_md = assemble_book(trimmed, chapters, metadata=doc_metadata)
    book_md = insert_footnotes(book_md, footnotes)

    # Only add TOC if the original document had one
    toc = generate_toc(book_md, force=doc_metadata.get("has_toc", False))
    if toc:
        divider_pos = book_md.find("---")
        if divider_pos > 0:
            insert_pos = book_md.find("\n", divider_pos) + 1
            book_md = book_md[:insert_pos] + "\n" + toc + "\n" + book_md[insert_pos:]
        else:
            book_md = toc + "\n\n" + book_md

    final = book_md

    # ── Step 8: QA Check ─────────────────────────────────────────────────────
    _step("8/8", "Quality Assurance")

    # Final output QA: fix duplicate headings, paragraphs, bylines
    from src.final_qa import run_final_qa
    final_qa_result = run_final_qa(final)
    if final_qa_result["issues_fixed"] > 0:
        final = final_qa_result["fixed_markdown"]
        print(f"  🔧 Fixed {final_qa_result['issues_fixed']} output anomalies:")
        for issue in final_qa_result["issues_found"]:
            print(f"     • {issue}")
    else:
        print("  ✅ No output anomalies found")

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
