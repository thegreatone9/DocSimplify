# 📚 DocSimplify — Neural PDF Simplifier

An AI-powered pipeline that transforms dense academic PDFs and EPUBs into plain-English Markdown, while preserving all information. Built for accuracy, not summarization.

## ✨ Features

- **Multi-format input** — PDF and EPUB support with automatic format detection
- **Multi-LLM backends** — Ollama (local), Groq, and Gemini (cloud)
- **Paragraph-level processing** — Breaks complex chunks into manageable paragraphs for better quality on smaller models
- **Rolling context compression** — Bridge summaries maintain coherence between chunks without consuming context window
- **Paragraph realignment** — Output matches the input's paragraph structure (no paragraph bloat)
- **Tone smoothing** — Post-simplification LLM pass harmonizes voice across paragraphs
- **Embedding-based QA** — Detects semantic drift using sentence-transformers (no LLM calls)
- **Final output QA** — Catches duplicate headings, meta-summaries, and other anomalies
- **Checkpointing** — Resume interrupted runs without re-processing completed chunks
- **Glossary & footnotes** — Technical terms get footnote definitions in the output

## 🏗️ Architecture

```
PDF/EPUB ──► Parser ──► Chunker ──► Simplifier ──► Assembler ──► Markdown
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
               Ollama (local)     Groq (cloud)       Gemini (cloud)
               qwen2.5:7b/14b    llama-3.1-8b       gemini-2.0-flash
```

### Pipeline Steps

| Step | Description |
|------|-------------|
| 1. Parse | Extract text from PDF (PyMuPDF/Surya) or EPUB |
| 2. Structure | Detect chapters, headings, document metadata |
| 3. Chunk | Split into ~2000-token chunks with 200-token overlap |
| 4. Glossary | Extract technical terms + generate book summary |
| 5. Simplify | Rewrite each chunk in plain English (paragraph or chunk mode) |
| 5b. Embedding QA | Check semantic similarity of original vs simplified |
| 6. Footnotes | Generate footnote definitions for glossary terms |
| 7. Assemble | Stitch chunks into Markdown with TOC and footnotes |
| 8. Final QA | Fix duplicates, meta-summaries, structural anomalies |

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Optional: embedding-based quality assurance
pip install sentence-transformers

# Optional: neural OCR for scanned PDFs
pip install surya-ocr "transformers<5.0"
```

### 2. Configure

Copy and edit the config file:

```bash
cp config.py.example config.py
```

Set your API keys (if using cloud LLMs):

```python
# config.py
GEMINI_API_KEY = "your-key-here"  # https://aistudio.google.com/apikey
GROQ_API_KEY   = "your-key-here"  # https://console.groq.com/keys
```

### 3. Add your document

Place a PDF or EPUB file in `data/input/`.

### 4. Run

```bash
# Local (Ollama) — no rate limits
python3 run.py

# Local with paragraph mode (better quality for small models)
python3 run.py --model qwen2.5:14b --paragraph

# Groq cloud — fast, free
python3 run.py --groq --paragraph

# Gemini cloud
python3 run.py --gemini

# Scanned PDF with neural OCR
python3 run.py --surya --groq --paragraph

# Resume an interrupted run
python3 run.py --groq --paragraph --resume
```

Output will be saved to `data/output/<filename>_simplified.md`.

## 🔧 CLI Flags

| Flag | Description |
|------|-------------|
| `--model NAME` | Ollama model name (default: `qwen2.5:7b`) |
| `--gemini` | Use Gemini API |
| `--groq` | Use Groq API (Llama 3.1 8B default) |
| `--surya` | Use Surya neural OCR for PDF parsing |
| `--paragraph` | Paragraph-level processing (recommended for ≤14B models) |
| `--workers N` | Parallel workers (default: 2, forced to 1 in paragraph mode) |
| `--no-verify` | Skip the verification pass |
| `--resume` | Resume from checkpoint |

## 📁 Project Structure

```
DocSimplify/
├── run.py                  # Main entry point & CLI
├── config.py               # API keys, model settings, tunable params
├── requirements.txt        # Python dependencies
├── src/
│   ├── pdf_parser.py       # PDF text extraction (PyMuPDF + Surya)
│   ├── epub_parser.py      # EPUB text extraction
│   ├── chunker.py          # Token-aware text chunking
│   ├── prompts.py          # All LLM prompt templates
│   ├── simplifier.py       # Core simplification logic
│   ├── assembler.py        # Output assembly & TOC generation
│   ├── llm_client.py       # Ollama client
│   ├── gemini_client.py    # Gemini API client
│   ├── groq_client.py      # Groq API client
│   ├── embedding_qa.py     # Sentence-transformer QA
│   ├── final_qa.py         # Post-assembly anomaly detection
│   └── sanity_check.py     # Structural validation
├── data/
│   ├── input/              # Place PDFs/EPUBs here
│   ├── intermediate/       # Chunks, checkpoints, glossary
│   └── output/             # Final simplified Markdown
└── notebooks/              # Development notebooks
```

## 🧠 Quality Techniques

### For Weaker Models (≤14B)

The `--paragraph` flag enables a multi-stage quality pipeline:

1. **Few-shot prompting** — Concrete before/after example in every prompt
2. **Paragraph-level processing** — Each paragraph simplified individually (simpler task)
3. **Similar paragraph merging** — Embedding similarity joins related consecutive paragraphs
4. **Tone smoothing** — Single LLM call harmonizes voice across merged paragraphs
5. **Structure realignment** — Output paragraph count matches input (prevents paragraph bloat)
6. **Rolling bridge summaries** — 2-sentence context compression between chunks

### Post-Processing QA

- **Embedding QA** — Flags chunks with <75% cosine similarity to original
- **Meta-summary removal** — Detects "The passage argues..." LLM artifacts
- **Duplicate detection** — Removes repeated headings and paragraphs
- **Byline deduplication** — Catches duplicate author lines from PDF metadata

## ⚡ LLM Provider Comparison

| Provider | Model | Speed | Free Tier | Best For |
|----------|-------|-------|-----------|----------|
| **Ollama** | qwen2.5:14b | ~15 tok/s | Unlimited | Paragraph mode, no quotas |
| **Groq** | llama-3.1-8b | ~800 tok/s | 14K req/day | Fast processing |
| **Groq** | llama-3.3-70b | ~300 tok/s | 1K req/day | Best free quality |
| **Gemini** | gemini-2.0-flash | ~100 tok/s | 1.5K req/day | Good balance |

## 📝 License

Personal project — not licensed for distribution.
