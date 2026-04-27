"""
Prompt Templates
================
All LLM prompts live here - simplification, verification, glossary extraction,
and book summary generation. Centralized so they're easy to iterate on.
"""

import json


def build_simplification_prompt(
    chunk_text: str,
    book_summary: str,
    glossary: dict,
    previous_context: str = "",
    is_retry: bool = False,
) -> tuple[str, str]:
    """
    Build the system + user prompts for simplifying one chunk of text.

    Args:
        chunk_text:       The text to simplify.
        book_summary:     ~500-token overview of the book.
        glossary:         Dict of {term: plain-english definition}.
        previous_context: Brief summary of what came just before this chunk.
        is_retry:         If True, this is a retry attempt - use even stricter prompt.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    import re

    # Count paragraphs for the explicit matching rule
    input_paragraphs = [p.strip() for p in chunk_text.split("\n\n") if p.strip()]
    paragraph_count = len(input_paragraphs)
    input_word_count = len(chunk_text.split())

    # Detect numbered items and bullets in this chunk
    numbered_items = re.findall(r"^(\d+)[\.\)]", chunk_text, re.MULTILINE)
    bullet_items = re.findall(r"^[-*\u2022]\s", chunk_text, re.MULTILINE)

    # Format glossary as a readable list
    glossary_text = ""
    if glossary:
        glossary_lines = [f"  • {term}: {defn}" for term, defn in glossary.items()]
        glossary_text = "\n".join(glossary_lines)

    system_prompt = f"""You are an expert book editor who rewrites dense, academic texts 
in plain English without losing ANY information. You rewrite - you do NOT summarize.

TARGET AUDIENCE: A high school graduate - intelligent and curious, but not specialized.

EXAMPLE:
Original: "The neoliberal paradigm, predicated on fiscal austerity and market 
deregulation, has engendered significant macroeconomic volatility in peripheral economies, 
exacerbating extant inequalities in wealth distribution."
Rewritten: "The economic approach based on cutting government spending and removing market 
regulations has caused major economic instability in smaller, developing economies. This 
has made the existing gaps between rich and poor even worse."

Notice: same information, simpler words, sentences split for clarity, roughly same length.

BOOK CONTEXT:
{book_summary}

KEY TERMS GLOSSARY (use these definitions consistently):
{glossary_text if glossary_text else "(No glossary provided)"}"""

    # Build the previous context section
    context_section = ""
    if previous_context:
        context_section = f"""
WHAT CAME JUST BEFORE THIS PASSAGE:
{previous_context}
"""

    # Build retry-specific emphasis
    retry_warning = ""
    if is_retry:
        retry_warning = """
⚠️ YOUR PREVIOUS ATTEMPT WAS TOO SHORT. You DROPPED information. This time, go 
paragraph by paragraph and make sure EVERY point from the original appears in your output. 
Your output MUST be at least as many words as the input.
"""
    # Note: numbered items and bullets are naturally preserved by the LLM
    # via rule #1 ("keep ALL facts"). Explicit structure notes were removed
    # because the regex detection produced false positives (page numbers,
    # footnote refs) that caused the LLM to fabricate fake numbered paragraphs.

    # Build placeholder note if non-prose elements were stripped
    placeholder_note = ""
    has_placeholders = any(marker in chunk_text for marker in ["<<IMG_", "<<TABLE_", "<<EQ_", "<<BLOCKQUOTE_", "<<HR_"])
    if has_placeholders:
        placeholder_note = """
6. PLACEHOLDERS: The text contains <<IMG_N>>, <<TABLE_N>>, <<EQ_N>>, <<BLOCKQUOTE_N>>, or <<HR_N>> markers.
   These represent images, tables, equations, citations, or horizontal rules.
   Copy them EXACTLY into your output in the same position. Do NOT remove, rename, or rewrite them.
"""

    user_prompt = f"""Rewrite the following passage in plain, clear English for a high school graduate.
{context_section}{retry_warning}
RULES:
1. This is a REWRITE, not a summary. Keep ALL facts, arguments, names, dates, and details. 
Your output should be roughly the same length (~{input_word_count} words, {paragraph_count} paragraphs).
2. Replace jargon with plain language. If a technical term has no simpler equivalent, 
briefly explain it in parentheses the first time.
3. Break long sentences into shorter, clearer ones. Keep the author's meaning and logical flow.
4. Do NOT add commentary, headings, labels, or new information that isn't in the original.
5. Output ONLY the rewritten text - nothing else.
{placeholder_note}
PASSAGE TO SIMPLIFY:

{chunk_text}"""

    return system_prompt, user_prompt


def build_paragraph_prompt(
    paragraph_text: str,
    book_summary: str,
    glossary: dict,
    previous_context: str = "",
) -> tuple[str, str]:
    """
    Build a minimal prompt for simplifying a single paragraph.
    Much simpler task = much better results from weaker models.

    Args:
        paragraph_text: A single paragraph to simplify.
        book_summary:   Book-level context.
        glossary:       Term→definition dict.
        previous_context: Brief bridge summary from previous content.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    glossary_text = ""
    if glossary:
        glossary_lines = [f"  • {term}: {defn}" for term, defn in glossary.items()]
        glossary_text = "\n".join(glossary_lines)

    system_prompt = f"""You rewrite dense academic text into clear, educated prose. Keep ALL information - 
rewrite, don't summarize. Match the original length.

TARGET AUDIENCE: A high school graduate — literate and intelligent, but without specialized 
academic training. Write at the level of a good newspaper or popular nonfiction book.

CRITICAL RULES:
1. Never write about the text - rewrite IT. Never start with "The passage argues", 
"The author states", etc. Write as if YOU are the author making the same points.
2. Replace obscure jargon with clear equivalents, but don't oversimplify.
   Use words like "government spending" instead of "fiscal expenditure", but keep words 
   like "deficit", "inflation", "capital" — a high schooler knows these.
3. Mix short and long sentences for natural rhythm. Not every sentence should be the same length.
4. Keep the author's logical connectives that carry argument weight - words like "as distinct 
from this", "to be sure", "indeed", "in retrospect". Don't flatten them all to "but" or "also".
5. When the original explains WHY something happens, keep the full causal chain. Do not 
simplify "X happened because of A, B, and C" into just "X happened because of A".
6. NEVER repeat information. Each sentence must add something new.
7. Do NOT use childish phrasing like "big companies take stuff" or "money goes away". 
   Write like a serious author explaining complex ideas clearly.

EXAMPLE:
Original: "The neoliberal paradigm, predicated on fiscal austerity, has exacerbated 
extant inequalities."
Rewritten: "The economic approach built on cutting government spending has made existing 
inequalities worse."

BOOK CONTEXT: {book_summary[:500]}

GLOSSARY (Use these definitions for key terms):
{glossary_text if glossary_text else '(none)'}"""

    context_line = ""
    if previous_context:
        context_line = f"\nCONTEXT (What came right before this): {previous_context}\n"

    word_count = len(paragraph_text.split())

    # Build placeholder note if non-prose elements were stripped
    placeholder_note = ""
    if "<<IMG_" in paragraph_text or "<<TABLE_" in paragraph_text or "<<EQ_" in paragraph_text:
        placeholder_note = (
            "\nIMPORTANT: Copy any <<IMG_N>>, <<TABLE_N>>, or <<EQ_N>> placeholders "
            "EXACTLY into your output in the same position. Do NOT remove them.\n"
        )

    user_prompt = f"""Rewrite this paragraph in clear, accessible prose (~{word_count} words). 
Target the reading level of a good newspaper — educated but not academic. 
Keep all facts, arguments, and the full reasoning chain. Do NOT dumb it down to a children's level.
Do NOT repeat any point twice. Do NOT summarize or comment on the text - rewrite it directly. 
Output ONLY the rewritten text.
{placeholder_note}{context_line}
{paragraph_text}"""

    return system_prompt, user_prompt


def build_bridge_summary_prompt(simplified_text: str) -> tuple[str, str]:
    """
    Build a prompt to generate a 2-sentence bridge summary of a simplified chunk.
    Used for rolling context compression between chunks.

    Args:
        simplified_text: The just-simplified chunk text.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = "You write brief summaries to maintain continuity between text sections."

    user_prompt = f"""Summarize the following passage in exactly 2 sentences. 
Focus on the main argument and any key terms introduced. 
This summary will be used as context for the next section.

{simplified_text[-1500:]}"""

    return system_prompt, user_prompt


def build_correction_prompt(
    original_text: str,
    simplified_text: str,
    similarity_score: float,
) -> tuple[str, str]:
    """
    Build a prompt for correcting a simplified chunk that lost too much meaning.

    The LLM sees both the original and its failed attempt, and is asked to
    surgically restore missing information while keeping the simple language.

    Args:
        original_text:    The original chunk text.
        simplified_text:  The bad simplified version (low similarity).
        similarity_score: The embedding similarity score (0.0-1.0).

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    orig_words = len(original_text.split())

    system_prompt = (
        "You are a meticulous editor. A previous simplification attempt lost important "
        "meaning from the original text. Your job is to FIX the simplified version by "
        "restoring any missing facts, arguments, or details — while keeping the language "
        "simple and clear. Do NOT start from scratch. Patch the gaps."
    )

    user_prompt = f"""The simplified version below scored only {similarity_score:.0%} semantic similarity 
with the original — meaning significant content was lost or distorted.

Compare the ORIGINAL with the SIMPLIFIED VERSION and fix the simplified version:
1. Identify facts, arguments, names, or details present in the original but missing from the simplified version.
2. Add the missing content back into the simplified version, using simple language.
3. Fix any meaning that was distorted or over-generalized.
4. Keep the output roughly the same length as the original (~{orig_words} words).
5. Output ONLY the corrected text — nothing else.

ORIGINAL:

{original_text}

SIMPLIFIED VERSION (needs fixing):

{simplified_text}"""

    return system_prompt, user_prompt


def build_smoothing_prompt(simplified_text: str) -> tuple[str, str]:
    """
    Build a prompt for harmonizing tone/style across paragraphs after
    paragraph-level simplification.

    Args:
        simplified_text: The full simplified chunk text (all paragraphs joined).

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    word_count = len(simplified_text.split())

    para_count = len([p for p in simplified_text.split("\n\n") if p.strip()])

    system_prompt = (
        "You are an editor who harmonizes tone and style. "
        "You do NOT change the content, facts, or meaning. You only make the voice "
        "consistent throughout - same level of formality, same use of contractions, "
        "smooth transitions between paragraphs. "
        "CRITICAL: Do NOT split or add paragraphs. Keep the EXACT same number of paragraphs."
    )

    # Build placeholder note if non-prose elements exist
    placeholder_note = ""
    if "<<IMG_" in simplified_text or "<<TABLE_" in simplified_text or "<<EQ_" in simplified_text:
        placeholder_note = (
            "\nIMPORTANT: The text contains <<IMG_N>>, <<TABLE_N>>, or <<EQ_N>> placeholders. "
            "Copy them EXACTLY into your output. Do NOT remove or rewrite them.\n"
        )

    user_prompt = f"""Polish the following text for consistent tone and smooth transitions. 
Do NOT add, remove, or change any facts. Keep the same length (~{word_count} words). 
Keep EXACTLY {para_count} paragraphs - do NOT split any paragraph into multiple ones. 
Output ONLY the polished text.
{placeholder_note}
{simplified_text}"""

    return system_prompt, user_prompt


def build_verification_prompt(
    original_text: str,
    simplified_text: str,
) -> tuple[str, str]:
    """
    Build prompts to verify that no information was lost during simplification.

    Args:
        original_text:   The original chunk before simplification.
        simplified_text: The simplified output to verify.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = """You are a meticulous fact-checker and editor. Your job is to compare 
an original text with its simplified version and identify ANY information that was lost, 
altered, or fabricated during simplification."""

    user_prompt = f"""Compare the ORIGINAL text with its SIMPLIFIED version below.

List any of the following issues you find:
1. DROPPED - Facts, arguments, data points, or details present in the original but missing from the simplified version.
2. ALTERED - Meaning that was changed, distorted, or over-generalized in the simplified version.
3. HALLUCINATED - New information, claims, or examples in the simplified version that do NOT appear in the original.

If there are NO issues and all information is preserved, respond with exactly: "ALL PRESERVED - no issues found."

ORIGINAL:
{original_text}

SIMPLIFIED:
{simplified_text}

ISSUES FOUND:"""

    return system_prompt, user_prompt


def build_glossary_prompt(text_sample: str) -> tuple[str, str]:
    """
    Build prompts to extract key terms/jargon from a text sample.

    Args:
        text_sample: A representative sample of the book text (~3000 tokens).

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = (
        "You are a vocabulary analyst. You identify specialized, technical, "
        "or unusual words and phrases in a text and provide clear, plain-English definitions that "
        "a high school graduate would understand."
    )

    user_prompt = f"""Read the following text sample from a book. Identify ALL specialized
terms, jargon, academic vocabulary, and unusual phrases that a typical high school graduate
might not immediately understand.


For each term, provide a brief, clear, plain-English definition (1 sentence max).

Format your response as a JSON object where keys are terms and values are definitions.
Example: {{"epistemology": "The study of knowledge - how we know what we know", 
"ontological": "Related to the nature of existence and what it means for something to be real"}}

Return ONLY the JSON object, no other text.

TEXT SAMPLE:
{text_sample}"""

    return system_prompt, user_prompt


def build_summary_prompt(chapter_intros: str) -> tuple[str, str]:
    """
    Build prompts to generate a document-level summary from section introductions.

    Args:
        chapter_intros: Concatenated introductory text from each section.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = """You are a skilled document analyst. You read introductory passages from 
each section of a document and produce a concise, accurate overview of the document's topic, 
purpose, and main themes."""

    user_prompt = f"""Below are the opening passages from each section of a document. Based on 
these excerpts, write a concise overview (about 300-500 words) that describes:

1. What this document is about (main topic/subject)
2. Who the intended audience seems to be
3. The main themes or arguments the document explores
4. The overall structure or progression of ideas

Write in clear, plain English. This summary will be used to provide context when 
simplifying individual sections of the document.

Output ONLY the summary, no other text.

SECTION OPENINGS:
{chapter_intros}"""

    return system_prompt, user_prompt


def build_chapter_transition_summary(
    previous_chapter_text: str,
) -> tuple[str, str]:
    """
    Build prompts to generate a brief summary of the previous chapter.

    Args:
        previous_chapter_text: The full simplified text of the previous chapter.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = """You are a concise summarizer. You produce brief, accurate summaries 
of book chapters to maintain reading continuity."""

    # Only use the last portion if the chapter is very long
    max_chars = 6000
    text = previous_chapter_text[-max_chars:] if len(previous_chapter_text) > max_chars else previous_chapter_text

    user_prompt = f"""Summarize the following chapter text in 2-3 sentences. Focus on the 
key points, arguments, or events. This summary will be provided as context when simplifying 
the next chapter.

Output ONLY the summary, no other text.

CHAPTER TEXT:
{text}"""

    return system_prompt, user_prompt


def build_footnote_prompt(simplified_text: str) -> tuple[str, str]:
    """
    Build prompts to identify terms in simplified text that need footnote explanations.

    The LLM uses its own internal knowledge to provide concise definitions.

    Args:
        simplified_text: A chunk of already-simplified text.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = """You are an expert educator. You read text and identify every term, 
name, concept, or reference that a typical high school graduate might not know. For each, 
you provide a brief, clear, 1-sentence explanation from your own knowledge."""

    user_prompt = f"""Read the following passage. Identify every word, name, phrase, concept, 
or reference that a typical high school graduate might NOT immediately understand.

For each term, provide a concise 1-sentence explanation using your own knowledge. 
Be selective - only flag terms that genuinely need explanation. Skip common words, 
basic concepts, and terms already explained inline in the text.

Return ONLY a JSON object where keys are the exact term/phrase as it appears in the text 
and values are the 1-sentence explanation.

If there are NO difficult terms, return exactly: {{}}

Example output:
{{"utilitarianism": "A philosophy that says the best action is the one that produces the most overall happiness.", 
"John Rawls": "An American philosopher (1921-2002) who wrote influential works on justice and fairness."}}

TEXT TO ANALYZE:

{simplified_text}"""

    return system_prompt, user_prompt


def build_concept_map_prompt(full_text: str) -> tuple[str, str]:
    """
    Build a prompt to generate a concept map connecting the main ideas
    in the document.

    Args:
        full_text: The full simplified document text.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = (
        "You are an expert educator who creates clear, visual concept maps "
        "that show how ideas in a text connect to each other. You help readers "
        "see the big picture before or after reading."
    )

    # Use a representative sample if the text is very long
    max_chars = 8000
    sample = full_text[:max_chars] if len(full_text) > max_chars else full_text

    user_prompt = f"""Read the following text and create a Mermaid flowchart that connects 
all the main ideas. Show how each idea leads to, causes, supports, or contrasts 
with other ideas.

FORMAT RULES:
1. Output a valid Mermaid flowchart using ```mermaid code block syntax
2. Use graph TD (top-down direction)
3. Use short, clear labels inside nodes (max 8 words per node)
4. Label the arrows with the relationship (e.g., -->|leads to|)
5. Include 8-15 main concepts, no more
6. Use different node shapes for different types:
   - Rounded boxes for main ideas: A(Main Idea)
   - Rectangles for supporting points: B[Supporting Point]
   - Diamonds for decisions/tensions: C{{Tension}}
7. Output ONLY the mermaid code block, no introduction or commentary
8. Do NOT use special characters like parentheses or quotes inside node labels

EXAMPLE:

```mermaid
graph TD
    A(Central Thesis) -->|leads to| B[Idea A]
    A -->|also causes| C[Idea B]
    B -->|conflicts with| C
    B -->|results in| D[Consequence]
    C -->|supported by| E[Evidence]
    D -->|which means| F(Conclusion)
    E -->|challenges| F
```

TEXT:

{sample}"""

    return system_prompt, user_prompt


def build_intro_prompt(
    doc_summary: str,
    title: str = "",
    author: str = "",
) -> tuple[str, str]:
    """
    Build a prompt to generate a short reader-facing introduction.

    This creates a 2-3 sentence orientation that sets context for the reader
    and invites them into the body text.

    Args:
        doc_summary: The generated document summary.
        title:       Document title.
        author:      Author name.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = """You write brief, engaging introductions for simplified academic texts. 
Your intro should orient the reader and make them want to read on.

RULES:
1. Write exactly 2-3 sentences.
2. First sentence: What is this text about? (topic + author's main question/argument)
3. Second sentence: Why does it matter? (stakes, relevance, or what the reader will learn)
4. Optional third sentence: A hook that transitions into the body text.
5. Write in second person ("you") or impersonal style — NOT "the author argues".
6. Do NOT use academic jargon. Write at a newspaper level.
7. Do NOT use the word "book", "essay", or "paper". Always refer to it as "this document".
8. Output ONLY the introduction text, nothing else."""

    user_prompt = f"""Write a 2-3 sentence introduction for this simplified academic text.

Title: {title}
Author: {author}
Document summary: {doc_summary}

The introduction should orient a general reader and invite them to read on. 
Do NOT call it a "book", "essay", or "paper" — always say "document".
Output ONLY the introduction text."""

    return system_prompt, user_prompt
