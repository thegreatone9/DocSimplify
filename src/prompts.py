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
    # Build structural note if numbered items exist
    structure_note = ""
    if numbered_items:
        structure_note = f"\nSTRUCTURE NOTE: This passage has {len(numbered_items)} numbered item(s): {', '.join(numbered_items)}. Keep exactly these numbers in the same order.\n"

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
{structure_note}
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

    system_prompt = f"""You rewrite dense academic text in plain English. Keep ALL information - 
rewrite, don't summarize. Match the original length.

TARGET AUDIENCE: A high school graduate - intelligent and curious, but not specialized.

CRITICAL RULES:
1. Never write about the text - rewrite IT. Never start with "The passage argues", 
"The author states", etc. Write as if YOU are the author making the same points.
2. Replace complex jargon with everyday equivalents.
3. Mix short and long sentences for natural rhythm. Not every sentence should be the same length.
4. Keep the author's logical connectives that carry argument weight - words like "as distinct 
from this", "to be sure", "indeed", "in retrospect". Don't flatten them all to "but" or "also".
5. When the original explains WHY something happens, keep the full causal chain. Do not 
simplify "X happened because of A, B, and C" into just "X happened because of A".
6. NEVER repeat information. Each sentence must add something new.

EXAMPLE:
Original: "The neoliberal paradigm, predicated on fiscal austerity, has exacerbated 
extant inequalities."
Rewritten: "The economic approach based on cutting government spending has made existing 
inequalities worse."

BOOK CONTEXT: {book_summary[:500]}

GLOSSARY (Use these definitions for key terms):
{glossary_text if glossary_text else "(none)"}"""

    context_line = ""
    if previous_context:
        context_line = f"\nCONTEXT (What came right before this): {previous_context}\n"

    word_count = len(paragraph_text.split())

    user_prompt = f"""Rewrite this paragraph in plain English (~{word_count} words). 
Target a high school reading level. Use simple vocabulary but vary your sentence lengths. 
Keep all facts and the full reasoning chain. Do NOT repeat any point twice. 
Do NOT summarize or comment on the text - rewrite it directly. 
Output ONLY the rewritten text.
{context_line}
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

    user_prompt = f"""Polish the following text for consistent tone and smooth transitions. 
Do NOT add, remove, or change any facts. Keep the same length (~{word_count} words). 
Keep EXACTLY {para_count} paragraphs - do NOT split any paragraph into multiple ones. 
Output ONLY the polished text.

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
    Build prompts to generate a book-level summary from chapter introductions.

    Args:
        chapter_intros: Concatenated introductory text from each chapter.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system_prompt = """You are a skilled book analyst. You read introductory passages from 
each chapter of a book and produce a concise, accurate overview of the book's topic, 
purpose, and main themes."""

    user_prompt = f"""Below are the opening passages from each chapter of a book. Based on 
these excerpts, write a concise overview (about 300-500 words) that describes:

1. What this book is about (main topic/subject)
2. Who the intended audience seems to be
3. The main themes or arguments the book explores
4. The overall structure or progression of ideas

Write in clear, plain English. This summary will be used to provide context when 
simplifying individual sections of the book.

Output ONLY the summary, no other text.

CHAPTER OPENINGS:
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
