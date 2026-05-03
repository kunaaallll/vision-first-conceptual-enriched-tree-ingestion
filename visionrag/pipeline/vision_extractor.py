"""Vision-based extraction using GPT-4o.

For each page image, sends to GPT-4o with a structured prompt to extract
formulas, derivations, tables, diagrams, graphs, and chemical equations.
Results are cached by image content hash to avoid re-processing.
"""

import asyncio
import base64
import json
import logging

from openai import AsyncOpenAI

from visionrag.utils.cache import get_cached_vision, set_cached_vision
from visionrag.config import get_settings

logger = logging.getLogger(__name__)

VISION_PROMPT = """You are an expert textbook content extractor preparing a STEM page for a retrieval system that a student will query for exam prep. Fidelity of mathematics is the top priority — every symbol must survive.

Return a JSON object with these keys:
{
  "prose_text": "<the full prose of the page as clean Unicode text, reading order left-to-right, top-to-bottom. Preserve section numbers like '5.1 INTRODUCTION' exactly. Omit page numbers, running headers/footers, and figure labels that aren't part of the sentence flow. Keep paragraph breaks as \\n\\n.>",
  "section_identifiers": ["5.1", "5.1.2"],
  "formulas": [
    {
      "name": "<human label, e.g. 'Coulomb's law', or '' if none'>",
      "latex": "<ONE self-contained LaTeX expression, WITHOUT $ delimiters>",
      "variables": [{"symbol": "F", "meaning": "force between charges"}, {"symbol": "q_1", "meaning": "first charge"}],
      "context": "<1-sentence description of when / where this formula applies>"
    }
  ],
  "derivations": [
    {
      "title": "<derivation name as printed>",
      "goal": "<what the derivation is trying to establish in 1 sentence>",
      "assumptions": ["assumption 1", "assumption 2"],
      "steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
      "result": "<final result in LaTeX, without $>"
    }
  ],
  "tables": [{"caption": "...", "headers": ["col1","col2"], "rows": [["...","..."]]}],
  "diagrams": [{"caption": "...", "concepts": ["..."], "description": "<conceptual meaning>"}],
  "graphs": [{"title": "...", "x_axis": "...", "y_axis": "...", "meaning": "..."}],
  "chemical_equations": [{"equation": "2H_2 + O_2 -> 2H_2O", "description": "..."}],
  "topics_detected": ["<section/topic title on this page>"],
  "key_terms": ["terms an exam question would use on this page"]
}

STRICT rules — the pipeline WILL drop formulas that violate these:
- prose_text: this is the CANONICAL text of the page. We trust YOU over the upstream OCR. Transcribe word-for-word from the image, including every section identifier (e.g. '5.1 INTRODUCTION') at the head of its paragraph. If a word is clearly a hyphenated line break, rejoin it. NEVER invent text that is not visibly on the page.
- section_identifiers: list every dotted number you see as a heading (e.g. "5.1", "5.2.3"). Empty list if the page has none.
- Formulas: write pure LaTeX WITHOUT surrounding $ / $$. Example: `F = k \\frac{q_1 q_2}{r^2}`.
- ALWAYS fill `variables` — every symbol that appears in the formula must have a one-line meaning. If you cannot explain a symbol from the page, OMIT the formula rather than guess.
- Do not merge two formulas into one string. One object per formula.
- Inline numerical results (e.g. `g = 9.8 m/s^2`) count as formulas if they are labelled.
- Derivations: keep steps in textbook order. Never skip an algebraic step that appears on the page.
- Tables: preserve header text verbatim, even units and symbols.
- Chemical equations: use `->` for the arrow, preserve subscripts/superscripts as `_` / `^`.
- key_terms: 5–15 exam-relevant terms literally present on the page. Do not invent.
- Empty categories → `[]`. Missing key is an error.
- Output ONLY the JSON object. No prose, no markdown fences, no comments."""


async def extract_single_page_vision(
    page_num: int,
    image_path: str,
    client: AsyncOpenAI,
    model: str | None = None,
) -> dict:
    """Extract structured content from a single page image via vision model."""
    cached = await get_cached_vision(image_path)
    if cached is not None:
        logger.info(f"Page {page_num}: vision cache hit")
        return cached

    settings = get_settings()
    vision_model = model or settings.vision_model

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    response = await client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=4096,
    )

    raw = response.choices[0].message.content
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Page {page_num}: vision model returned invalid JSON")
        result = _empty_vision_result()

    result = _validate_vision_result(result)
    await set_cached_vision(image_path, result)
    logger.info(f"Page {page_num}: vision extraction complete")
    return result


async def extract_all_pages_vision(
    image_paths: list[str],
    client: AsyncOpenAI,
    model: str | None = None,
    concurrency: int | None = None,
) -> dict[int, dict]:
    """Extract vision data for all pages with controlled concurrency.

    Returns dict mapping page_number (1-indexed) to vision extraction result.
    """
    settings = get_settings()
    max_concurrent = concurrency or settings.vision_concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _extract_with_limit(page_num: int, path: str) -> tuple[int, dict]:
        async with semaphore:
            result = await extract_single_page_vision(page_num, path, client, model)
            return page_num, result

    tasks = [
        _extract_with_limit(i + 1, path)
        for i, path in enumerate(image_paths)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    vision_data = {}
    for item in results:
        if isinstance(item, Exception):
            logger.error(f"Vision extraction failed: {item}")
            continue
        page_num, data = item
        vision_data[page_num] = data

    return vision_data


def _empty_vision_result() -> dict:
    return {
        "prose_text": "",
        "section_identifiers": [],
        "formulas": [],
        "derivations": [],
        "tables": [],
        "diagrams": [],
        "graphs": [],
        "chemical_equations": [],
        "topics_detected": [],
        "key_terms": [],
    }


def _validate_vision_result(data: dict) -> dict:
    """Ensure all expected keys exist with correct types."""
    template = _empty_vision_result()
    for key, default in template.items():
        if key not in data:
            data[key] = default
            continue
        if isinstance(default, list) and not isinstance(data[key], list):
            data[key] = default
        elif isinstance(default, str) and not isinstance(data[key], str):
            data[key] = default
    return data
