"""
LLM Generator
=============
Calls Groq (Llama-3) with retrieved context.
Produces answers with inline citations referencing source chunks.
"""

import os
import json
from groq import Groq
from typing import Iterator

# Using Llama 3 70B on Groq for high-quality reasoning
GENERATION_MODEL = "llama-3.3-70b-versatile"
MAX_OUTPUT_TOKENS = 1000


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        heading = chunk.get("section_heading") or "—"
        lines.append(
            f"[{i}] Document: {chunk['doc_name']} | "
            f"Page: {chunk['page_number']} | "
            f"Section: {heading}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(lines)


SYSTEM_PROMPT = """You are a precise research assistant. Answer questions using ONLY the provided context chunks.

Rules:
1. After every factual claim, add an inline citation in this exact format:
   [Source: <doc_name>, p.<page_number>, §<section_heading>]
   Example: The model uses scaled dot-product attention [Source: Attention_is_All_You_Need, p.3, §3.2 Attention].

2. If multiple chunks support a claim, cite all of them.

3. If the answer is not found in the context, say:
   "I could not find information about this in the provided documents."
   Do NOT make up information.

4. Structure your answer clearly. Use bullet points or numbered steps for complex answers.

5. Start your answer directly — do not say "Based on the context..." or similar preamble.
"""


def _get_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


def generate_answer(
    query: str,
    chunks: list[dict],
    stream: bool = False,
) -> str | Iterator[str]:
    """
    Generate an answer grounded in the retrieved chunks using Groq.
    """
    client = _get_groq_client()
    context_block = _build_context_block(chunks)

    user_message = f"""Context:
{context_block}

Question: {query}"""

    if stream:
        def _stream_gen() -> Iterator[str]:
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                max_tokens=MAX_OUTPUT_TOKENS,
                stream=True,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return _stream_gen()

    else:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        return response.choices[0].message.content


def generate_answer_streamed_sse(
    query: str,
    chunks: list[dict],
) -> Iterator[str]:
    """
    Yields Server-Sent Events (SSE) formatted strings for streaming to frontend.
    """
    client = _get_groq_client()
    context_block = _build_context_block(chunks)

    user_message = f"""Context:
{context_block}

Question: {query}"""

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=MAX_OUTPUT_TOKENS,
        stream=True,
    )

    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield f"data: {json.dumps({'type': 'delta', 'text': delta})}\n\n"

    # Send sources as final event
    sources = [
        {
            "doc_name": c.get("doc_name", ""),
            "page_number": c.get("page_number", ""),
            "section_heading": c.get("section_heading", ""),
            "text_snippet": c.get("text", "")[:200] + "...",
            "rrf_score": c.get("rrf_score", 0),
            "rank": c.get("rank", 0),
        }
        for c in chunks
    ]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
    yield "data: [DONE]\n\n"
