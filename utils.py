"""
This is where we process LLM calls, JSON parsing, and free web retrieval.

Provides:
  - call_llm():          Thin wrapper around OpenAI Chat Completions API
  - safe_parse():        JSON parsing with one automatic retry via the LLM
  - search_wikipedia():  Free Wikipedia REST API search + summary retrieval
  - search_duckduckgo(): DuckDuckGo Instant Answer API (free, no key)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

logger = logging.getLogger("job_prep")


# -- OpenAI client (lazy init) ------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Return a cached OpenAI client, reading the key from environment."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Export it or add it to a .env file."
            )
        _client = OpenAI(api_key=api_key)
    return _client


# -- LLM call -----------------------------------------------------------------

def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """Call the OpenAI Chat Completions API and return the response text."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# -- Safe JSON parsing with one retry -----------------------------------------

def safe_parse(raw: str, model: str = "gpt-4.1-mini") -> Dict[str, Any]:
    """
    Parse raw LLM output as JSON. If it fails, ask the LLM to fix the
    malformed JSON once. Raises ValueError on second failure.
    """
    cleaned = raw.strip()

    # Strip markdown code fences that models sometimes add
    if cleaned.startswith("```"):
        # Remove first line (```json or ```)
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        else:
            cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed, attempting LLM repair...")
        fixed = call_llm(
            system_prompt=(
                "Fix this malformed JSON. Return ONLY valid JSON, "
                "no explanation, no markdown fences."
            ),
            user_prompt=cleaned,
            model=model,
        )
        fixed = fixed.strip()
        if fixed.startswith("```"):
            first_nl = fixed.find("\n")
            if first_nl != -1:
                fixed = fixed[first_nl + 1:]
        if fixed.endswith("```"):
            fixed = fixed[:-3]
        fixed = fixed.strip()
        return json.loads(fixed)


# -- HTTP session with proper headers -----------------------------------------

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Return a cached requests.Session with proper User-Agent."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            "User-Agent": (
                "JobPrepAgent/1.0 "
                "(ASU WiCS academic project; Python/requests)"
            ),
            "Accept": "application/json",
        })
    return _session


# -- Wikipedia search ----------------------------------------------------------

def search_wikipedia(
    keywords: List[str], max_results: int = 3
) -> List[Dict[str, str]]:
    """
    Search Wikipedia and return article summaries.

    Two-step process:
      1. Search for article titles matching keywords
      2. Fetch the summary for each matching article
    """
    session = _get_session()
    query = " ".join(keywords)
    snippets: List[Dict[str, str]] = []

    try:
        # Step 1: search for articles
        resp = session.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])

        # Step 2: fetch summaries
        for r in results:
            title = r["title"]
            try:
                summary_url = (
                    "https://en.wikipedia.org/api/rest_v1/page/summary/"
                    + title.replace(" ", "_")
                )
                summary_resp = session.get(summary_url, timeout=10)
                if summary_resp.status_code == 200:
                    data = summary_resp.json()
                    page_url = (
                        data.get("content_urls", {})
                            .get("desktop", {})
                            .get("page", f"https://en.wikipedia.org/wiki/{title}")
                    )
                    snippets.append({
                        "text": data.get("extract", "")[:2000],
                        "url": page_url,
                        "title": title,
                    })
            except Exception:
                continue

    except Exception as exc:
        logger.error("Wikipedia search failed: %s", exc)

    return snippets


# -- DuckDuckGo Instant Answer -------------------------------------------------

def search_duckduckgo(keywords: List[str]) -> List[Dict[str, str]]:
    """
    Use the DuckDuckGo Instant Answer API (free, no key needed).

    Returns abstract text and related topics for the query.
    """
    session = _get_session()
    query = " ".join(keywords)
    snippets: List[Dict[str, str]] = []

    try:
        resp = session.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # Main abstract
        if data.get("AbstractText"):
            snippets.append({
                "text": data["AbstractText"][:2000],
                "url": data.get("AbstractURL", ""),
                "title": data.get("Heading", query),
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                snippets.append({
                    "text": topic["Text"][:1000],
                    "url": topic.get("FirstURL", ""),
                    "title": topic.get("Text", "")[:80],
                })

    except Exception as exc:
        logger.error("DuckDuckGo search failed: %s", exc)

    return snippets
