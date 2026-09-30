"""
Gemini API client wrapper — uses the new google-genai SDK (v2+).
Supports text and vision (multimodal) calls with JSON-mode output.
"""

from __future__ import annotations
import os
import json
import logging
from functools import lru_cache

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-2.0-flash"


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    """Lazily initialise and cache the Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Add it to your .env file."
        )
    return genai.Client(api_key=api_key)


def call_gemini_text(prompt: str, expect_json: bool = True) -> str:
    """
    Send a text-only prompt to Gemini and return the response string.
    When expect_json=True, uses JSON response MIME type for deterministic output.
    """
    client = _get_client()

    config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json" if expect_json else "text/plain",
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=config,
    )
    return response.text


def call_gemini_vision(prompt: str, image_base64: str, mime_type: str = "image/jpeg") -> str:
    """
    Send an image + text prompt to Gemini Vision and return the response string.
    image_base64: base64-encoded image bytes string.
    """
    client = _get_client()

    image_part = types.Part.from_bytes(
        data=__import__("base64").b64decode(image_base64),
        mime_type=mime_type,
    )
    text_part = types.Part.from_text(text=prompt)

    config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json",
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[text_part, image_part],
        config=config,
    )
    return response.text


def safe_parse_json(raw: str) -> dict:
    """
    Safely parse a JSON string returned by Gemini.
    Strips markdown code fences if present.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini JSON response: %s\nRaw: %s", e, raw)
        raise ValueError(f"Gemini returned invalid JSON: {e}") from e
