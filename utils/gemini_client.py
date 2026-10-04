from __future__ import annotations
import os
import json
import time
import logging
from functools import lru_cache

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-3.8-flash"
_FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-flash-latest"]


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


def _generate_with_retry(client, model: str, contents, config, retries: int = 3) -> str:
    """Call Gemini with automatic retry + model fallback.
    
    - 503 UNAVAILABLE → retry same model with backoff
    - 404 NOT_FOUND   → model unavailable for this key, skip to next fallback
    - Other errors    → bubble up immediately
    """
    models_to_try = [model] + _FALLBACK_MODELS
    for attempt, m in enumerate(models_to_try):
        for i in range(retries):
            try:
                response = client.models.generate_content(
                    model=m, contents=contents, config=config
                )
                if attempt > 0 or i > 0:
                    logger.info("Succeeded with model=%s on attempt %d", m, i + 1)
                return response.text
            except Exception as e:
                err = str(e)
                if "404" in err or "NOT_FOUND" in err:
                    # Model not available for this API key — skip to next fallback immediately
                    logger.warning("Model %s not available for this key, trying next fallback…", m)
                    break  # break inner loop → try next model
                elif "503" in err or "UNAVAILABLE" in err:
                    wait = 2 ** i  # 1s, 2s, 4s
                    logger.warning("Model %s overloaded (attempt %d/%d). Retrying in %ds…", m, i + 1, retries, wait)
                    time.sleep(wait)
                else:
                    raise  # non-503/404 errors bubble up immediately
    raise RuntimeError("All Gemini models are currently unavailable. Please try again in a few minutes.")


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
    return _generate_with_retry(client, MODEL_ID, prompt, config)



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

    return _generate_with_retry(client, MODEL_ID, [text_part, image_part], config)



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
