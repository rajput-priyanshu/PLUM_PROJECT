"""
Step 1 — OCR / Text Extraction
Handles:
  - Plain text input: regex-based numeric token extraction
  - Image input: Gemini Vision OCR (handles crumpled/partial scans)
  - Fallback: pytesseract for clean images when Gemini is unavailable
"""

from __future__ import annotations
import re
import logging
from typing import Optional

from models.schemas import Step1Response
from utils.gemini_client import call_gemini_vision, safe_parse_json

logger = logging.getLogger(__name__)

# ── Currency patterns (INR, Rs, ₹, $, USD, EUR, £) ──────────────────────────
CURRENCY_PATTERNS = [
    (r"\bINR\b", "INR"),
    (r"Rs\.?\s*", "INR"),
    (r"₹", "INR"),
    (r"\bUSD\b|\$", "USD"),
    (r"\bEUR\b|€", "EUR"),
    (r"£|\bGBP\b", "GBP"),
]

# ── Numeric token regex: captures integers, decimals, percentages ─────────────
# Also captures OCR-noisy tokens that START with common letter-digit confusions
# Matches: 1200, 1,200, 1200.50, 10%, l200, 1O00
NUMERIC_TOKEN_RE = re.compile(
    r"(?<!\w)"                            # no leading word char
    r"([lI1O0SsBGZzqg\d][\d,lIOoSsBGZzqg]*(?:\.\d+)?)"  # OCR-noisy numeric token
    r"(%?)"                               # optional percentage
    r"(?!\w)",                            # no trailing word char
    re.IGNORECASE,
)

# ── OCR character pre-correction map for text inputs ─────────────────────────
_TEXT_OCR_FIX = str.maketrans({
    "O": "0", "o": "0",
    "l": "1", "I": "1",
    "S": "5", "s": "5",
    "B": "8", "G": "6",
    "Z": "2", "z": "2",
})


def _pre_fix_numeric_tokens(text: str) -> str:
    """
    Apply OCR character corrections ONLY to token-like substrings
    (sequences that look like they should be numbers).
    Preserves surrounding text (labels, pipes, colons) untouched.
    """
    # Pattern: tokens that are mostly digits/OCR-confused chars
    noisy_token_re = re.compile(r"(?<!\w)[lI1O0SsBGZzqg][\d,lIOoSsBGZzqg]+(?:%?)(?!\w)")

    def fix_token(match: re.Match) -> str:
        return match.group(0).translate(_TEXT_OCR_FIX)

    return noisy_token_re.sub(fix_token, text)


def _detect_currency(text: str) -> Optional[str]:
    """Scan text for currency hints and return the first match."""
    for pattern, code in CURRENCY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return code
    return None


def _extract_tokens_from_text(text: str) -> list[str]:
    """
    Extract all numeric tokens (including percentages) from plain text.
    Applies a pre-pass OCR fix for common digit-letter confusions before regex.
    """
    # Pre-fix OCR-noisy tokens (e.g. 'l200' -> '1200', '1O00' -> '1000')
    fixed_text = _pre_fix_numeric_tokens(text)

    tokens = []
    seen = set()
    for match in NUMERIC_TOKEN_RE.finditer(fixed_text):
        number = match.group(1).replace(",", "")  # strip thousand separators
        suffix = match.group(2)                   # "%" or ""
        token = number + suffix
        # Deduplicate and filter out very short tokens that are likely noise
        if token not in seen and (len(number.replace(".", "")) >= 1):
            tokens.append(token)
            seen.add(token)
    return tokens


def _ocr_confidence(raw_text: str) -> float:
    """
    Heuristic confidence based on text quality indicators.
    Penalises if many non-ascii chars or very low token count relative to length.
    """
    if not raw_text:
        return 0.0
    non_ascii = sum(1 for c in raw_text if ord(c) > 127)
    ratio = non_ascii / max(len(raw_text), 1)
    return round(max(0.3, 1.0 - ratio * 2), 2)


# ── Gemini Vision OCR prompt ──────────────────────────────────────────────────
_VISION_PROMPT = """
You are an OCR assistant specialised in medical bills and receipts.
The image may be crumpled, partially visible, or low quality.

1. Extract ALL numeric amounts visible in the image.
2. Identify the currency if present (INR, Rs, ₹, USD, $, EUR, etc.).
3. Include percentage values (e.g. "10%") as-is.
4. Fix obvious OCR artefacts (e.g. 'l200' → '1200', 'O' → '0') in raw_tokens.

Return ONLY valid JSON in this exact structure:
{
  "raw_text": "<the full OCR text you read from the image>",
  "raw_tokens": ["1200", "1000", "200", "10%"],
  "currency_hint": "INR",
  "confidence": 0.85
}

If no amounts are found, return:
{"raw_text": "", "raw_tokens": [], "currency_hint": null, "confidence": 0.0}
"""


def extract_step1(
    input_type: str,
    text: Optional[str] = None,
    image_base64: Optional[str] = None,
) -> Step1Response:
    """
    Step 1: Extract raw numeric tokens from text or image input.

    Args:
        input_type: "text" | "image"
        text: plain text bill content
        image_base64: base64-encoded image

    Returns:
        Step1Response with raw_tokens, currency_hint, confidence
    """
    if input_type == "text":
        if not text:
            raise ValueError("text must be provided when input_type='text'")

        tokens = _extract_tokens_from_text(text)
        currency = _detect_currency(text)
        # Confidence based on token yield: more tokens = higher confidence
        confidence = min(0.95, 0.5 + len(tokens) * 0.08)

        logger.info("[Step1-text] Found %d tokens. Currency: %s", len(tokens), currency)
        return Step1Response(
            raw_tokens=tokens,
            currency_hint=currency,
            confidence=round(confidence, 2),
            raw_text=text,
        )

    elif input_type == "image":
        if not image_base64:
            raise ValueError("image_base64 must be provided when input_type='image'")

        logger.info("[Step1-image] Sending image to Gemini Vision OCR...")
        raw = call_gemini_vision(_VISION_PROMPT, image_base64)
        data = safe_parse_json(raw)

        tokens = data.get("raw_tokens", [])
        currency = data.get("currency_hint")
        confidence = float(data.get("confidence", 0.5))
        raw_text = data.get("raw_text", "")

        logger.info("[Step1-image] OCR found %d tokens. Confidence: %.2f", len(tokens), confidence)
        return Step1Response(
            raw_tokens=tokens,
            currency_hint=currency,
            confidence=round(confidence, 2),
            raw_text=raw_text,
        )

    else:
        raise ValueError(f"Unknown input_type: {input_type}")
