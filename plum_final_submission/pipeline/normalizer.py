"""
Step 2 — Numeric Normalization
Fixes OCR character-substitution errors common in scanned documents:
  - 'l' or 'I' → '1' (in numeric context)
  - 'O' → '0'
  - 'S' → '5'
  - 'B' → '8'
  - thousand-separator commas removed
  - Percentage tokens kept separate (not cast to float)
"""

from __future__ import annotations
import re
import logging
from typing import List

from models.schemas import Step1Response, Step2Response

logger = logging.getLogger(__name__)

# ── OCR Character Correction Table ────────────────────────────────────────────
# Applied only to what looks like a digit position context
_OCR_CHAR_MAP = {
    "l": "1",
    "I": "1",
    "O": "0",
    "o": "0",
    "S": "5",
    "s": "5",
    "B": "8",
    "G": "6",
    "Z": "2",
    "z": "2",
    "q": "9",
    "g": "9",
}

_DIGIT_CONTEXT_RE = re.compile(r"^[\d\.,lIOoSsBGZzqg]+%?$")
_PERCENTAGE_RE = re.compile(r"^[\d\.,]+%$")


def _fix_ocr_chars(token: str) -> str:
    """
    Replace OCR-confused characters with their correct digit equivalents.
    Only operates on tokens that look numeric (possibly with OCR noise).
    """
    if not _DIGIT_CONTEXT_RE.match(token):
        logger.debug("[Normalizer] Skipping non-numeric token: %s", token)
        return token

    corrected = []
    for char in token:
        corrected.append(_OCR_CHAR_MAP.get(char, char))
    return "".join(corrected)


def _token_to_float(token: str) -> float:
    """
    Convert a cleaned token string to float.
    Handles: '1200', '1,200', '1200.50'
    """
    clean = token.replace(",", "").strip()
    return float(clean)


def normalize_step2(step1: Step1Response) -> Step2Response:
    """
    Step 2: Normalize raw tokens into clean float values.

    Strategy:
    1. Fix OCR character substitutions
    2. Remove thousand-separator commas
    3. Cast to float (skip percentage tokens)
    4. Compute normalization confidence

    Args:
        step1: Output from Step 1 (contains raw_tokens)

    Returns:
        Step2Response with normalized_amounts and normalization_confidence
    """
    normalized: List[float] = []
    total = len(step1.raw_tokens)
    success_count = 0

    for token in step1.raw_tokens:
        # Skip percentage tokens — they get classified separately
        if _PERCENTAGE_RE.match(token):
            logger.debug("[Normalizer] Skipping percentage token: %s", token)
            continue

        fixed = _fix_ocr_chars(token)
        try:
            value = _token_to_float(fixed)
            normalized.append(value)
            success_count += 1
            if fixed != token:
                logger.info("[Normalizer] Corrected '%s' → '%s' (%.0f)", token, fixed, value)
        except ValueError:
            logger.warning("[Normalizer] Could not parse token as number: '%s' (fixed: '%s')", token, fixed)

    # Confidence: ratio of successfully parsed tokens × step1 confidence dampener
    parse_ratio = success_count / max(total, 1)
    normalization_confidence = round(parse_ratio * step1.confidence * 1.1, 2)
    normalization_confidence = min(normalization_confidence, 0.99)

    logger.info(
        "[Step2] Normalized %d/%d tokens. Confidence: %.2f",
        success_count, total, normalization_confidence
    )

    return Step2Response(
        normalized_amounts=normalized,
        normalization_confidence=normalization_confidence,
    )
