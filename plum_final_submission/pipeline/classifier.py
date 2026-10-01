"""
Step 3 — Context Classification
Uses Gemini to read surrounding text context and assign semantic labels
to each normalized amount:
  total_bill | paid | due | discount | copay | insurance | tax | other
"""

from __future__ import annotations
import logging
from typing import List

from models.schemas import Step1Response, Step2Response, Step3Response, ClassifiedAmount
from utils.gemini_client import call_gemini_text, safe_parse_json

logger = logging.getLogger(__name__)

# ── Classification Prompt ─────────────────────────────────────────────────────
_CLASSIFY_PROMPT_TEMPLATE = """
You are a financial document analyst specialised in medical bills and receipts.

Given the raw text of a medical bill and a list of numeric amounts extracted from it,
classify each amount into one of these categories:
  - total_bill   : the total/gross amount charged
  - paid         : amount already paid
  - due          : amount still owed / balance due
  - discount     : reduction applied (absolute value, not %)
  - copay        : copayment amount
  - insurance    : insurance cover amount
  - tax          : tax charged
  - other        : cannot be determined

RAW TEXT FROM DOCUMENT:
---
{raw_text}
---

NORMALIZED AMOUNTS TO CLASSIFY: {amounts}

Rules:
1. Map each value to a category based on its surrounding context in the text.
2. Every amount must appear exactly once in the output.
3. If an amount appears multiple times in the text, use the context of its first occurrence.
4. Return ONLY valid JSON in this exact structure:
{{
  "amounts": [
    {{"type": "total_bill", "value": 1200, "raw_token": "1200"}},
    {{"type": "paid",       "value": 1000, "raw_token": "1000"}},
    {{"type": "due",        "value": 200,  "raw_token": "200"}}
  ],
  "confidence": 0.88
}}
"""


def classify_step3(
    step1: Step1Response,
    step2: Step2Response,
) -> Step3Response:
    """
    Step 3: Use Gemini to classify each normalized amount by context.

    Args:
        step1: Step 1 output (provides raw text and raw_tokens for context)
        step2: Step 2 output (provides normalized_amounts to classify)

    Returns:
        Step3Response with labelled amounts and confidence
    """
    if not step2.normalized_amounts:
        logger.warning("[Step3] No amounts to classify — returning empty.")
        return Step3Response(amounts=[], confidence=0.0)

    raw_text = step1.raw_text or ""
    amounts_list = step2.normalized_amounts

    prompt = _CLASSIFY_PROMPT_TEMPLATE.format(
        raw_text=raw_text,
        amounts=amounts_list,
    )

    logger.info("[Step3] Sending %d amounts to Gemini for classification...", len(amounts_list))
    raw_response = call_gemini_text(prompt, expect_json=True)
    data = safe_parse_json(raw_response)

    classified: List[ClassifiedAmount] = []
    for item in data.get("amounts", []):
        classified.append(ClassifiedAmount(
            type=item.get("type", "other"),
            value=float(item.get("value", 0)),
            raw_token=item.get("raw_token"),
        ))

    confidence = float(data.get("confidence", 0.5))
    logger.info("[Step3] Classified %d amounts. Confidence: %.2f", len(classified), confidence)

    return Step3Response(amounts=classified, confidence=round(confidence, 2))
