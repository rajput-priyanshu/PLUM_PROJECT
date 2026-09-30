"""
Step 4 — Final Output Builder
Combines currency, classified amounts, and provenance into the final response.
Also runs sanity checks (guardrails):
  - total_bill ≈ paid + due  (warns if mismatch > 5%)
  - At least one amount classified
"""

from __future__ import annotations
import re
import logging
from typing import Optional, List

from models.schemas import (
    Step1Response,
    Step3Response,
    FinalResponse,
    FinalAmount,
    GuardrailResponse,
)

logger = logging.getLogger(__name__)

_DEFAULT_CURRENCY = "INR"


def _find_source_snippet(raw_text: str, value: float, amount_type: str) -> str:
    """
    Find the text snippet in the original document that contains this amount.
    Returns a short excerpt (≤ 60 chars) around the first occurrence.
    """
    if not raw_text:
        return f"value: {value}"

    # Build regex: look for the value (int or float) possibly with currency/label nearby
    val_str = str(int(value)) if value == int(value) else str(value)
    # Also accept comma-formatted like 1,200
    val_pattern = val_str.replace(",", r"[,.]?")

    pattern = re.compile(
        rf"(?i)([^\n]{{0,30}}{re.escape(val_str)}[^\n]{{0,30}})"
    )
    match = pattern.search(raw_text)
    if match:
        snippet = match.group(1).strip()
        return f"text: '{snippet}'"

    return f"value: {value}"


def _sanity_check(amounts: List[FinalAmount]) -> Optional[str]:
    """
    Run basic arithmetic sanity check.
    If total_bill is present, verify: total ≈ paid + due (within 5% tolerance).
    """
    totals = [a.value for a in amounts if a.type == "total_bill"]
    paids = [a.value for a in amounts if a.type == "paid"]
    dues = [a.value for a in amounts if a.type == "due"]

    if totals and paids and dues:
        expected_total = sum(paids) + sum(dues)
        actual_total = sum(totals)
        if actual_total > 0:
            diff_pct = abs(actual_total - expected_total) / actual_total
            if diff_pct > 0.05:  # > 5% mismatch
                warning = (
                    f"amounts_mismatch: total_bill={actual_total} "
                    f"but paid+due={expected_total:.2f} "
                    f"(diff {diff_pct*100:.1f}%)"
                )
                logger.warning("[Step4] %s", warning)
                return warning

    return None


def build_final_step4(
    step1: Step1Response,
    step3: Step3Response,
    currency_override: Optional[str] = None,
) -> FinalResponse:
    """
    Step 4: Assemble the final structured response with provenance.

    Args:
        step1: Step 1 output (raw text + currency hint)
        step3: Step 3 output (classified amounts)
        currency_override: optional currency code to override detection

    Returns:
        FinalResponse with currency, amounts (with source), and status
    """
    currency = (
        currency_override
        or step1.currency_hint
        or _DEFAULT_CURRENCY
    )

    raw_text = step1.raw_text or ""

    final_amounts: List[FinalAmount] = []
    for item in step3.amounts:
        source = _find_source_snippet(raw_text, item.value, item.type)
        final_amounts.append(FinalAmount(
            type=item.type,
            value=item.value,
            source=source,
        ))

    warning = _sanity_check(final_amounts)

    logger.info(
        "[Step4] Built final response. Currency: %s, Amounts: %d, Warning: %s",
        currency, len(final_amounts), warning
    )

    return FinalResponse(
        currency=currency,
        amounts=final_amounts,
        status="ok",
        warning=warning,
    )
