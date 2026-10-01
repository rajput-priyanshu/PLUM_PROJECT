"""
Tests for Step 4 — Builder (sanity checks, provenance, currency fallback).
Does NOT require Gemini API key.
"""

import pytest
from pipeline.builder import build_final_step4, _find_source_snippet, _sanity_check
from models.schemas import Step1Response, Step3Response, ClassifiedAmount, FinalAmount


def _make_step1(raw_text="Total: INR 1200 | Paid: 1000 | Due: 200", currency="INR"):
    return Step1Response(
        raw_tokens=["1200", "1000", "200"],
        currency_hint=currency,
        confidence=0.80,
        raw_text=raw_text,
    )


def _make_step3(amounts):
    return Step3Response(
        amounts=[ClassifiedAmount(**a) for a in amounts],
        confidence=0.85,
    )


class TestSourceSnippet:
    def test_finds_total(self):
        text = "Total: INR 1200 | Paid: 1000 | Due: 200"
        snippet = _find_source_snippet(text, 1200, "total_bill")
        assert "1200" in snippet

    def test_empty_text_fallback(self):
        snippet = _find_source_snippet("", 500, "paid")
        assert "500" in snippet


class TestSanityCheck:
    def test_matching_amounts_no_warning(self):
        amounts = [
            FinalAmount(type="total_bill", value=1200, source=""),
            FinalAmount(type="paid",       value=1000, source=""),
            FinalAmount(type="due",        value=200,  source=""),
        ]
        assert _sanity_check(amounts) is None

    def test_mismatch_produces_warning(self):
        amounts = [
            FinalAmount(type="total_bill", value=1200, source=""),
            FinalAmount(type="paid",       value=500,  source=""),
            FinalAmount(type="due",        value=200,  source=""),
        ]
        warning = _sanity_check(amounts)
        assert warning is not None
        assert "mismatch" in warning

    def test_no_warning_when_types_incomplete(self):
        # Only total + paid (no due) → no check triggered
        amounts = [
            FinalAmount(type="total_bill", value=1200, source=""),
            FinalAmount(type="paid",       value=1000, source=""),
        ]
        assert _sanity_check(amounts) is None


class TestBuildFinal:
    def test_currency_from_step1(self):
        step1 = _make_step1(currency="INR")
        step3 = _make_step3([
            {"type": "total_bill", "value": 1200},
            {"type": "paid",       "value": 1000},
            {"type": "due",        "value": 200},
        ])
        result = build_final_step4(step1, step3)
        assert result.currency == "INR"
        assert result.status == "ok"
        assert len(result.amounts) == 3

    def test_currency_fallback_to_default(self):
        step1 = _make_step1(currency=None)
        step3 = _make_step3([{"type": "total_bill", "value": 500}])
        result = build_final_step4(step1, step3)
        assert result.currency == "INR"  # Default fallback

    def test_currency_override(self):
        step1 = _make_step1(currency="INR")
        step3 = _make_step3([{"type": "paid", "value": 100}])
        result = build_final_step4(step1, step3, currency_override="USD")
        assert result.currency == "USD"

    def test_provenance_attached(self):
        step1 = _make_step1()
        step3 = _make_step3([{"type": "total_bill", "value": 1200}])
        result = build_final_step4(step1, step3)
        assert result.amounts[0].source != ""
        assert "1200" in result.amounts[0].source
