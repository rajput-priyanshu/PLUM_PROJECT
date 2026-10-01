"""
Tests for Step 1 — OCR / Text Extraction.
Covers: token detection, currency detection, percentage handling, guardrail.
"""

import pytest
from pipeline.ocr_extractor import extract_step1, _detect_currency, _extract_tokens_from_text


class TestCurrencyDetection:
    def test_inr_code(self):
        assert _detect_currency("Total: INR 1200") == "INR"

    def test_rs_symbol(self):
        assert _detect_currency("Rs. 500 paid") == "INR"

    def test_rupee_unicode(self):
        assert _detect_currency("₹ 2500") == "INR"

    def test_usd(self):
        assert _detect_currency("Amount: $150.00") == "USD"

    def test_no_currency(self):
        assert _detect_currency("Total 999 balance 100") is None


class TestTokenExtraction:
    def test_basic_amounts(self):
        tokens = _extract_tokens_from_text("Total: INR 1200 | Paid: 1000 | Due: 200")
        assert "1200" in tokens
        assert "1000" in tokens
        assert "200" in tokens

    def test_percentage_included(self):
        tokens = _extract_tokens_from_text("Discount: 10% applied")
        assert "10%" in tokens

    def test_decimal_amounts(self):
        tokens = _extract_tokens_from_text("Amount: 1200.50")
        assert "1200.50" in tokens

    def test_comma_separated(self):
        tokens = _extract_tokens_from_text("Total: 1,200")
        assert "1200" in tokens

    def test_empty_text(self):
        tokens = _extract_tokens_from_text("")
        assert tokens == []


class TestStep1TextInput:
    def test_standard_bill_text(self):
        result = extract_step1(
            input_type="text",
            text="Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%"
        )
        assert "1200" in result.raw_tokens
        assert "1000" in result.raw_tokens
        assert "200" in result.raw_tokens
        assert "10%" in result.raw_tokens
        assert result.currency_hint == "INR"
        assert result.confidence > 0.0

    def test_raises_on_missing_text(self):
        with pytest.raises(ValueError, match="text must be provided"):
            extract_step1(input_type="text", text=None)

    def test_raises_on_bad_input_type(self):
        with pytest.raises(ValueError, match="Unknown input_type"):
            extract_step1(input_type="pdf", text="some text")
