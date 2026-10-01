"""
Tests for Step 2 — Normalization.
Covers: OCR char fixes, percentage skipping, comma removal, parse failures.
"""

import pytest
from pipeline.normalizer import normalize_step2, _fix_ocr_chars, _token_to_float
from models.schemas import Step1Response


class TestOCRCharFix:
    def test_lowercase_l_to_1(self):
        assert _fix_ocr_chars("l200") == "1200"

    def test_uppercase_O_to_0(self):
        assert _fix_ocr_chars("O200") == "0200"

    def test_mixed_errors(self):
        assert _fix_ocr_chars("1OO") == "100"

    def test_clean_token_unchanged(self):
        assert _fix_ocr_chars("1200") == "1200"

    def test_non_numeric_token_unchanged(self):
        # Non-numeric tokens should not be mangled
        result = _fix_ocr_chars("Total")
        assert result == "Total"


class TestTokenToFloat:
    def test_integer(self):
        assert _token_to_float("1200") == 1200.0

    def test_decimal(self):
        assert _token_to_float("1200.50") == 1200.50

    def test_with_commas(self):
        assert _token_to_float("1,200") == 1200.0


class TestNormalizeStep2:
    def _make_step1(self, tokens, confidence=0.80):
        return Step1Response(
            raw_tokens=tokens,
            currency_hint="INR",
            confidence=confidence,
            raw_text="",
        )

    def test_basic_normalization(self):
        step1 = self._make_step1(["1200", "1000", "200"])
        result = normalize_step2(step1)
        assert 1200.0 in result.normalized_amounts
        assert 1000.0 in result.normalized_amounts
        assert 200.0 in result.normalized_amounts

    def test_ocr_correction(self):
        step1 = self._make_step1(["l200", "1O00"])
        result = normalize_step2(step1)
        assert 1200.0 in result.normalized_amounts
        assert 1000.0 in result.normalized_amounts

    def test_percentage_tokens_skipped(self):
        step1 = self._make_step1(["1200", "10%"])
        result = normalize_step2(step1)
        assert 1200.0 in result.normalized_amounts
        # 10% should be skipped (not cast to float)
        assert 10.0 not in result.normalized_amounts
        assert len(result.normalized_amounts) == 1

    def test_empty_tokens(self):
        step1 = self._make_step1([])
        result = normalize_step2(step1)
        assert result.normalized_amounts == []

    def test_confidence_in_range(self):
        step1 = self._make_step1(["1200", "200"])
        result = normalize_step2(step1)
        assert 0.0 <= result.normalization_confidence <= 1.0
