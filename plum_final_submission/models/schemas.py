"""
Pydantic v2 schemas for the Amount Detection Pipeline.
Each pipeline step has clearly typed inputs and outputs.
"""

from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────

class ExtractRequest(BaseModel):
    input_type: Literal["text", "image"] = Field(
        ..., description="'text' for plain text input, 'image' for base64-encoded image"
    )
    text: Optional[str] = Field(
        None, description="Raw text from the medical bill (required if input_type='text')"
    )
    image_base64: Optional[str] = Field(
        None, description="Base64-encoded image string (required if input_type='image')"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input_type": "text",
                    "text": "Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%"
                }
            ]
        }
    }


# ─────────────────────────────────────────────
# STEP 1 — OCR / Token Extraction
# ─────────────────────────────────────────────

class Step1Response(BaseModel):
    raw_tokens: List[str] = Field(..., description="Raw numeric tokens found in the document")
    currency_hint: Optional[str] = Field(None, description="Detected currency symbol/code (e.g. 'INR', 'USD')")
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_text: Optional[str] = Field(None, description="OCR extracted text (set when input is image)")


# ─────────────────────────────────────────────
# STEP 2 — Normalization
# ─────────────────────────────────────────────

class Step2Response(BaseModel):
    normalized_amounts: List[float] = Field(..., description="Cleaned numeric values")
    normalization_confidence: float = Field(..., ge=0.0, le=1.0)


# ─────────────────────────────────────────────
# STEP 3 — Classification
# ─────────────────────────────────────────────

class ClassifiedAmount(BaseModel):
    type: str = Field(..., description="Amount category: total_bill | paid | due | discount | copay | insurance | tax | other")
    value: float
    raw_token: Optional[str] = None


class Step3Response(BaseModel):
    amounts: List[ClassifiedAmount]
    confidence: float = Field(..., ge=0.0, le=1.0)


# ─────────────────────────────────────────────
# STEP 4 — Final Output
# ─────────────────────────────────────────────

class FinalAmount(BaseModel):
    type: str
    value: float
    source: str = Field(..., description="The text snippet that produced this amount")


class FinalResponse(BaseModel):
    currency: str = Field(..., description="ISO currency code or symbol")
    amounts: List[FinalAmount]
    status: str = Field(default="ok")
    warning: Optional[str] = Field(None, description="Optional sanity check warnings")


# ─────────────────────────────────────────────
# GUARDRAIL / ERROR RESPONSES
# ─────────────────────────────────────────────

class GuardrailResponse(BaseModel):
    status: str
    reason: str
