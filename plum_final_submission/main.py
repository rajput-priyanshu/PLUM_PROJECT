"""
PS4 — AI-Powered Amount Detection in Medical Documents
FastAPI backend service.

Endpoints:
  GET  /health           → liveness check
  POST /extract-amounts  → full pipeline (text or image input)
  POST /step/1/ocr       → Step 1 only (debug)
  POST /step/2/normalize → Steps 1–2 only (debug)
  POST /step/3/classify  → Steps 1–3 only (debug)
"""

from __future__ import annotations
import logging
import os
import base64
from typing import Union

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

from models.schemas import (
    ExtractRequest,
    Step1Response,
    Step2Response,
    Step3Response,
    FinalResponse,
    GuardrailResponse,
)
from pipeline.ocr_extractor import extract_step1
from pipeline.normalizer import normalize_step2
from pipeline.classifier import classify_step3
from pipeline.builder import build_final_step4
from utils.image_utils import bytes_to_base64, preprocess_image

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Guardrail thresholds ──────────────────────────────────────────────────────
MIN_OCR_CONFIDENCE = 0.30   # Below this → reject as too noisy
MIN_TOKENS_REQUIRED = 1     # At least 1 amount needed to proceed

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PS4 — AI Amount Detection in Medical Documents",
    description=(
        "Extracts and classifies financial amounts from medical bills and receipts. "
        "Supports both typed text and scanned/crumpled image inputs via Gemini Vision OCR."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health():
    """Liveness check."""
    return {"status": "ok", "service": "PS4 Amount Detection"}


# ════════════════════════════════════════════════════════════════════════════
# MAIN ENDPOINT
# ════════════════════════════════════════════════════════════════════════════

@app.post(
    "/extract-amounts",
    response_model=Union[FinalResponse, GuardrailResponse],
    tags=["Pipeline"],
    summary="Extract and classify financial amounts from a medical bill/receipt",
)
async def extract_amounts(request: ExtractRequest):
    """
    Full 4-step pipeline:
    1. OCR / text extraction  → raw tokens + currency hint
    2. Normalization          → clean numeric values
    3. Context classification → label each amount (total, paid, due…)
    4. Final builder          → add provenance + sanity check

    **Input types:**
    - `input_type: "text"` with `text` field
    - `input_type: "image"` with `image_base64` field (base64-encoded image)
    """
    logger.info("=== /extract-amounts request [input_type=%s] ===", request.input_type)

    # ── Validation ────────────────────────────────────────────────────────────
    if request.input_type == "text" and not request.text:
        raise HTTPException(status_code=422, detail="'text' field is required when input_type='text'")
    if request.input_type == "image" and not request.image_base64:
        raise HTTPException(status_code=422, detail="'image_base64' field is required when input_type='image'")

    # ── Step 1: OCR / Extraction ──────────────────────────────────────────────
    try:
        step1 = extract_step1(
            input_type=request.input_type,
            text=request.text,
            image_base64=request.image_base64,
        )
    except Exception as e:
        logger.error("[Step1] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"OCR extraction failed: {str(e)}")

    # Guardrail: too noisy / no tokens found
    if len(step1.raw_tokens) < MIN_TOKENS_REQUIRED or step1.confidence < MIN_OCR_CONFIDENCE:
        logger.warning("[Guardrail] No amounts found or confidence too low. Tokens: %s, Conf: %.2f",
                        step1.raw_tokens, step1.confidence)
        return JSONResponse(
            status_code=200,
            content={"status": "no_amounts_found", "reason": "document too noisy"},
        )

    # ── Step 2: Normalization ─────────────────────────────────────────────────
    try:
        step2 = normalize_step2(step1)
    except Exception as e:
        logger.error("[Step2] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Normalization failed: {str(e)}")

    if not step2.normalized_amounts:
        return JSONResponse(
            status_code=200,
            content={"status": "no_amounts_found", "reason": "all tokens failed normalization"},
        )

    # ── Step 3: Classification ────────────────────────────────────────────────
    try:
        step3 = classify_step3(step1, step2)
    except Exception as e:
        logger.error("[Step3] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

    # ── Step 4: Final Builder ─────────────────────────────────────────────────
    try:
        final = build_final_step4(step1, step3)
    except Exception as e:
        logger.error("[Step4] Failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Final build failed: {str(e)}")

    logger.info("=== Request complete. Returning %d amounts ===", len(final.amounts))
    return final


# ════════════════════════════════════════════════════════════════════════════
# UPLOAD ENDPOINT (multipart/form-data for image files)
# ════════════════════════════════════════════════════════════════════════════

@app.post(
    "/extract-amounts/upload",
    response_model=Union[FinalResponse, GuardrailResponse],
    tags=["Pipeline"],
    summary="Upload an image file directly (PNG/JPG)",
)
async def extract_amounts_upload(file: UploadFile = File(...)):
    """
    Accepts a direct image file upload (PNG, JPG, WEBP).
    Preprocesses the image and runs the full 4-step pipeline.
    Useful for Postman/curl file uploads.
    """
    logger.info("=== /extract-amounts/upload [filename=%s] ===", file.filename)

    image_bytes = await file.read()

    # Preprocess for better OCR quality
    try:
        processed_bytes = preprocess_image(image_bytes)
    except Exception:
        processed_bytes = image_bytes  # Use raw bytes if preprocessing fails

    image_b64 = bytes_to_base64(processed_bytes)

    # Reuse main logic via ExtractRequest
    req = ExtractRequest(input_type="image", image_base64=image_b64)
    return await extract_amounts(req)


# ════════════════════════════════════════════════════════════════════════════
# DEBUG / STEP-BY-STEP ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.post("/step/1/ocr", response_model=Step1Response, tags=["Debug Steps"])
async def step1_only(request: ExtractRequest):
    """Run only Step 1: OCR / token extraction."""
    return extract_step1(request.input_type, request.text, request.image_base64)


@app.post("/step/2/normalize", response_model=Step2Response, tags=["Debug Steps"])
async def step2_only(request: ExtractRequest):
    """Run Steps 1 + 2: OCR + normalization."""
    step1 = extract_step1(request.input_type, request.text, request.image_base64)
    return normalize_step2(step1)


@app.post("/step/3/classify", response_model=Step3Response, tags=["Debug Steps"])
async def step3_only(request: ExtractRequest):
    """Run Steps 1 + 2 + 3: OCR + normalization + classification."""
    step1 = extract_step1(request.input_type, request.text, request.image_base64)
    step2 = normalize_step2(step1)
    return classify_step3(step1, step2)
