# PS4 — AI-Powered Amount Detection in Medical Documents

A FastAPI backend service that extracts and classifies financial amounts from medical bills and receipts — supporting both typed text and scanned/crumpled image inputs.

---

## Architecture

```
Input (text / image)
       │
       ▼
Step 1: OCR / Token Extraction   → raw_tokens + currency_hint
       │
       ▼
Step 2: Numeric Normalization    → fix 'l'→'1', 'O'→'0' etc.
       │
       ▼
Step 3: Context Classification   → label amounts (total, paid, due…)
       │
       ▼
Step 4: Final Builder            → add currency + provenance + sanity check
       │
       ▼
   JSON Response
```

**Tech Stack:** Python 3.13 · FastAPI · Gemini 2.0 Flash (Vision + Text) · Pydantic v2 · uvicorn

---

## Setup

### 1. Clone & install dependencies
```bash
git clone <your-repo-url>
cd plum
pip install -r requirements.txt
```

### 2. Set your Gemini API Key
```bash
cp .env .env.local
# Edit .env and set your key:
echo "GEMINI_API_KEY=your_key_here" > .env
```
Get a free key at: https://aistudio.google.com/app/apikey

### 3. Run the server
```bash
uvicorn main:app --reload --port 8000
```

The API docs will be at: http://localhost:8000/docs

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/extract-amounts` | Full pipeline (JSON body) |
| `POST` | `/extract-amounts/upload` | Full pipeline (image file upload) |
| `POST` | `/step/1/ocr` | Debug: Step 1 only |
| `POST` | `/step/2/normalize` | Debug: Steps 1–2 |
| `POST` | `/step/3/classify` | Debug: Steps 1–3 |

---

## Sample Requests

### Text Input
```bash
curl -X POST http://localhost:8000/extract-amounts \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "text",
    "text": "Total: INR 1200 | Paid: 1000 | Due: 200 | Discount: 10%"
  }'
```

**Expected Response:**
```json
{
  "currency": "INR",
  "amounts": [
    {"type": "total_bill", "value": 1200.0, "source": "text: 'Total: INR 1200'"},
    {"type": "paid",       "value": 1000.0, "source": "text: 'Paid: 1000'"},
    {"type": "due",        "value": 200.0,  "source": "text: 'Due: 200'"}
  ],
  "status": "ok",
  "warning": null
}
```

### Noisy OCR Text (simulating crumpled scan)
```bash
curl -X POST http://localhost:8000/extract-amounts \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "text",
    "text": "T0tal: Rs l200 | Pald: 1O00 | Oue: 2O0"
  }'
```

### Image File Upload
```bash
curl -X POST http://localhost:8000/extract-amounts/upload \
  -F "file=@/path/to/medical_bill.jpg"
```

### Image as Base64 JSON
```bash
# Encode your image
B64=$(base64 -i /path/to/bill.jpg)

curl -X POST http://localhost:8000/extract-amounts \
  -H "Content-Type: application/json" \
  -d "{\"input_type\": \"image\", \"image_base64\": \"$B64\"}"
```

### Step-by-step debug
```bash
# Only OCR
curl -X POST http://localhost:8000/step/1/ocr \
  -H "Content-Type: application/json" \
  -d '{"input_type": "text", "text": "Total: INR 1200 | Due: 200"}'

# OCR + Normalize
curl -X POST http://localhost:8000/step/2/normalize \
  -H "Content-Type: application/json" \
  -d '{"input_type": "text", "text": "T0tal: Rs l200"}'
```

---

## Guardrails

| Condition | Response |
|-----------|----------|
| No numeric tokens found | `{"status": "no_amounts_found", "reason": "document too noisy"}` |
| OCR confidence < 0.30 | `{"status": "no_amounts_found", "reason": "document too noisy"}` |
| total ≠ paid + due (>5% diff) | Response includes `"warning": "amounts_mismatch: ..."` |
| No currency found | Defaults to `"INR"` |

---

## Running Tests
```bash
# From the project root
python -m pytest tests/ -v
```

Tests that require Gemini API (Steps 1-image, 3) are mocked separately and won't hit the API.

---

## Expose with ngrok (for submission demo)
```bash
# Install ngrok: https://ngrok.com/download
uvicorn main:app --port 8000 &
ngrok http 8000
```

Use the ngrok URL in your Postman collection and screen recording.

---

## Project Structure
```
plum/
├── main.py                   # FastAPI app + all endpoints
├── requirements.txt
├── .env                      # GEMINI_API_KEY
├── README.md
├── pipeline/
│   ├── ocr_extractor.py      # Step 1: OCR / text extraction
│   ├── normalizer.py         # Step 2: Digit correction
│   ├── classifier.py         # Step 3: Context classification (Gemini)
│   └── builder.py            # Step 4: Final JSON + provenance + guardrails
├── models/
│   └── schemas.py            # Pydantic v2 request/response models
├── utils/
│   ├── gemini_client.py      # Gemini API wrapper (text + vision)
│   └── image_utils.py        # Image preprocessing for OCR
└── tests/
    ├── test_ocr.py
    ├── test_normalizer.py
    ├── test_builder.py
    └── sample_inputs/
        ├── text_input.json
        └── noisy_ocr_input.json
```
