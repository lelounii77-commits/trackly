# ============================================================
# Trackly — FastAPI Backend
# ============================================================
# Run: uvicorn backend:app --reload --port 8000
# ============================================================

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import os
import io
from groq import Groq

# ── App ──────────────────────────────────────────────────────
app = FastAPI(title="Trackly API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve Frontend ───────────────────────────────────────────
@app.get("/app")
def serve_frontend():
    return FileResponse("trackly_dashboard.html")

# ── Constants ────────────────────────────────────────────────
VALID_SEMESTERS = ["2023_Fall", "2024_Spring", "2024_Fall", "2025_Spring"]
VALID_MAJORS = [
    "Artificial Intelligence", "Computer Science", "Cybersecurity",
    "Data Science", "Information Systems", "Information Technology",
    "Software Engineering",
]
REQUIRED_COLUMNS = [
    "semester", "major", "attendance_rate_pct",
    "missed_assignments", "midterms_total_30", "coursework_30"
]
MODEL_PATH = "academic_risk_model.joblib"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"

# ── Model Loading ─────────────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# ── Schemas ──────────────────────────────────────────────────
class PredictRequest(BaseModel):
    semester: str
    major: str
    attendance_rate_pct: float = Field(..., ge=40, le=100)
    missed_assignments: int = Field(..., ge=0, le=12)
    midterms_total_30: float = Field(..., ge=0, le=30)
    coursework_30: float = Field(..., ge=0, le=30)
    groq_api_key: str = ""

    @validator("semester")
    def check_semester(cls, v):
        if v not in VALID_SEMESTERS:
            raise ValueError(f"Invalid semester. Choose from: {VALID_SEMESTERS}")
        return v

    @validator("major")
    def check_major(cls, v):
        if v not in VALID_MAJORS:
            raise ValueError(f"Invalid major. Choose from: {VALID_MAJORS}")
        return v

class PredictResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    risk_probability: float
    recommendation: str = ""

# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Trackly API is running"}

@app.get("/health")
def health():
    return {
        "model_loaded": model is not None,
        "semesters": VALID_SEMESTERS,
        "majors": VALID_MAJORS,
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded.")

    input_df = pd.DataFrame([{
        "semester":            req.semester,
        "major":               req.major,
        "attendance_rate_pct": req.attendance_rate_pct,
        "missed_assignments":  req.missed_assignments,
        "midterms_total_30":   req.midterms_total_30,
        "coursework_30":       req.coursework_30,
    }])

    predicted = int(model.predict(input_df)[0])
    risk_prob = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else 0.5
    label     = "At Risk" if predicted == 1 else "Not at Risk"

    recommendation = ""
    api_key = req.groq_api_key or GROQ_API_KEY
    if api_key:
        try:
            client = Groq(api_key=api_key)
            prompt = f"""You are an academic advisor assistant at a university.
An AI model analyzed a student and predicted: {label} (probability: {risk_prob:.1%}).

Student data:
- Semester: {req.semester}
- Major: {req.major}
- Attendance: {req.attendance_rate_pct}%
- Missed Assignments: {req.missed_assignments}
- Midterm Score: {req.midterms_total_30}/30
- Coursework Score: {req.coursework_30}/30

Write 3 short bullet-point recommendations for the academic advisor.
Each point starts with •. Be specific, supportive, and actionable."""
            msg = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            recommendation = msg.choices[0].message.content.strip()
        except Exception as e:
            recommendation = f"(LLM unavailable: {str(e)[:60]})"

    return PredictResponse(
        predicted_class=predicted,
        predicted_label=label,
        risk_probability=round(risk_prob, 4),
        recommendation=recommendation,
    )


# ── Batch Prediction from Excel ───────────────────────────────
@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not loaded.")

    # Check file type
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Please upload an Excel file (.xlsx or .xls)")

    # Read the Excel file
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not read Excel file: {str(e)}")

    # Check required columns exist
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(400, f"Missing columns: {missing_cols}. Required: {REQUIRED_COLUMNS}")

    # Run predictions
    results = []
    for i, row in df.iterrows():
        try:
            input_df = pd.DataFrame([{
                "semester":            row["semester"],
                "major":               row["major"],
                "attendance_rate_pct": float(row["attendance_rate_pct"]),
                "missed_assignments":  int(row["missed_assignments"]),
                "midterms_total_30":   float(row["midterms_total_30"]),
                "coursework_30":       float(row["coursework_30"]),
            }])

            predicted = int(model.predict(input_df)[0])
            risk_prob = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else 0.5
            label     = "At Risk" if predicted == 1 else "Not at Risk"

            results.append({
                "row":               i + 1,
                "semester":          row["semester"],
                "major":             row["major"],
                "attendance_rate_pct": float(row["attendance_rate_pct"]),
                "missed_assignments":  int(row["missed_assignments"]),
                "midterms_total_30":   float(row["midterms_total_30"]),
                "coursework_30":       float(row["coursework_30"]),
                "predicted_label":   label,
                "risk_probability":  round(risk_prob, 4),
                "error":             None,
            })
        except Exception as e:
            results.append({
                "row":             i + 1,
                "predicted_label": "Error",
                "risk_probability": None,
                "error":           str(e),
            })

    total      = len(results)
    at_risk    = sum(1 for r in results if r["predicted_label"] == "At Risk")
    not_at_risk = sum(1 for r in results if r["predicted_label"] == "Not at Risk")

    return {
        "total_students": total,
        "at_risk_count":  at_risk,
        "safe_count":     not_at_risk,
        "results":        results,
    }
