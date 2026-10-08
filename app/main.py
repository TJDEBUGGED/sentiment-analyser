"""
main.py — FastAPI app exposing the sentiment analysis API.
Run: uvicorn app.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

from app.predictor import predictor

app = FastAPI(
    title="Sentiment Analyser API",
    description="Classifies text as positive, negative, or neutral.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    sentiment: str
    confidence: float
    scores: dict
    text_preview: str


@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 chars).")

    result = predictor.predict(text)
    return {
        **result,
        "text_preview": text[:120] + ("..." if len(text) > 120 else ""),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
