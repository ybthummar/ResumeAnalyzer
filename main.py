from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import fitz  # PyMuPDF
import joblib
import os
import uvicorn

# Load ML model and vectorizer (use relative paths!)
model = joblib.load("models/resume_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for now; secure later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    content = await file.read()

    try:
        prediction = real_model_predict(content)
    except Exception as e:
        return {"error": f"Failed to process resume: {str(e)}"}

    return {"prediction": prediction}

def real_model_predict(file_bytes: bytes) -> str:
    text = extract_text_from_pdf(file_bytes)
    vectorized_text = vectorizer.transform([text])
    predicted_label = model.predict(vectorized_text)[0]
    return predicted_label

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open("pdf", file_bytes)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
