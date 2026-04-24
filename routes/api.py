from fastapi import FastAPI
import uuid
from fastapi.responses import JSONResponse
from db.models import insert_document, query_document, answer_question
from src.predictor import classify_document

app = FastAPI()

@app.post("/ask")
async def ask(question: str ):
    answer = answer_question(question)
    return {
        "answer": answer,
        "status": "ok"
    }

@app.post("/upload-document")
async def upload_document(document: str):
    insert_document(document, str(uuid.uuid4()))
    return JSONResponse({"status": "ok"}, status_code=201)

@app.post("/predict-type")
async def predict_type(document: str):
    response = classify_document(document)
    return JSONResponse({
        "classification": response["classification"],
        "confidence": response["confidence"],
        "probs": response["probs"]
    },
    status_code=201
    )