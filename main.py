from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List
from MedicalTreeBot import MedicalTreeBot 
from fastapi.responses import HTMLResponse 
from fastapi.templating import Jinja2Templates
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    print("Trying to load files...")
    model = joblib.load(os.path.join(BASE_DIR, "MedicalTreeModel.joblib"))
    le = joblib.load(os.path.join(BASE_DIR, "LabelEncoder.joblib"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "features.joblib"))
    print("All files loaded successfully!")
except Exception as e:
    print(f"Error loading files: {e}")

app = FastAPI(title="Medical Diagnosis Expert System")
templates = Jinja2Templates(directory="templates")

class DiagnosisRequest(BaseModel):
    node_id: int = 0
    answer: Optional[int] = None

class PredictionResult(BaseModel):
    disease: str
    confidence: str

class DiagnosisResponse(BaseModel):
    status: str
    node_id: Optional[int] = None
    question: Optional[str] = None
    results: Optional[List[PredictionResult]] = None


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/diagnose/", response_model=DiagnosisResponse)
def diagnose(request: DiagnosisRequest):
    bot = MedicalTreeBot(model, feature_names, le.classes_)
    bot.current_node = request.node_id

    if request.answer is not None:
        bot.submit_answer(request.answer)

    if bot.is_leaf():
        final_results = bot.get_result()
        return DiagnosisResponse(
            status="final", 
            results=final_results
        )

    next_question = bot.get_question()
    
    return DiagnosisResponse(
        status="question",
        node_id=int(bot.current_node),
        question=f"Do you have {next_question.replace('_', ' ')}?"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)