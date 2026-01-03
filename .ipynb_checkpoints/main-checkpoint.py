from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import joblib
import numpy as np
from MedicalTreeBot import MedicalTreeBot

app = FastAPI(title="Medical Diagnosis Expert System")

# 1. Load the saved assets
try:
    model = joblib.load("MedicalTreeModel.joblib")
    le = joblib.load("LabelEncoder.joblib")
    feature_names = joblib.load("features.joblib")
except FileNotFoundError:
    print("Error: Make sure all .joblib files are in the same directory.")

# 2. Define the Request Structure
class DiagnosisRequest(BaseModel):
    node_id: int = 0
    answer: Optional[int] = None  # 1 for Yes, 0 for No

# 3. Define the Response Structure
class PredictionResult(BaseModel):
    disease: str
    confidence: str

class DiagnosisResponse(BaseModel):
    status: str  # "question" or "final"
    node_id: Optional[int] = None
    question: Optional[str] = None
    results: Optional[List[PredictionResult]] = None

@app.post("/diagnose/", response_model=DiagnosisResponse)
def diagnose(request: DiagnosisRequest):
    tree = model.tree_
    node = request.node_id

    # Traverse logic: Move to the next node based on answer
    if request.answer is not None:
        if request.answer == 1:
            node = tree.children_right[node]
        else:
            node = tree.children_left[node]

    # Check if we reached a Leaf Node
    if tree.children_left[node] == -1:
        # Calculate probabilities for the classes in this leaf
        node_values = tree.value[node][0]
        total_samples = np.sum(node_values)
        
        # Get indices of top 3 classes
        top_indices = np.argsort(node_values)[-3:][::-1]
        
        final_results = []
        for idx in top_indices:
            if node_values[idx] > 0:
                final_results.append(
                    PredictionResult(
                        disease=le.classes_[idx],
                        confidence=f"{(node_values[idx] / total_samples) * 100:.2f}%"
                    )
                )
        
        return DiagnosisResponse(status="final", results=final_results)

    # If not a leaf, get the next feature (question)
    feature_idx = tree.feature[node]
    current_feature = feature_names[feature_idx]
    
    return DiagnosisResponse(
        status="question",
        node_id=int(node),
        question=f"Do you have {current_feature.replace('_', ' ')}?"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)