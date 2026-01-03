# MedicalDiagnosisAPI
MedicalTreeAPI is a FastAPI-based medical diagnosis backend that uses a decision tree model to interactively assess symptoms and provide probable disease predictions with confidence scores.
---

## Features

- Interactive symptom-based diagnosis
- Returns top probable diseases with confidence percentages
- Built with **FastAPI** for easy integration
- Lightweight and ready for deployment on platforms like Render or Railway

---

## Installation

1. Clone the repository:

git clone https://github.com/Mariam-A15/MedicalTreeAPI.git
cd MedicalTreeAPI

2. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

3. Install dependencies:

pip install -r requirements.txt

---

## Usage

Start the FastAPI server:

python -m uvicorn main:app --host 0.0.0.0 --port 8000

Access the interactive API documentation (Swagger UI) at:

http://127.0.0.1:8000/docs

### Example Request

POST /diagnose/
{
  "node_id": 0,
  "answer": null
}

### Example Response

{
  "status": "question",
  "node_id": 3,
  "question": "Do you have headache?"
}

---

## Files

- main.py – FastAPI app entry point
- MedicalTreeBot.py – Decision tree bot logic
- MedicalTreeModel.joblib – Trained decision tree model
- LabelEncoder.joblib – Encoder for disease labels
- features.joblib – List of symptom features

---

## Deployment

- Can be deployed on **Render**, **Railway**, or any Python-compatible cloud server
- Make sure to set the host to `0.0.0.0` and port to the platform-assigned `$PORT` when deploying online

---

## Notes

- This project is for **educational and demonstration purposes**
- The diagnosis is **not a replacement for professional medical advice**
- For local testing, the server runs on `localhost:8000`

---

## Links

- GitHub Repo: https://github.com/Mariam-A15/MedicalTreeAPI
- API Docs: http://127.0.0.1:8000/docs
