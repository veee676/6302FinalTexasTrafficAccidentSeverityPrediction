from fastapi import FastAPI
import joblib
from utils.feature_engineering import process_input
from rag.rag_pipeline import generate_explanation

app = FastAPI()

clf_model = joblib.load('models/clf_agent.pkl')
reg_model = joblib.load('models/reg_agent.pkl')

@app.post("/predict")
def predict(data: dict):
    df = process_input(data)
    severity_pred = clf_model.predict(df)[0]
    injury_pred = reg_model.predict(df)[0]
    explanation = generate_explanation(data, severity_pred, injury_pred)

    return {
        "predicted_severity": int(severity_pred),
        "predicted_injury_count": float(injury_pred),
        "explanation": explanation
    }
