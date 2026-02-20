import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.feature_engine import build_feature_vector

app = FastAPI(
    title="Sri Lanka Rice Price Prediction API",
    description="API for predicting retail rice prices using XGBoost.",
    version="1.0.0"
)

# Load model on startup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "xgb_model.pkl")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
    model = bundle["model"]
    features = bundle["features"]

class PredictionInput(BaseModel):
    year: int
    month: int
    province: str
    variety: str
    lag1: float
    lag2: float
    lag3: float
    usd_lkr: float
    inflation: float
    crisis: bool
    covid: bool

class PredictionOutput(BaseModel):
    prediction: float
    currency: str = "LKR"
    unit: str = "KG"

@app.get("/")
def read_root():
    return {"message": "Welcome to the Rice Price Prediction API", "status": "active"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    try:
        # Calculate derived rolling features
        roll3    = (data.lag1 + data.lag2 + data.lag3) / 3
        roll6    = roll3 * 0.97 # Approximated as in frontend
        roll3std = abs(data.lag1 - data.lag3) / 2 # Approximated

        X = build_feature_vector(
            features, data.year, data.month, data.province, data.variety,
            data.lag1, data.lag2, data.lag3, roll3, roll6, roll3std,
            data.usd_lkr, data.inflation, data.crisis, data.covid
        )
        
        prediction = model.predict(X)[0]
        return PredictionOutput(prediction=float(prediction))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
