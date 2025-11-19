from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from src.ensemble.ensemble import FraudEnsemble

app = FastAPI(title="Fraud Detection API")

class FraudInput(BaseModel):
    features: list


# Load ensemble (boosted, no calibrator)
ensemble = FraudEnsemble()

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict_fraud(data: FraudInput):
    try:
        features = data.features

        # Validate feature count
        if len(features) != 30:
            return JSONResponse(
                status_code=400,
                content={"error": f"Expected 30 features, got {len(features)}"}
            )

        # Convert to float
        try:
            features = [float(x) for x in features]
        except:
            return JSONResponse(
                status_code=400,
                content={"error": "All features must be numeric."}
            )

        # Predict (ensemble.predict_proba handles single-sample input)
        prob = float(ensemble.predict_proba(features))

        if prob < ensemble.safe_threshold:
            label = "SAFE TRANSACTION"
        elif prob > ensemble.fraud_threshold:
            label = "FRAUD DETECTED"
        else:
            label = "REVIEW REQUIRED"

        return {
            "fraud_probability": prob,
            "risk_category": label,
            "features_used": features
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})