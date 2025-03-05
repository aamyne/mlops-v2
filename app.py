from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import os

app = FastAPI(title="Churn Prediction API", 
              description="API for predicting customer churn using a trained machine learning model")

# Define the path to the saved model
MODEL_PATH = "churn_model.joblib"

# Load the model when the app starts
try:
    loaded_data = joblib.load(MODEL_PATH)
    model = loaded_data["model"]
    encoder = loaded_data["encoder"]
    scaler = loaded_data["scaler"]
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    encoder = None
    scaler = None

# Define the input data model with all required fields
class InputData(BaseModel):
    state: str
    account_length: float
    area_code: float
    international_plan: str
    voice_mail_plan: str
    number_vmail_messages: float
    total_day_minutes: float
    total_day_calls: float
    total_eve_minutes: float
    total_eve_calls: float
    total_night_minutes: float
    total_night_calls: float
    total_intl_minutes: float
    total_intl_calls: float
    customer_service_calls: float

# Define the prediction route
@app.post("/predict", summary="Make a churn prediction")
async def predict(data: InputData):
    if model is None or encoder is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or preprocessors not loaded")
    
    try:
        # Convert input data to dictionary
        input_dict = data.dict()
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([input_dict])
        
        # Apply categorical encoding using the loaded encoder
        categorical_features = ["State", "International plan", "Voice mail plan"]
        
        # Rename columns to match training data
        input_df = input_df.rename(columns={
            "state": "State",
            "international_plan": "International plan",
            "voice_mail_plan": "Voice mail plan"
        })
        
        # Apply encoder
        input_encoded = encoder.transform(input_df[categorical_features])
        
        # Drop original categorical columns and join encoded ones
        input_processed = input_df.drop(columns=categorical_features).join(input_encoded)
        
        # Convert to numeric array for scaling
        input_array = input_processed.to_numpy()
        
        # Apply scaling
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        # Return the prediction
        return {
            "prediction": int(prediction[0]),
            "churn_status": "churn" if prediction[0] == 1 else "not churn",
            "probability": float(prediction_proba[0][1])  # Probability of class 1 (churn)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Excellence: Add retrain endpoint
class RetrainRequest(BaseModel):
    n_estimators: int = 50
    random_state: int = 42

@app.post("/retrain", summary="Retrain the model with new parameters")
async def retrain(request: RetrainRequest):
    try:
        from model_pipeline import prepare_data, train_model, evaluate_model, save_model
        
        # Prepare data
        X_train, y_train, X_test, y_test, new_encoder, new_scaler = prepare_data()
        
        # Train model with new parameters
        new_model = train_model(X_train, y_train, 
                           n_estimators=request.n_estimators, 
                           random_state=request.random_state)
        
        # Evaluate model
        accuracy, report = evaluate_model(new_model, X_test, y_test)
        
        # Save model
        save_model(new_model, new_encoder, new_scaler, MODEL_PATH)
        
        # Update global model
        global model, encoder, scaler
        model = new_model
        encoder = new_encoder
        scaler = new_scaler
        
        return {
            "message": "Model retrained successfully",
            "parameters": {
                "n_estimators": request.n_estimators,
                "random_state": request.random_state
            },
            "performance": {
                "accuracy": accuracy,
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1_score": report["weighted avg"]["f1-score"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

# Health check endpoint
@app.get("/", summary="Health check")
async def health_check():
    return {
        "status": "API is running", 
        "model_loaded": model is not None,
        "encoder_loaded": encoder is not None,
        "scaler_loaded": scaler is not None
    }

# Example endpoint that shows expected input format
@app.get("/example", summary="Example input data")
async def example():
    return {
        "state": "OH",
        "account_length": 107.0,
        "area_code": 415.0,
        "international_plan": "No",
        "voice_mail_plan": "Yes",
        "number_vmail_messages": 26.0,
        "total_day_minutes": 161.6,
        "total_day_calls": 123.0,
        "total_eve_minutes": 195.5,
        "total_eve_calls": 103.0,
        "total_night_minutes": 254.4,
        "total_night_calls": 103.0,
        "total_intl_minutes": 13.7,
        "total_intl_calls": 3.0,
        "customer_service_calls": 1.0
    }
