from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Model aur columns load kar rahe hain
model = load("model_pipeline.pkl")
COLUMNS = load("columns.pkl")

app = FastAPI(title="Telehealth AI - Speech Recognition & Urgency Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audio features aur unke realistic ranges ke sath Pydantic Schema
class Data(BaseModel):
    Duration_Sec: float = Field(..., ge=0.0, le=3600.0, description="Duration of the audio in seconds")
    Mfcc_Mean: float = Field(..., description="Mean value of Mel-Frequency Cepstral Coefficients")
    Mel_Spectrogram_Mean: float = Field(..., description="Mean value of Mel Spectrogram")
    Log_Mel_Spectrogram_Mean: float = Field(..., description="Mean value of Log Mel Spectrogram")
    Pitch_Hz: float = Field(..., ge=0.0, le=500.0, description="Fundamental pitch frequency in Hz")
    Energy: float = Field(..., ge=0.0, description="Signal energy")
    RMS_Energy: float = Field(..., ge=0.0, description="Root Mean Square Energy")
    Zero_Crossing_Rate: float = Field(..., ge=0.0, le=1.0, description="Zero crossing rate")
    Delta_Features_Mean: float = Field(..., description="Mean of delta features")
    Delta_Delta_Features_Mean: float = Field(..., description="Mean of delta-delta features")

# Regression ke liye Professional Response Structure
class PredictionResponse(BaseModel):
    predicted_urgency_score: float = Field(..., description="The continuous regression output score predicted by the model")
    urgency_level: Literal['Low Urgency', 'Moderate Urgency', 'High Urgency'] = Field(..., description="Categorical interpretation of the urgency score")

@app.get("/")
def greet():
    return {"message": "Welcome to Amir WED - Telehealth AI Speech Recognition API"}

@app.post('/predict', response_model=PredictionResponse)
def predict(data: Data):
    # Input data ko exact training columns ke order mein DataFrame banana
    input_row = pd.DataFrame([{
        "duration_sec": data.Duration_Sec,
        "mfcc_mean": data.Mfcc_Mean,
        "mel_spectrogram_mean": data.Mel_Spectrogram_Mean,
        "log_mel_spectrogram_mean": data.Log_Mel_Spectrogram_Mean,
        "pitch_hz": data.Pitch_Hz,
        "energy": data.Energy,
        "rms_energy": data.RMS_Energy,
        "zero_crossing_rate": data.Zero_Crossing_Rate,
        "delta_features_mean": data.Delta_Features_Mean,
        "delta_delta_features_mean": data.Delta_Delta_Features_Mean
    }])[COLUMNS] # columns.pkl ke order ko ensure karne ke liye
    
    # Regression model prediction (Continuous float value)
    raw_prediction = float(model.predict(input_row)[0])
    urgency_score = round(raw_prediction, 4)
    
    # Score ke basis par interpretive urgency level assign karna (distribution ke hisaab se)
    if urgency_score < 0.20:
        level = "Low Urgency"
    elif urgency_score <= 0.30:
        level = "Moderate Urgency"
    else:
        level = "High Urgency"
        
    return PredictionResponse(
        predicted_urgency_score=urgency_score,
        urgency_level=level
    )