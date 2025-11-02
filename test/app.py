"""
FastAPI Application for ESPset Vibration Fault Diagnosis
======================================================

This FastAPI application provides a REST API for predicting fault types
in Electric Submersible Pumps using the trained machine learning model.

Author: Anuraj Ramesh
Date: 29-10-2025
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ESPset Vibration Fault Diagnosis API",
    description="API for predicting fault types in Electric Submersible Pumps using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
model = None
label_encoder = None
scaler = None
feature_names = None
model_metadata = None

# Pydantic models for request/response
class VibrationFeatures(BaseModel):
    """Vibration features for fault prediction"""
    median_8_13: float = Field(..., description="Median amplitude in interval (8% X, 13% X)")
    rms_98_102: float = Field(..., description="Root mean square in interval (98% X, 102% X)")
    median_98_102: float = Field(..., description="Median amplitude in interval (98% X, 102% X)")
    peak1x: float = Field(..., description="Amplitude at rotation frequency (X)")
    peak2x: float = Field(..., description="Amplitude at 2X (second harmonic)")
    a: float = Field(..., description="Exponential coefficient a")
    b: float = Field(..., description="Exponential coefficient b")
    
    class Config:
        schema_extra = {
            "example": {
                "median_8_13": 0.00217,
                "rms_98_102": 0.07393,
                "median_98_102": 0.000745,
                "peak1x": 0.04894,
                "peak2x": 0.0104,
                "a": -0.0002529389979972854,
                "b": -6.485688869308957
            }
        }

class PredictionResponse(BaseModel):
    """Response model for fault prediction"""
    predicted_fault: str = Field(..., description="Predicted fault type")
    confidence: float = Field(..., description="Prediction confidence score")
    probabilities: Dict[str, float] = Field(..., description="Probability for each fault type")
    model_info: Dict[str, Any] = Field(..., description="Model metadata")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_accuracy: float = None
    available_classes: List[str] = None

def load_model():
    """Load the trained model and preprocessing objects"""
    global model, label_encoder, scaler, feature_names, model_metadata
    
    try:
        model_path = "models/best_model.joblib"
        encoder_path = "models/label_encoder.joblib"
        scaler_path = "models/scaler.joblib"
        features_path = "models/feature_names.joblib"
        metadata_path = "models/model_metadata.joblib"
        
        # Check if all required files exist
        required_files = [model_path, encoder_path, scaler_path, features_path, metadata_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Load all components
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        model_metadata = joblib.load(metadata_path)
        
        logger.info("Model loaded successfully")
        logger.info(f"Model: {model_metadata['model_name']}")
        logger.info(f"Accuracy: {model_metadata['accuracy']:.4f}")
        logger.info(f"Classes: {model_metadata['classes']}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts"""
    try:
        load_model()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        # Don't raise the exception to allow the app to start without the model
        # The health check will indicate if the model is loaded

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ESPset Vibration Fault Diagnosis API",
        "version": "1.0.0",
        "docs": "/docs",
        "model_health": "/health",
        "model_info": "/model/info",
        "model_classes": "/classes"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global model, label_encoder, model_metadata
    
    if model is not None and label_encoder is not None:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_accuracy=model_metadata.get("accuracy"),
            available_classes=model_metadata.get("classes")
        )
    else:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_fault(features: VibrationFeatures):
    """Predict fault type based on vibration features"""
    global model, label_encoder, scaler, feature_names, model_metadata
    
    if model is None or label_encoder is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check the health endpoint."
        )
    
    try:
        # Convert input features to the expected format
        input_features = np.array([
            features.median_8_13,
            features.rms_98_102,
            features.median_98_102,
            features.peak1x,
            features.peak2x,
            features.a,
            features.b
        ]).reshape(1, -1)
        
        # Scale the features
        scaled_features = scaler.transform(input_features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        predicted_fault = label_encoder.inverse_transform([prediction])[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities_raw = model.predict_proba(scaled_features)[0]
            probabilities = {
                class_name: float(prob) 
                for class_name, prob in zip(label_encoder.classes_, probabilities_raw)
            }
            confidence = float(max(probabilities_raw))
        else:
            # For models without predict_proba, use a default confidence
            probabilities = {predicted_fault: 1.0}
            confidence = 1.0
        
        return PredictionResponse(
            predicted_fault=predicted_fault,
            confidence=confidence,
            probabilities=probabilities,
            model_info={
                "model_name": model_metadata.get("model_name"),
                "accuracy": model_metadata.get("accuracy"),
                "cv_mean": model_metadata.get("cv_mean"),
                "cv_std": model_metadata.get("cv_std")
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get information about the loaded model"""
    global model_metadata
    
    if model_metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return model_metadata

@app.get("/classes", response_model=List[str])
async def get_available_classes():
    """Get list of available fault classes"""
    global label_encoder
    
    if label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return label_encoder.classes_.tolist()

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(features_list: List[VibrationFeatures]):
    """Predict fault types for multiple samples"""
    global model, label_encoder, scaler, model_metadata
    
    if model is None or label_encoder is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check the health endpoint."
        )
    
    if len(features_list) > 100:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Batch size too large. Maximum 100 samples allowed."
        )
    
    try:
        results = []
        
        for features in features_list:
            # Convert input features to the expected format
            input_features = np.array([
                features.median_8_13,
                features.rms_98_102,
                features.median_98_102,
                features.peak1x,
                features.peak2x,
                features.a,
                features.b
            ]).reshape(1, -1)
            
            # Scale the features
            scaled_features = scaler.transform(input_features)
            
            # Make prediction
            prediction = model.predict(scaled_features)[0]
            predicted_fault = label_encoder.inverse_transform([prediction])[0]
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities_raw = model.predict_proba(scaled_features)[0]
                probabilities = {
                    class_name: float(prob) 
                    for class_name, prob in zip(label_encoder.classes_, probabilities_raw)
                }
                confidence = float(max(probabilities_raw))
            else:
                probabilities = {predicted_fault: 1.0}
                confidence = 1.0
            
            results.append(PredictionResponse(
                predicted_fault=predicted_fault,
                confidence=confidence,
                probabilities=probabilities,
                model_info={
                    "model_name": model_metadata.get("model_name"),
                    "accuracy": model_metadata.get("accuracy"),
                    "cv_mean": model_metadata.get("cv_mean"),
                    "cv_std": model_metadata.get("cv_std")
                }
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000)