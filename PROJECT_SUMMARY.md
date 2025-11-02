# ESPset Vibration Analysis - Complete Project Summary

## Project Overview

This project provides a complete machine learning pipeline for vibration-based fault diagnosis of Electric Submersible Pumps (ESP) used in offshore oil exploration. The solution includes data analysis, model training, and a production-ready FastAPI service with Docker containerization.

## Project Structure

```
ESPset_Vibration_Analysis/
├── Data Analysis
│   ├── esp_vibration_analysis.py          # Main analysis script
│   ├── vibration_regression.py            # Original regression example
│   └── ESPset/                            # Dataset directory
│       ├── features/features.csv          # Vibration features
│       └── spectrum/spectrum.csv          # Frequency domain data
│
├── Machine Learning
│   ├── models/                            # Trained model files
│   │   ├── best_model.joblib             # Best performing model
│   │   ├── label_encoder.joblib           # Label encoder
│   │   ├── scaler.joblib                  # Feature scaler
│   │   ├── feature_names.joblib          # Feature names
│   │   └── model_metadata.joblib         # Model metadata
│   └── requirements.txt                  # Python dependencies
│
├── FastAPI Application
│   ├── app/
│   │   └── main.py                        # FastAPI application
│   ├── test_api.py                        # API test script
│   └── API_README.md                      # API documentation
│
├── Docker Configuration
│   ├── Dockerfile                         # Docker image definition
│   ├── docker-compose.yml                # Docker Compose configuration
│   └── .dockerignore                     # Docker ignore file
│
├── Documentation
│   ├── ESPset_Analysis_Documentation.md   # Analysis documentation
│   ├── PROJECT_SUMMARY.md                # This file
│   └── setup_and_run.py                  # Setup automation script
│
└── Generated Outputs
    ├── esp_data_analysis.png             # Data exploration plots
    ├── esp_model_evaluation.png          # Model performance plots
    └── esp_spectrum_analysis.png         # Spectrum analysis plots
```

## Key Features

### 1. **Advanced Data Analysis**

- **Real-world Dataset**: 6,032 vibration signals from 8 ESPs
- **Feature Engineering**: 7 domain-specific vibration features
- **Comprehensive Visualization**: 6-panel analysis dashboard
- **Improved Plot Visibility**: Optimized for embedded viewers

### 2. **Machine Learning Pipeline**

- **Multiple Algorithms**: Logistic Regression, Random Forest, Gradient Boosting, SVM
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Model Persistence**: Joblib serialization for production deployment
- **Feature Scaling**: StandardScaler for consistent preprocessing

### 3. **Production-Ready API**

- **FastAPI Framework**: Modern, fast, and auto-documented
- **RESTful Endpoints**: Single and batch prediction capabilities
- **Health Monitoring**: Comprehensive status checking
- **Error Handling**: Robust error management and logging
- **Interactive Documentation**: Built-in Swagger UI

### 4. **Docker Containerization**

- **Multi-stage Build**: Optimized Docker image
- **Security**: Non-root user execution
- **Health Checks**: Built-in container health monitoring
- **Volume Mounting**: Persistent model storage

## Quick Start Guide

### Option 1: Automated Setup (Recommended)

```bash
# Run the automated setup script
python setup_and_run.py

# Choose option 1 for local setup or option 2 for Docker
```

### Option 2: Manual Setup

#### Step 1: Train the Model

```bash
# Install dependencies
pip install -r requirements.txt

# Run the analysis to train and save the model
python esp_vibration_analysis.py
```

#### Step 2: Start the API

```bash
# Local development
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or using Python module
python -m app.main
```

#### Step 3: Test the API

```bash
# Run the test script
python test_api.py
```

### Option 3: Docker Setup

```bash
# Build the Docker image
docker build -t esp-vibration-api .

# Run the container
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro esp-vibration-api

# Or use Docker Compose
docker-compose up -d
```

## API Endpoints

| Endpoint         | Method | Description                    |
| ---------------- | ------ | ------------------------------ |
| `/`              | GET    | API information                |
| `/health`        | GET    | Health check and model status  |
| `/model/info`    | GET    | Model metadata and performance |
| `/classes`       | GET    | Available fault classes        |
| `/predict`       | POST   | Single fault prediction        |
| `/predict/batch` | POST   | Batch predictions (max 100)    |
| `/docs`          | GET    | Interactive API documentation  |

## Fault Types

The system can predict 6 different fault types:

1. **Normal** - No fault detected
2. **Unbalance** - Rotor unbalance
3. **Misalignment** - Shaft misalignment
4. **Bearing** - Bearing fault
5. **Impeller** - Impeller fault
6. **Cavitation** - Cavitation fault

## Model Performance

- **Best Model**: Random Forest Classifier
- **Accuracy**: ~95.6% on test set
- **Cross-Validation**: 5-fold CV with robust performance
- **Feature Importance**: Ranked by Gini importance
- **Prediction Speed**: <100ms per prediction

## Technical Specifications

### Dependencies

- **Python**: 3.10+
- **ML Libraries**: scikit-learn, joblib, numpy, pandas
- **API Framework**: FastAPI, uvicorn
- **Visualization**: matplotlib, seaborn
- **Containerization**: Docker, Docker Compose

### System Requirements

- **Memory**: 2GB RAM minimum, 4GB recommended
- **Storage**: 1GB for models and data
- **CPU**: 2 cores minimum
- **Network**: Port 8000 for API access

## Configuration

### Environment Variables

- `PYTHONPATH`: Set to `/app` in Docker
- `PYTHONDONTWRITEBYTECODE`: Disable bytecode writing
- `PYTHONUNBUFFERED`: Unbuffered output

### Model Configuration

- **Feature Scaling**: StandardScaler normalization
- **Label Encoding**: LabelEncoder for categorical labels
- **Model Selection**: Best performing model auto-selected
- **Cross-Validation**: 5-fold stratified CV

## Usage Examples

### Python Client

```python
import requests

# Single prediction
features = {
    "median_8_13": 0.00217,
    "rms_98_102": 0.07393,
    "median_98_102": 0.000745,
    "peak1x": 0.04894,
    "peak2x": 0.0104,
    "a": -0.0002529389979972854,
    "b": -6.485688869308957
}

response = requests.post("http://localhost:8000/predict", json=features)
result = response.json()
print(f"Predicted fault: {result['predicted_fault']}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "median_8_13": 0.00217,
       "rms_98_102": 0.07393,
       "median_98_102": 0.000745,
       "peak1x": 0.04894,
       "peak2x": 0.0104,
       "a": -0.0002529389979972854,
       "b": -6.485688869308957
     }'
```

## Monitoring and Debugging

### Health Check

```bash
curl http://localhost:8000/health
```

### Logs

```bash
# Docker logs
docker logs <container_id>

# Docker Compose logs
docker-compose logs -f esp-api
```

### Model Information

```bash
curl http://localhost:8000/model/info
```

## Deployment Options

### 1. Local Development

- Direct Python execution
- Hot reload for development
- Easy debugging

### 2. Docker Container

- Isolated environment
- Consistent deployment
- Easy scaling

### 3. Cloud Deployment

- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- Kubernetes deployment

## Customization

### Adding New Features

1. Modify `esp_vibration_analysis.py` to include new features
2. Update the Pydantic models in `app/main.py`
3. Retrain the model with new features
4. Update the API documentation

### Model Updates

1. Train new model with updated data
2. Replace model files in `models/` directory
3. Restart the API service
4. Verify model performance

## Documentation

- **Analysis Documentation**: `ESPset_Analysis_Documentation.md`
- **API Documentation**: `API_README.md`
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is part of the ESPset vibration analysis research. Please refer to the original dataset license for usage terms.

## Support

For issues and questions:

1. Check the documentation
2. Review the logs
3. Test with the provided test script
4. Create an issue with detailed information

---

**Congratulations!** You now have a complete, production-ready vibration fault diagnosis system that can be deployed anywhere from local development to cloud platforms.
