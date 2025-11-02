# ESPset Vibration Fault Diagnosis API

A FastAPI-based REST API for predicting fault types in Electric Submersible Pumps using machine learning models trained on the ESPset dataset.

## Features

- **Fault Prediction**: Predict fault types based on vibration features
- **Batch Processing**: Handle multiple predictions at once
- **Model Information**: Get details about the trained model
- **Health Monitoring**: Check API and model status
- **Docker Support**: Easy deployment with Docker
- **Interactive Documentation**: Built-in Swagger UI

## Quick Start

### 1. Train and Save the Model

First, run the analysis script to train and save the model:

```bash
python esp_vibration_analysis.py
```

This will create a `models/` directory with the trained model files.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API Locally

```bash
# Option 1: Direct uvicorn
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Option 2: Python module
python -m app.main
```

### 4. Test the API

```bash
python test_api.py
```

## API Endpoints

### Health Check

- **GET** `/health` - Check API and model status
- **GET** `/` - API information

### Model Information

- **GET** `/model/info` - Get model metadata
- **GET** `/classes` - Get available fault classes

### Predictions

- **POST** `/predict` - Single prediction
- **POST** `/predict/batch` - Batch predictions (max 100 samples)

## API Usage Examples

### Single Prediction

```python
import requests

# Sample vibration features
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
print(f"Confidence: {result['confidence']:.3f}")
print(f"Probabilities: {result['probabilities']}")
```

### Batch Prediction

```python
import requests

# Multiple samples
features_list = [
    {
        "median_8_13": 0.00217,
        "rms_98_102": 0.07393,
        "median_98_102": 0.000745,
        "peak1x": 0.04894,
        "peak2x": 0.0104,
        "a": -0.0002529389979972854,
        "b": -6.485688869308957
    },
    {
        "median_8_13": 0.00029,
        "rms_98_102": 0.26643,
        "median_98_102": 0.002295,
        "peak1x": 0.1595,
        "peak2x": 0.02152,
        "a": -0.0005463389799923974,
        "b": -7.6302091646600045
    }
]

response = requests.post("http://localhost:8000/predict/batch", json=features_list)
results = response.json()

for i, result in enumerate(results):
    print(f"Sample {i+1}: {result['predicted_fault']} (confidence: {result['confidence']:.3f})")
```

## Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t esp-vibration-api .

# Run the container
docker run -p 8000:8000 -v ${PWD}/models:/app/models:ro esp-vibration-api
```

### Using Docker Compose

```bash
# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## Input Features

The API expects the following vibration features:

| Feature         | Type  | Description                                  |
| --------------- | ----- | -------------------------------------------- |
| `median_8_13`   | float | Median amplitude in interval (8% X, 13% X)   |
| `rms_98_102`    | float | Root mean square in interval (98% X, 102% X) |
| `median_98_102` | float | Median amplitude in interval (98% X, 102% X) |
| `peak1x`        | float | Amplitude at rotation frequency (X)          |
| `peak2x`        | float | Amplitude at 2X (second harmonic)            |
| `a`             | float | Exponential coefficient a                    |
| `b`             | float | Exponential coefficient b                    |

## Output Format

### Single Prediction Response

```json
{
  "predicted_fault": "Normal",
  "confidence": 0.956,
  "probabilities": {
    "Normal": 0.956,
    "Unbalance": 0.032,
    "Misalignment": 0.008,
    "Bearing": 0.002,
    "Impeller": 0.001,
    "Cavitation": 0.001
  },
  "model_info": {
    "model_name": "Random Forest",
    "accuracy": 0.9561,
    "cv_mean": 0.9523,
    "cv_std": 0.0123
  }
}
```

## Interactive Documentation

Once the API is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Fault Types

The API can predict the following fault types:

1. **Normal** - No fault detected
2. **Unbalance** - Rotor unbalance
3. **Misalignment** - Shaft misalignment
4. **Bearing** - Bearing fault
5. **Impeller** - Impeller fault
6. **Cavitation** - Cavitation fault

## Error Handling

The API includes comprehensive error handling:

- **503 Service Unavailable**: Model not loaded
- **400 Bad Request**: Invalid input data
- **500 Internal Server Error**: Prediction failed
- **422 Unprocessable Entity**: Validation errors

## Performance

- **Latency**: < 100ms for single predictions
- **Throughput**: ~100 predictions/second
- **Memory**: ~500MB with model loaded
- **Batch Size**: Maximum 100 samples per batch

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_accuracy": 0.9561,
  "available_classes": [
    "Normal",
    "Unbalance",
    "Misalignment",
    "Bearing",
    "Impeller",
    "Cavitation"
  ]
}
```

### Model Information

```bash
curl http://localhost:8000/model/info
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run tests
pytest test_api.py
```

### Code Structure

```
├── app/
│   └── main.py          # FastAPI application
├── models/              # Trained model files
│   ├── best_model.joblib
│   ├── label_encoder.joblib
│   ├── scaler.joblib
│   ├── feature_names.joblib
│   └── model_metadata.joblib
├── requirements.txt     # Python dependencies
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── test_api.py        # API test script
└── API_README.md      # This documentation
```

## Troubleshooting

### Common Issues

1. **Model not loaded**: Ensure the `models/` directory contains all required files
2. **Port already in use**: Change the port in the command or Docker configuration
3. **Memory issues**: Increase Docker memory limits for large models
4. **CORS errors**: The API includes CORS middleware, but check client configuration

### Logs

```bash
# Docker logs
docker logs <container_id>

# Docker Compose logs
docker-compose logs -f esp-api
```

## License

This API is part of the ESPset vibration analysis project. Please refer to the original dataset license for usage terms.
