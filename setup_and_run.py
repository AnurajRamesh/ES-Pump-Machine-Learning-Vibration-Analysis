"""
Setup and Run Script for ESPset Vibration Analysis API
=====================================================

This script helps set up and run the complete ESPset vibration analysis
and API system.

Author: Anuraj Ramesh
Date: 29.10.2025
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("Success!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if os.path.exists(file_path):
        print(f"{description}: {file_path}")
        return True
    else:
        print(f"{description}: {file_path} (not found)")
        return False

def wait_for_api(url, max_attempts=30, delay=2):
    """Wait for API to be ready"""
    print(f"\nWaiting for API to be ready at {url}...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("API is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(delay)
    
    print("API failed to start within expected time")
    return False

def main():
    """Main setup and run function"""
    print("ESPset Vibration Analysis API Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("esp_vibration_analysis.py"):
        print("Please run this script from the project root directory")
        return False
    
    # Step 1: Install dependencies
    print("\nInstalling dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("Failed to install dependencies")
        return False
    
    # Step 2: Check if ESPset data exists
    print("\nChecking ESPset data...")
    esp_data_exists = check_file_exists("ESPset/features/features.csv", "ESPset features file")
    
    if not esp_data_exists:
        print("ESPset data not found. Please ensure the ESPset dataset is in the ESPset/ directory")
        print("   You can download it from: https://github.com/NINFA-UFES/ESPset")
        return False
    
    # Step 3: Train and save the model
    print("\nTraining and saving the model...")
    if not run_command("python esp_vibration_analysis.py", "Training machine learning model"):
        print("Failed to train the model")
        return False
    
    # Step 4: Check if model files were created
    print("\nChecking model files...")
    model_files = [
        "models/best_model.joblib",
        "models/label_encoder.joblib", 
        "models/scaler.joblib",
        "models/feature_names.joblib",
        "models/model_metadata.joblib"
    ]
    
    all_models_exist = True
    for model_file in model_files:
        if not check_file_exists(model_file, f"Model file"):
            all_models_exist = False
    
    if not all_models_exist:
        print("Some model files are missing")
        return False
    
    # Step 5: Start the API
    print("\nStarting the API...")
    print("Starting FastAPI server on http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Start the API in the background
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
        
        # Wait for API to be ready
        if wait_for_api("http://localhost:8000/health"):
            print("\nAPI is running successfully!")
            print("\nAvailable endpoints:")
            print("  - API Documentation: http://localhost:8000/docs")
            print("  - Health Check: http://localhost:8000/health")
            print("  - Model Info: http://localhost:8000/model/info")
            print("  - Predict: http://localhost:8000/predict")
            
            print("\nTo test the API, run:")
            print("  python test_api.py")
            
            print("\nPress Ctrl+C to stop the API...")
            
            # Keep the script running
            try:
                api_process.wait()
            except KeyboardInterrupt:
                print("\nStopping API...")
                api_process.terminate()
                api_process.wait()
                print("API stopped")
        
    except Exception as e:
        print(f"Failed to start API: {e}")
        return False
    
    return True

def docker_setup():
    """Setup using Docker"""
    print("\nDocker Setup")
    print("=" * 30)
    
    # Build Docker image
    if not run_command("docker build -t esp-vibration-api .", "Building Docker image"):
        return False
    
    # Run Docker container
    print("\nStarting Docker container...")
    docker_cmd = "docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro esp-vibration-api"
    
    print(f"Running: {docker_cmd}")
    print("The API will be available at http://localhost:8000")
    print("Press Ctrl+C to stop the container")
    
    try:
        subprocess.run(docker_cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print("\nStopping Docker container...")
        subprocess.run("docker stop $(docker ps -q --filter ancestor=esp-vibration-api)", shell=True)
        print("Docker container stopped")
    
    return True

if __name__ == "__main__":
    print("ESPset Vibration Analysis API Setup")
    print("Choose setup method:")
    print("1. Local setup (recommended for development)")
    print("2. Docker setup (recommended for production)")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        success = main()
        if success:
            print("\nSetup completed successfully!")
        else:
            print("\nSetup failed. Please check the errors above.")
            sys.exit(1)
    elif choice == "2":
        success = docker_setup()
        if success:
            print("\nDocker setup completed successfully!")
        else:
            print("\nDocker setup failed. Please check the errors above.")
            sys.exit(1)
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
        sys.exit(1)

