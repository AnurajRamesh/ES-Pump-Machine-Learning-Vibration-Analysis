# ESPset Vibration Analysis Documentation

## Overview

This documentation describes the `esp_vibration_analysis.py` script, which performs comprehensive analysis of real-world vibration data from Electric Submersible Pumps (ESP) used in offshore oil exploration. The script implements machine learning-based fault diagnosis using the ESPset dataset.

## Dataset Information

### ESPset Dataset

- **Source**: Real-world vibration signals from Electric Submersible Centrifugal Pumps
- **Samples**: 6,032 vibration signals from 8 different ESPs
- **Features**: 7 engineered features from vibration spectra
- **Labels**: 6 fault types (Normal, Unbalance, Misalignment, Bearing, Impeller, Cavitation)
- **Spectrum Data**: 6,032 × 12,103 frequency domain amplitudes

### Features Description

| Feature          | Description                                  |
| ---------------- | -------------------------------------------- |
| `median(8,13)`   | Median amplitude in interval (8% X, 13% X)   |
| `median(98,102)` | Median amplitude in interval (98% X, 102% X) |
| `rms(98,102)`    | Root mean square in interval (98% X, 102% X) |
| `peak1x`         | Amplitude at rotation frequency (X)          |
| `peak2x`         | Amplitude at 2X (second harmonic)            |
| `a`              | Exponential coefficient a in e^(a\*A+b)      |
| `b`              | Exponential coefficient b in e^(a\*A+b)      |

Where X is the rotation frequency of the ESP.

## Script Structure

### Main Functions

#### 1. Data Loading and Exploration

```python
load_esp_features(features_path="ESPset/features/features.csv")
explore_esp_data(data)
```

- Loads ESPset features from CSV file
- The dataset for the analysis can be downloaded from https://drive.google.com/drive/folders/1VLuOmOIODiXa3ybYe3Z6845pZ6Sg2ce4?usp=drive_link
- Provides comprehensive data exploration including:
  - Dataset shape and basic statistics
  - Label distribution analysis
  - ESP ID distribution
  - Missing value analysis

#### 2. Data Visualization

```python
visualize_esp_data(data)
```

Creates a 2×3 grid of visualizations:

- **Fault Type Distribution**: Pie chart with improved percentage visibility
- **Samples per ESP ID**: Bar chart showing data distribution across ESPs
- **Peak1x vs Peak2x**: Scatter plot colored by fault type
- **Median(8,13) vs RMS(98,102)**: Scatter plot showing feature relationships
- **Exponential Coefficients**: a vs b scatter plot
- **Feature Correlation Matrix**: Heatmap of feature correlations

#### 3. Data Preparation

```python
prepare_esp_data(data)
```

- Separates features and target variables
- Encodes categorical labels using LabelEncoder
- Splits data into training/testing sets (80/20 split)
- Applies StandardScaler for feature normalization
- Returns prepared datasets and feature names

#### 4. Model Training

```python
train_esp_models(X_train, X_test, y_train, y_test)
```

Trains four classification models:

- **Logistic Regression**: Linear baseline model
- **Random Forest**: Ensemble method with 100 trees
- **Gradient Boosting**: Sequential ensemble method
- **SVM**: Support Vector Machine with probability estimates

Each model is evaluated using:

- Training and testing accuracy
- 5-fold cross-validation scores
- Performance metrics storage

#### 5. Model Evaluation

```python
evaluate_esp_models(results, label_encoder)
```

Creates comprehensive evaluation visualizations:

- **Model Accuracy Comparison**: Bar chart with CV scores and error bars
- **Confusion Matrix**: Heatmap for best performing model
- **Feature Importance**: Horizontal bar chart (Random Forest)
- **Classification Report**: Detailed performance metrics

#### 6. Spectrum Analysis

```python
analyze_spectrum_data(spectrum_path, sample_size=100)
```

Analyzes frequency domain data:

- **Average Spectrum**: Mean amplitude across all samples
- **Sample Spectra**: Individual spectrum examples
- **Spectrum Statistics**: Mean and standard deviation plots
- **Amplitude Distribution**: Histogram of all amplitude values

## Key Features

### Improved Plot Visibility

- **Pie Chart Percentages**: White, bold text with proper positioning
- **Legend Placement**: Moved inside plots for better space utilization
- **Font Sizes**: Optimized for readability in embedded viewers
- **Constrained Layout**: Automatic spacing optimization
- **High DPI**: 300 DPI for crisp image output

### Machine Learning Pipeline

- **Stratified Splitting**: Maintains class distribution in train/test splits
- **Feature Scaling**: StandardScaler for consistent feature ranges
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Multiple Algorithms**: Comprehensive model comparison

### Real-World Application

- **Fault Diagnosis**: Classification of 6 different fault types
- **Feature Engineering**: Domain-specific vibration features
- **Scalable Analysis**: Handles large spectrum datasets efficiently
- **Production Ready**: Error handling and robust data loading

## Usage

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Analysis

```python
python esp_vibration_analysis.py
```

### Expected Outputs

1. **esp_data_analysis.png**: Initial data exploration plots
2. **esp_model_evaluation.png**: Model performance comparison
3. **esp_spectrum_analysis.png**: Frequency domain analysis

## Performance Metrics

### Model Evaluation

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-Validation**: 5-fold CV mean ± standard deviation

### Feature Importance

Random Forest feature importance ranking:

1. Most important features for fault classification
2. Feature contribution to model decisions
3. Engineering insights for sensor placement

## Technical Specifications

### Data Requirements

- **Input Format**: CSV files with semicolon separators
- **Memory Usage**: Optimized for large spectrum datasets
- **Processing**: Efficient pandas operations
- **Visualization**: Matplotlib with constrained layout

### Error Handling

- **File Not Found**: Graceful handling of missing data files
- **Data Validation**: Automatic data type checking
- **Memory Management**: Sample-based spectrum analysis
- **Robust Loading**: Exception handling for data corruption

## Future Enhancements

### Potential Improvements

1. **Deep Learning**: Neural networks for spectrum analysis
2. **Real-time Processing**: Streaming data analysis
3. **Advanced Features**: Time-frequency domain features
4. **Model Deployment**: Production-ready fault detection system

### Research Directions

1. **Transfer Learning**: Cross-ESP model adaptation
2. **Anomaly Detection**: Unsupervised fault identification
3. **Temporal Analysis**: Time-series fault progression
4. **Sensor Fusion**: Multi-sensor data integration

## References

- **ESPset Dataset**: [GitHub Repository](https://github.com/NINFA-UFES/ESPset)
- **Electric Submersible Pumps**: Offshore oil exploration equipment
- **Vibration Analysis**: Condition monitoring and fault diagnosis
- **Machine Learning**: Scikit-learn documentation and best practices

## Contact

For questions about this analysis or the ESPset dataset, please refer to the original dataset repository or contact the development team.

---

_This documentation is generated for the ESPset vibration analysis script. The analysis provides comprehensive insights into vibration-based fault diagnosis for Electric Submersible Pumps used in offshore oil exploration._

