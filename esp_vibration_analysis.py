"""
ESPset Vibration Analysis for Electric Submersible Pumps
========================================================

This script analyzes real-world vibration data from Electric Submersible Pumps (ESP)
used in offshore oil exploration. The ESPset dataset contains vibration signals and
engineered features for fault diagnosis.

Dataset Information:
- 6032 vibration signals from 8 different ESPs
- Features: median amplitudes, RMS values, peak amplitudes, exponential coefficients
- Labels: Normal, Unbalance, Misalignment, Bearing, Impeller, Cavitation
- Spectrum data: 6032 x 12103 matrix of frequency domain amplitudes

Author: Anuraj Ramesh
Date: 29-10-2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
from pathlib import Path
import os

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Make figures render cleanly inside embedded viewers
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["figure.constrained_layout.use"] = True


def load_esp_features(features_path="ESPset/features/features.csv"):
    """
    Load ESPset features data from CSV file.
    
    Parameters:
    features_path (str): Path to the features CSV file
    
    Returns:
    pandas.DataFrame: Loaded features data
    """
    print("Loading ESPset features data...")
    
    try:
        # Load the features data with semicolon separator
        data = pd.read_csv(features_path, sep=';')
        
        print(f"Successfully loaded {len(data)} samples with {len(data.columns)} features")
        print(f"Features: {list(data.columns)}")
        
        return data
        
    except FileNotFoundError:
        print(f"Error: Could not find features file at {features_path}")
        print("Please ensure the ESPset dataset is in the correct location.")
        return None
    except Exception as e:
        print(f"Error loading features data: {e}")
        return None


def explore_esp_data(data):
    """
    Explore the ESPset dataset to understand its structure and relationships.
    
    Parameters:
    data (pandas.DataFrame): The ESPset dataset to explore
    """
    print("\n" + "=" * 60)
    print("ESPSET DATASET EXPLORATION")
    print("=" * 60)
    
    # Basic information
    print("\n1. Dataset Shape:")
    print(f"   Samples: {data.shape[0]}")
    print(f"   Features: {data.shape[1]}")
    
    # Column information
    print("\n2. Column Information:")
    print(data.info())
    
    # First few rows
    print("\n3. First 5 samples:")
    print(data.head())
    
    # Statistical summary
    print("\n4. Statistical Summary:")
    print(data.describe())
    
    # Check for missing values
    print("\n5. Missing Values:")
    missing_values = data.isnull().sum()
    if missing_values.sum() == 0:
        print("   No missing values found!")
    else:
        print(missing_values[missing_values > 0])
    
    # Label distribution
    print("\n6. Label Distribution:")
    label_counts = data['label'].value_counts()
    print(label_counts)
    print(f"\nLabel percentages:")
    for label, count in label_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # ESP ID distribution
    print("\n7. ESP ID Distribution:")
    esp_counts = data['esp_id'].value_counts().sort_index()
    print(esp_counts)
    
    return label_counts, esp_counts


def visualize_esp_data(data):
    """
    Create visualizations to understand the ESPset data better.
    
    Parameters:
    data (pandas.DataFrame): The ESPset dataset to visualize
    """
    print("\n" + "=" * 60)
    print("CREATING ESPset DATA VISUALIZATIONS")
    print("=" * 60)
    
    # Set up plotting style
    plt.style.use("default")
    sns.set_palette("husl")
    sns.set_context("talk")
    
    # Create a comprehensive figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 14), constrained_layout=True)
    fig.suptitle("ESPset Vibration Data Analysis", fontsize=22, fontweight="bold")
    
    # 1. Label distribution
    label_counts = data['label'].value_counts()
    wedges, texts, autotexts = axes[0, 0].pie(label_counts.values, labels=label_counts.index, 
                                            autopct='%1.1f%%', startangle=90, 
                                            textprops={'fontsize': 11, 'fontweight': 'bold'},
                                            labeldistance=1.15,   # move labels slightly away from center
                                            pctdistance=0.75      # move percentage labels toward edge
                                            )
    # Make percentage text more visible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9.5)
        autotext.set_fontweight('bold')
    axes[0, 0].set_title("Fault Type Distribution", fontsize=16, fontweight="bold")
    
    # 2. ESP ID distribution
    esp_counts = data['esp_id'].value_counts().sort_index()
    axes[0, 1].bar(esp_counts.index, esp_counts.values, color='skyblue', alpha=0.8)
    axes[0, 1].set_title("Samples per ESP ID", fontsize=16, fontweight="bold")
    axes[0, 1].set_xlabel("ESP ID", fontsize=12)
    axes[0, 1].set_ylabel("Number of Samples", fontsize=12)
    axes[0, 1].tick_params(axis='both', labelsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Peak1x vs Peak2x colored by label
    for label in data['label'].unique():
        subset = data[data['label'] == label]
        axes[0, 2].scatter(subset['peak1x'], subset['peak2x'], 
                          label=label, alpha=0.6, s=15)
    axes[0, 2].set_title("Peak1x vs Peak2x by Fault Type", fontsize=16, fontweight="bold")
    axes[0, 2].set_xlabel("Peak1x (Amplitude at X)", fontsize=12)
    axes[0, 2].set_ylabel("Peak2x (Amplitude at 2X)", fontsize=12)
    axes[0, 2].tick_params(axis='both', labelsize=10)
    axes[0, 2].legend(loc='best', fontsize=10, framealpha=0.9)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Median(8,13) vs RMS(98,102) colored by label
    for label in data['label'].unique():
        subset = data[data['label'] == label]
        axes[1, 0].scatter(subset['median(8,13)'], subset['rms(98,102)'], 
                          label=label, alpha=0.6, s=15)
    axes[1, 0].set_title("Median(8,13) vs RMS(98,102) by Fault Type", fontsize=16, fontweight="bold")
    axes[1, 0].set_xlabel("Median(8,13)", fontsize=12)
    axes[1, 0].set_ylabel("RMS(98,102)", fontsize=12)
    axes[1, 0].tick_params(axis='both', labelsize=10)
    axes[1, 0].legend(loc='best', fontsize=10, framealpha=0.9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Exponential coefficients a vs b
    for label in data['label'].unique():
        subset = data[data['label'] == label]
        axes[1, 1].scatter(subset['a'], subset['b'], 
                          label=label, alpha=0.6, s=15)
    axes[1, 1].set_title("Exponential Coefficients a vs b", fontsize=16, fontweight="bold")
    axes[1, 1].set_xlabel("Coefficient a", fontsize=12)
    axes[1, 1].set_ylabel("Coefficient b", fontsize=12)
    axes[1, 1].tick_params(axis='both', labelsize=10)
    axes[1, 1].legend(loc='upper left', fontsize=10, framealpha=0.9)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Feature correlation heatmap
    feature_cols = ['median(8,13)', 'rms(98,102)', 'median(98,102)', 
                   'peak1x', 'peak2x', 'a', 'b']
    correlation_matrix = data[feature_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0,
                square=False, ax=axes[1, 2], cbar_kws={"shrink": 0.8},
                annot_kws={"size": 10}, fmt='.2f')
    axes[1, 2].set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold")
    axes[1, 2].tick_params(axis='both', labelsize=10)
    
    plt.savefig("esp_data_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print("ESPset visualizations saved as 'esp_data_analysis.png'")


def prepare_esp_data(data):
    """
    Prepare the ESPset data for machine learning classification.
    
    Parameters:
    data (pandas.DataFrame): The ESPset dataset to prepare
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test, feature_names, label_encoder)
    """
    print("\n" + "=" * 60)
    print("PREPARING ESPset DATA FOR MACHINE LEARNING")
    print("=" * 60)
    
    # Define feature columns (excluding id, esp_id, and label)
    feature_columns = ['median(8,13)', 'rms(98,102)', 'median(98,102)', 
                      'peak1x', 'peak2x', 'a', 'b']
    
    X = data[feature_columns] 
    y = data['label']
    
    print(f"Features: {feature_columns}")
    print(f"Target: label (fault classification)")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Classes: {label_encoder.classes_}")
    print(f"Encoded classes: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features have been scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns, label_encoder


def train_esp_models(X_train, X_test, y_train, y_test):
    """
    Train multiple classification models for ESP fault diagnosis.
    
    Parameters:
    X_train, X_test: Training and testing feature matrices
    y_train, y_test: Training and testing target vectors
    
    Returns:
    dict: Dictionary containing model results
    """
    print("\n" + "=" * 60)
    print("TRAINING ESP FAULT CLASSIFICATION MODELS")
    print("=" * 60)
    
    # Define models to try
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Store results
        results[name] = {
            "model": model,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred_test": y_pred_test,
            "y_test": y_test
        }
        
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Testing Accuracy: {test_accuracy:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results


def save_trained_model(results, label_encoder, scaler, feature_names, model_dir="models"):
    """
    Save the trained model and preprocessing objects for later use.
    
    Parameters:
    results: Dictionary containing model results
    label_encoder: LabelEncoder used for encoding classes
    scaler: StandardScaler used for feature scaling
    feature_names: List of feature names
    model_dir: Directory to save the model files
    """
    print("\n" + "=" * 60)
    print("SAVING TRAINED MODEL")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Get the best model
    best_model_name = max(results.keys(), key=lambda x: results[x]["test_accuracy"])
    best_model = results[best_model_name]["model"]
    
    print(f"Saving best model: {best_model_name}")
    print(f"Model accuracy: {results[best_model_name]['test_accuracy']:.4f}")
    
    # Save the best model
    model_path = os.path.join(model_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save the label encoder
    encoder_path = os.path.join(model_dir, "label_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)
    print(f"Label encoder saved to: {encoder_path}")
    
    # Save the scaler
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save feature names
    features_path = os.path.join(model_dir, "feature_names.joblib")
    joblib.dump(feature_names, features_path)
    print(f"Feature names saved to: {features_path}")
    
    # Save model metadata
    metadata = {
        "model_name": best_model_name,
        "accuracy": results[best_model_name]["test_accuracy"],
        "cv_mean": results[best_model_name]["cv_mean"],
        "cv_std": results[best_model_name]["cv_std"],
        "feature_names": feature_names,
        "classes": label_encoder.classes_.tolist()
    }
    
    metadata_path = os.path.join(model_dir, "model_metadata.joblib")
    joblib.dump(metadata, metadata_path)
    print(f"Model metadata saved to: {metadata_path}")
    
    print("All model files saved successfully!")
    return model_path, encoder_path, scaler_path, features_path, metadata_path


def evaluate_esp_models(results, label_encoder):
    """
    Create visualizations to compare ESP model performance.
    
    Parameters:
    results: Dictionary containing model results
    label_encoder: LabelEncoder used for encoding classes
    """
    print("\n" + "=" * 60)
    print("ESP MODEL EVALUATION AND COMPARISON")
    print("=" * 60)
    
    # Create comparison plots with better spacing
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
    fig.suptitle("ESP Fault Classification Model Performance", fontsize=22, fontweight="bold")
    
    # 1. Accuracy comparison
    model_names = list(results.keys())
    test_accuracies = [results[name]["test_accuracy"] for name in model_names]
    cv_means = [results[name]["cv_mean"] for name in model_names]
    cv_stds = [results[name]["cv_std"] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    bars = axes[0, 0].bar(x_pos, test_accuracies, color=["skyblue", "lightgreen", "salmon", "gold"], alpha=0.8)
    axes[0, 0].errorbar(x_pos, cv_means, yerr=cv_stds, fmt='ro', capsize=5, label='CV Score', markersize=8)
    axes[0, 0].set_title("Model Accuracy Comparison", fontsize=18, fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy", fontsize=14)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names, rotation=30, ha='right', fontsize=12)
    axes[0, 0].tick_params(axis='y', labelsize=12)
    axes[0, 0].legend(fontsize=12, loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, test_accuracies)):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Confusion matrix for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]["test_accuracy"])
    best_predictions = results[best_model_name]["y_pred_test"]
    best_y_test = results[best_model_name]["y_test"]
    
    cm = confusion_matrix(best_y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=axes[0, 1], annot_kws={"size": 12, "fontweight": "bold"})
    axes[0, 1].set_title(f"Confusion Matrix ({best_model_name})", fontsize=18, fontweight="bold")
    axes[0, 1].set_xlabel("Predicted", fontsize=14)
    axes[0, 1].set_ylabel("Actual", fontsize=14)
    axes[0, 1].tick_params(axis='both', labelsize=11)
    
    # 3. Feature importance (for tree-based models)
    if "Random Forest" in results:
        rf_model = results["Random Forest"]["model"]
        feature_importance = rf_model.feature_importances_
        feature_names = ['median(8,13)', 'rms(98,102)', 'median(98,102)', 
                        'peak1x', 'peak2x', 'a', 'b']
        
        importance_df = pd.DataFrame({
            "feature": feature_names, 
            "importance": feature_importance
        }).sort_values("importance", ascending=True)
        
        bars = axes[1, 0].barh(importance_df["feature"], importance_df["importance"], 
                              color="lightcoral", alpha=0.8)
        axes[1, 0].set_title("Feature Importance (Random Forest)", fontsize=18, fontweight="bold")
        axes[1, 0].set_xlabel("Importance Score", fontsize=14)
        axes[1, 0].tick_params(axis='both', labelsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars, importance_df["importance"])):
            axes[1, 0].text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
                           f'{imp:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')
    
    # 4. Classification report as text
    axes[1, 1].axis('off')
    report = classification_report(best_y_test, best_predictions, 
                                 target_names=label_encoder.classes_,
                                 output_dict=True)
    
    # Create a text representation of the classification report
    report_text = f"Classification Report ({best_model_name})\n\n"
    report_text += f"Overall Accuracy: {results[best_model_name]['test_accuracy']:.4f}\n\n"
    
    for class_name in label_encoder.classes_:
        if class_name in report:
            metrics = report[class_name]
            report_text += f"{class_name}:\n"
            report_text += f"  Precision: {metrics['precision']:.3f}\n"
            report_text += f"  Recall: {metrics['recall']:.3f}\n"
            report_text += f"  F1-Score: {metrics['f1-score']:.3f}\n\n"
    
    axes[1, 1].text(0.05, 0.95, report_text, transform=axes[1, 1].transAxes, 
                   fontsize=12, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
    
    plt.savefig("esp_model_evaluation.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"Best performing model: {best_model_name}")
    print(f"Best Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
    print("ESP model evaluation plots saved as 'esp_model_evaluation.png'")


def analyze_spectrum_data(spectrum_path="ESPset/spectrum/spectrum.csv", sample_size=100):
    """
    Analyze a sample of the spectrum data to understand frequency characteristics.
    
    Parameters:
    spectrum_path (str): Path to the spectrum CSV file
    sample_size (int): Number of samples to analyze (due to large file size)
    """
    print("\n" + "=" * 60)
    print("ESP SPECTRUM DATA ANALYSIS")
    print("=" * 60)
    
    try:
        print(f"Loading spectrum data sample ({sample_size} samples)...")
        
        # Read only a sample of the spectrum data
        spectrum_data = pd.read_csv(spectrum_path, sep=';', nrows=sample_size)
        
        print(f"Loaded spectrum data: {spectrum_data.shape}")
        print(f"Frequency bins: {spectrum_data.shape[1]}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
        fig.suptitle("ESP Spectrum Data Analysis", fontsize=20, fontweight="bold")
        
        # 1. Average spectrum
        avg_spectrum = spectrum_data.mean(axis=0)
        axes[0, 0].plot(avg_spectrum.values, linewidth=2)
        axes[0, 0].set_title("Average Spectrum", fontsize=16, fontweight="bold")
        axes[0, 0].set_xlabel("Frequency Bin", fontsize=12)
        axes[0, 0].set_ylabel("Amplitude (in/s)", fontsize=12)
        axes[0, 0].tick_params(axis='both', labelsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Sample spectra
        for i in range(min(5, len(spectrum_data))):
            axes[0, 1].plot(spectrum_data.iloc[i].values, alpha=0.7, label=f'Sample {i}', linewidth=1.5)
        axes[0, 1].set_title("Sample Spectra", fontsize=16, fontweight="bold")
        axes[0, 1].set_xlabel("Frequency Bin", fontsize=12)
        axes[0, 1].set_ylabel("Amplitude (in/s)", fontsize=12)
        axes[0, 1].tick_params(axis='both', labelsize=10)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Spectrum statistics
        spectrum_stats = spectrum_data.describe()
        axes[1, 0].plot(spectrum_stats.loc['mean'].values, label='Mean', linewidth=2)
        axes[1, 0].plot(spectrum_stats.loc['std'].values, label='Std Dev', linewidth=2)
        axes[1, 0].set_title("Spectrum Statistics", fontsize=16, fontweight="bold")
        axes[1, 0].set_xlabel("Frequency Bin", fontsize=12)
        axes[1, 0].set_ylabel("Amplitude (in/s)", fontsize=12)
        axes[1, 0].tick_params(axis='both', labelsize=10)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Frequency distribution
        all_amplitudes = spectrum_data.values.flatten()
        axes[1, 1].hist(all_amplitudes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title("Amplitude Distribution", fontsize=16, fontweight="bold")
        axes[1, 1].set_xlabel("Amplitude (in/s)", fontsize=12)
        axes[1, 1].set_ylabel("Frequency", fontsize=12)
        axes[1, 1].tick_params(axis='both', labelsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.savefig("esp_spectrum_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        print("ESP spectrum analysis saved as 'esp_spectrum_analysis.png'")
        
    except FileNotFoundError:
        print(f"Spectrum file not found at {spectrum_path}")
        print("Skipping spectrum analysis...")
    except Exception as e:
        print(f"Error analyzing spectrum data: {e}")


def main():
    """
    Main function to run the complete ESPset vibration analysis.
    """
    print("ESPSET VIBRATION FAULT DIAGNOSIS ANALYSIS")
    print("=" * 60)
    print("Analyzing real-world vibration data from Electric Submersible Pumps")
    print("Used in offshore oil exploration and extraction")
    print("=" * 60)
    
    # Step 1: Load ESPset features data
    data = load_esp_features()
    if data is None:
        print("Failed to load data. Exiting...")
        return
    
    # Step 2: Explore the data
    label_counts, esp_counts = explore_esp_data(data)
    
    # Step 3: Visualize the data
    visualize_esp_data(data)
    
    # Step 4: Prepare data for machine learning
    X_train, X_test, y_train, y_test, feature_names, label_encoder = prepare_esp_data(data)
    
    # Step 5: Train classification models
    results = train_esp_models(X_train, X_test, y_train, y_test)
    
    # Step 6: Save the trained model
    # Get the scaler from the prepare_data function (we need to recreate it)
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on the original training data
    save_trained_model(results, label_encoder, scaler, feature_names)
    
    # Step 7: Evaluate and compare models
    evaluate_esp_models(results, label_encoder)
    
    # Step 8: Analyze spectrum data (optional, due to large file size)
    analyze_spectrum_data()
    
    print("\n" + "=" * 60)
    print("ESPset ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Generated files:")
    print("  - esp_data_analysis.png (data visualizations)")
    print("  - esp_model_evaluation.png (model performance comparison)")
    print("  - esp_spectrum_analysis.png (spectrum data analysis)")
    print("\nKey Findings:")
    print(f"  - Total samples: {len(data)}")
    print(f"  - Fault types: {len(label_counts)}")
    print(f"  - ESP units: {len(esp_counts)}")
    print(f"  - Features: {len(feature_names)}")
    print("\nNext steps:")
    print("  1. Experiment with different feature engineering approaches")
    print("  2. Try deep learning models for spectrum data")
    print("  3. Implement real-time fault detection system")
    print("  4. Analyze temporal patterns in vibration signals")




if __name__ == "__main__":
    main()
