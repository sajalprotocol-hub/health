import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import streamlit as st

def train_models(X_train, y_train, X_test, y_test):
    """
    Train machine learning models for disease prediction.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing labels
        
    Returns:
        tuple: (trained_models, feature_names)
            - trained_models: Dictionary of trained ML models
            - feature_names: List of feature names used by the models
    """
    # Get feature names (assuming they're in the column names of X_train if it's a dataframe)
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
    else:
        # If X_train is not a dataframe, use generic feature names
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # Use current time as a component for random state to vary predictions
    import time
    current_time_seed = int(time.time()) % 10000
    
    # Initialize models with dynamic random states
    rf_model = RandomForestClassifier(
        n_estimators=150,  # Increased from 100
        max_depth=12,      # Increased from 10
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=current_time_seed,  # Dynamic random state
        n_jobs=-1,
        class_weight='balanced'  # Added to improve minority class prediction
    )
    
    xgb_model = XGBClassifier(
        n_estimators=150,  # Increased from 100
        max_depth=7,       # Increased from 6
        learning_rate=0.05,  # Decreased for better generalization
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=current_time_seed,  # Dynamic random state
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # Train models
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    
    # Evaluate models
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    
    rf_accuracy = accuracy_score(y_test, rf_pred)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    
    # Print model metrics
    st.write("Model Performance Metrics:")
    st.write(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    st.write(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    
    # Create a dictionary of trained models
    trained_models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }
    
    return trained_models, feature_names

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models and return performance metrics.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing labels
        
    Returns:
        dict: Dictionary of evaluation metrics for each model
    """
    evaluation_results = {}
    
    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store metrics
        evaluation_results[model_name] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    return evaluation_results

def plot_model_comparison(models, X_test, y_test):
    """
    Create visualizations comparing model performance.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing labels
        
    Returns:
        matplotlib.figure.Figure: Figure containing performance comparison visualizations
    """
    # Get evaluation metrics
    eval_results = evaluate_models(models, X_test, y_test)
    
    # Create a figure for the comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy comparison
    model_names = list(eval_results.keys())
    accuracies = [eval_results[model]['accuracy'] for model in model_names]
    
    axes[0].bar(model_names, accuracies, color=['#4CAF50', '#2196F3'])
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, acc in enumerate(accuracies):
        axes[0].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    # Plot feature importances (using Random Forest)
    rf_model = models.get('Random Forest')
    if rf_model and hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        
        # If X_test is a dataframe, use its column names
        if isinstance(X_test, pd.DataFrame):
            feature_names = X_test.columns
        else:
            # Otherwise, create generic feature names
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Get the top N features
        top_n = min(10, len(feature_names))
        indices = np.argsort(importances)[-top_n:]
        
        axes[1].barh(range(top_n), importances[indices], color='#4CAF50')
        axes[1].set_yticks(range(top_n))
        axes[1].set_yticklabels([feature_names[i] for i in indices])
        axes[1].set_xlabel('Feature Importance')
        axes[1].set_title('Top Feature Importances')
        axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def create_ensemble_model(models, weights=None):
    """
    Create an ensemble model by combining multiple models with optional weights.
    
    Args:
        models (dict): Dictionary of trained models
        weights (dict, optional): Dictionary of model weights. If None, equal weights are used.
        
    Returns:
        object: An ensemble model object with a predict method
    """
    if weights is None:
        # Equal weights if not specified
        weights = {model_name: 1/len(models) for model_name in models.keys()}
    
    class EnsembleModel:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights
            
        def predict(self, X):
            # Get predictions from each model
            predictions = {}
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    # For models that provide probability estimates
                    predictions[model_name] = model.predict_proba(X)
                else:
                    # For models that don't provide probability estimates
                    predictions[model_name] = np.eye(len(np.unique(model.predict(X))))[model.predict(X)]
            
            # Combine predictions using weights
            weighted_preds = np.zeros_like(list(predictions.values())[0])
            for model_name, pred in predictions.items():
                weighted_preds += self.weights[model_name] * pred
            
            # Return the class with highest probability
            return np.argmax(weighted_preds, axis=1)
        
        def predict_proba(self, X):
            # Get probability predictions from each model
            predictions = {}
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    predictions[model_name] = model.predict_proba(X)
                else:
                    # For models that don't provide probability estimates, create a one-hot encoding
                    predictions[model_name] = np.eye(len(np.unique(model.predict(X))))[model.predict(X)]
            
            # Combine predictions using weights
            weighted_preds = np.zeros_like(list(predictions.values())[0])
            for model_name, pred in predictions.items():
                weighted_preds += self.weights[model_name] * pred
            
            return weighted_preds
    
    return EnsembleModel(models, weights)
