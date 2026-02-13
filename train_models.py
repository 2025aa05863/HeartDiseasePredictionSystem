"""
ML Assignment 2 - Model Training Script
BITS Pilani MTech (AIML/DSE)

This script trains 6 classification models on the Heart Disease dataset:
1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             confusion_matrix, classification_report)
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load the Combined Heart Disease dataset from UCI repository
    Combines data from 4 locations: Cleveland, Hungarian, Switzerland, Long Beach VA
    Total: 920 instances (meets minimum requirement of 500)
    """
    # Heart Disease Dataset from UCI - All 4 locations combined
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
    ]
    
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Try to load data with SSL verification disabled
    import ssl
    import urllib.request
    
    all_dataframes = []
    
    for i, url in enumerate(urls):
        location_names = ['Cleveland', 'Hungarian', 'Switzerland', 'Long Beach VA']
        print(f"   Loading {location_names[i]} data...")
        
        try:
            # First try normal load
            df_temp = pd.read_csv(url, names=column_names, na_values='?')
            all_dataframes.append(df_temp)
            print(f"   ✓ Loaded {len(df_temp)} records from {location_names[i]}")
        except Exception as e:
            # If SSL error, try with SSL context
            try:
                print(f"   Trying alternative method for {location_names[i]}...")
                ssl_context = ssl._create_unverified_context()
                with urllib.request.urlopen(url, context=ssl_context) as response:
                    df_temp = pd.read_csv(response, names=column_names, na_values='?')
                    all_dataframes.append(df_temp)
                    print(f"   ✓ Loaded {len(df_temp)} records from {location_names[i]}")
            except Exception as e2:
                print(f"   ✗ Failed to load {location_names[i]}: {e2}")
                continue
    
    # Combine all datasets
    df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n   Combined dataset: {len(df)} total records")
    
    # Handle missing values - Use imputation instead of dropping
    from sklearn.impute import SimpleImputer
    
    print(f"   Rows before handling missing values: {len(df)}")
    print(f"   Missing values per column:\n{df.isnull().sum()}")
    
    # For numeric columns with missing values, use median imputation
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    # Drop rows only if all values are missing (very rare)
    df = df.dropna(how='all')
    
    print(f"   Rows after handling missing values: {len(df)}")
    
    # Convert target to binary (0: no disease, 1: disease)
    df['target'] = (df['target'] > 0).astype(int)
    
    return df

def evaluate_model(y_true, y_pred, y_pred_proba):
    """
    Calculate all evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all 6 models and save them
    """
    models_config = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    
    results = {}
    
    for model_name, model in models_config.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Save model
        model_filename = f"model/{model_name.replace(' ', '_').lower()}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        # Store results
        results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Print metrics
        print(f"\nMetrics for {model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  MCC:       {metrics['mcc']:.4f}")
    
    return results

def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("ML Assignment 2 - Model Training")
    print("BITS Pilani MTech (AIML/DSE)")
    print("="*60)
    
    # Load data
    print("\n1. Loading Heart Disease dataset...")
    df = load_and_prepare_data()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Target distribution:\n{df['target'].value_counts()}")
    
    # Save sample data for Streamlit testing
    test_sample = df.sample(n=min(50, len(df)), random_state=42)
    test_sample.to_csv('data/test_data.csv', index=False)
    print(f"\n   Saved test sample: data/test_data.csv")
    
    # Prepare features and target
    print("\n2. Preparing features and target...")
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"   Training set size: {X_train.shape[0]}")
    print(f"   Test set size: {X_test.shape[0]}")
    
    # Train models
    print("\n3. Training all models...")
    results = train_all_models(X_train, X_test, y_train, y_test)
    
    # Save results
    with open('model/results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for model_name, data in results.items():
            json_results[model_name] = {
                'metrics': {k: float(v) for k, v in data['metrics'].items()},
                'confusion_matrix': data['confusion_matrix'],
                'classification_report': data['classification_report']
            }
        json.dump(json_results, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nSummary of Results:")
    print("-" * 90)
    print(f"{'Model':<25} {'Accuracy':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}")
    print("-" * 90)
    for model_name, data in results.items():
        m = data['metrics']
        print(f"{model_name:<25} {m['accuracy']:<10.4f} {m['auc']:<10.4f} "
              f"{m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {m['mcc']:<10.4f}")
    print("-" * 90)
    
    print("\n✓ All models saved in 'model/' directory")
    print("✓ Results saved in 'model/results.json'")
    print("✓ Test data saved in 'data/test_data.csv'")
    print("\nNext step: Run the Streamlit app using 'streamlit run app.py'")

if __name__ == "__main__":
    main()
