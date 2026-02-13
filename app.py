"""
Heart Disease Prediction - Interactive ML App
BITS Pilani MTech (AIML/DSE) - Machine Learning Assignment 2

Author: Student Name
Date: January 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #FF4B4B;
        padding-bottom: 1rem;
    }
    h2 {
        color: #262730;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_model(model_name):
    """Load a trained model from disk"""
    model_file = f"model/{model_name.replace(' ', '_').lower()}_model.pkl"
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_file}")
        return None

@st.cache_data
def load_scaler():
    """Load the fitted scaler"""
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except FileNotFoundError:
        st.error("Scaler file not found. Please train models first.")
        return None

@st.cache_data
def load_results():
    """Load pre-computed results"""
    try:
        with open('model/results.json', 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        return None

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix using seaborn"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def display_metrics(metrics):
    """Display metrics in a nice format"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        st.metric("Precision", f"{metrics['Precision']:.4f}")
    
    with col2:
        st.metric("AUC", f"{metrics['AUC']:.4f}")
        st.metric("Recall", f"{metrics['Recall']:.4f}")
    
    with col3:
        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        st.metric("MCC", f"{metrics['MCC']:.4f}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("❤️ Heart Disease Prediction System")
    st.markdown("### Machine Learning Assignment 2 - BITS Pilani MTech (AIML/DSE)")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 Configuration")
    st.sidebar.markdown("### Select Model")
    
    # Model selection dropdown
    model_options = [
        'Logistic Regression',
        'Decision Tree',
        'kNN',
        'Naive Bayes',
        'Random Forest',
        'XGBoost'
    ]
    
    selected_model = st.sidebar.selectbox(
        "Choose a classification model:",
        model_options,
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Information")
    
    model_descriptions = {
        'Logistic Regression': 'Linear model for binary classification',
        'Decision Tree': 'Tree-based model with interpretable rules',
        'kNN': 'Instance-based learning algorithm',
        'Naive Bayes': 'Probabilistic classifier based on Bayes theorem',
        'Random Forest': 'Ensemble of decision trees',
        'XGBoost': 'Gradient boosting ensemble method'
    }
    
    st.sidebar.info(model_descriptions[selected_model])
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📈 Model Performance", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Test Dataset")
        st.markdown("Upload a CSV file with heart disease patient data for prediction.")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with patient features"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
                
                # Display first few rows
                with st.expander("📋 View Dataset Preview"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Check if target column exists
                has_target = 'target' in df.columns
                
                if has_target:
                    X = df.drop('target', axis=1)
                    y = df['target']
                else:
                    X = df
                    y = None
                
                # Load model and scaler
                model = load_model(selected_model)
                scaler = load_scaler()
                
                if model is not None and scaler is not None:
                    st.markdown("---")
                    st.subheader(f"🔮 Predictions using {selected_model}")
                    
                    # Preprocess and predict
                    X_scaled = scaler.transform(X)
                    predictions = model.predict(X_scaled)
                    prediction_proba = model.predict_proba(X_scaled)[:, 1]
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Prediction': ['Disease' if p == 1 else 'No Disease' for p in predictions],
                        'Probability': prediction_proba
                    })
                    
                    # Display predictions
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("#### Prediction Results")
                        display_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
                        st.dataframe(display_df, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Summary Statistics")
                        disease_count = (predictions == 1).sum()
                        no_disease_count = (predictions == 0).sum()
                        
                        st.metric("Total Samples", len(predictions))
                        st.metric("Disease Predicted", f"{disease_count} ({disease_count/len(predictions)*100:.1f}%)")
                        st.metric("No Disease Predicted", f"{no_disease_count} ({no_disease_count/len(predictions)*100:.1f}%)")
                    
                    # If target exists, show evaluation metrics
                    if has_target:
                        st.markdown("---")
                        st.subheader("📊 Model Evaluation Metrics")
                        
                        metrics = calculate_metrics(y, predictions, prediction_proba)
                        display_metrics(metrics)
                        
                        # Confusion Matrix
                        st.markdown("---")
                        st.subheader("🔢 Confusion Matrix")
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            cm = confusion_matrix(y, predictions)
                            fig = plot_confusion_matrix(cm, selected_model)
                            st.pyplot(fig)
                        
                        with col2:
                            st.markdown("#### Classification Report")
                            report = classification_report(y, predictions, 
                                                          target_names=['No Disease', 'Disease'])
                            st.text(report)
                        
                    # Download predictions
                    st.markdown("---")
                    csv = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"predictions_{selected_model.replace(' ', '_').lower()}.csv",
                        mime="text/csv",
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your CSV has the correct columns and format.")
        
        else:
            st.info("👆 Please upload a CSV file to begin prediction.")
            st.markdown("#### Sample Data Available")
            st.markdown("You can test the app using the provided test data: `data/test_data.csv`")
    
    with tab2:
        st.header("📈 Model Performance Comparison")
        
        # Load pre-computed results
        results = load_results()
        
        if results:
            st.markdown("### Comparison of All 6 Models")
            
            # Create comparison table
            comparison_data = []
            for model_name, data in results.items():
                metrics = data['metrics']
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': f"{metrics['accuracy']:.4f}",
                    'AUC': f"{metrics['auc']:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1': f"{metrics['f1']:.4f}",
                    'MCC': f"{metrics['mcc']:.4f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display table
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Visualizations
            st.markdown("---")
            st.subheader("📊 Visual Comparison")
            
            # Prepare data for plotting
            metrics_to_plot = ['accuracy', 'auc', 'precision', 'recall', 'f1', 'mcc']
            model_names = list(results.keys())
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for idx, metric in enumerate(metrics_to_plot):
                values = [results[model]['metrics'][metric] for model in model_names]
                
                axes[idx].bar(range(len(model_names)), values, color='steelblue', alpha=0.7)
                axes[idx].set_xlabel('Model', fontsize=10)
                axes[idx].set_ylabel(metric.upper(), fontsize=10)
                axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
                axes[idx].set_xticks(range(len(model_names)))
                axes[idx].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
                axes[idx].set_ylim([0, 1.0])
                axes[idx].grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for i, v in enumerate(values):
                    axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show best model
            st.markdown("---")
            st.subheader("🏆 Best Performing Model")
            
            # Find best model by accuracy
            best_model = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])
            st.success(f"**{best_model[0]}** achieved the highest accuracy: {best_model[1]['metrics']['accuracy']:.4f}")
            
        else:
            st.warning("⚠️ Model results not found. Please train the models first by running `python train_models.py`")
    
    with tab3:
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ### Heart Disease Prediction System
        
        This application uses machine learning to predict the presence of heart disease in patients
        based on various clinical features.
        
        #### Dataset: Heart Disease (Combined - 4 Locations)
        - **Source**: UCI Machine Learning Repository
        - **Features**: 13 clinical attributes
        - **Target**: Binary classification (Disease/No Disease)
        - **Samples**: 920 patients (Cleveland, Hungarian, Switzerland, Long Beach VA)
        - **After cleaning**: 500+ patients (meets assignment requirements)
        
        #### Features Used:
        1. **age**: Age in years
        2. **sex**: Sex (1 = male; 0 = female)
        3. **cp**: Chest pain type (0-3)
        4. **trestbps**: Resting blood pressure (mm Hg)
        5. **chol**: Serum cholesterol (mg/dl)
        6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        7. **restecg**: Resting ECG results (0-2)
        8. **thalach**: Maximum heart rate achieved
        9. **exang**: Exercise induced angina (1 = yes; 0 = no)
        10. **oldpeak**: ST depression induced by exercise
        11. **slope**: Slope of peak exercise ST segment (0-2)
        12. **ca**: Number of major vessels colored by fluoroscopy (0-3)
        13. **thal**: Thalassemia (0-3)
        
        #### Models Implemented:
        1. **Logistic Regression** - Linear classification baseline
        2. **Decision Tree** - Interpretable tree-based model
        3. **k-Nearest Neighbors (kNN)** - Instance-based learning
        4. **Naive Bayes** - Probabilistic classifier
        5. **Random Forest** - Ensemble of decision trees
        6. **XGBoost** - Gradient boosting ensemble
        
        #### Evaluation Metrics:
        - **Accuracy**: Overall correctness
        - **AUC**: Area under ROC curve
        - **Precision**: Positive predictive value
        - **Recall**: Sensitivity
        - **F1 Score**: Harmonic mean of precision and recall
        - **MCC**: Matthews Correlation Coefficient
        
        #### Technology Stack:
        - **Frontend**: Streamlit
        - **ML Framework**: Scikit-learn, XGBoost
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Matplotlib, Seaborn
        
        ---
        
        **Assignment**: Machine Learning Assignment 2  
        **Course**: MTech (AIML/DSE)  
        **Institution**: BITS Pilani  
        **Date**: January 2026
        
        ---
        
        #### 📞 Support
        For issues or questions, please refer to the GitHub repository.
        """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<p style='text-align: center; color: gray;'>"
            "Made with ❤️ using Streamlit | BITS Pilani MTech (AIML/DSE)"
            "</p>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
