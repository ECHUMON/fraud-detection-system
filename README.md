🔐 Financial Fraud Detection System (End-to-End Machine Learning Project)

A complete machine learning-based fraud detection system that trains multiple classification models, selects the best-performing model based on F1-score, and deploys it using a Streamlit web application for real-time fraud prediction.

🚀 Project Overview

Financial fraud detection is a critical problem in banking and digital transactions.
This system builds an end-to-end ML pipeline that:

Trains multiple machine learning models
Evaluates and compares performance
Selects the best model automatically
Deploys it in a real-time web application
Predicts fraud probability for financial transactions
🧠 System Architecture
Dataset (fraud.csv)
        ↓
Data Preprocessing
        ↓
Train-Test Split (Stratified)
        ↓
Model Training:
   - Logistic Regression
   - Decision Tree
   - Random Forest
        ↓
Model Evaluation (Accuracy, Precision, Recall, F1 Score)
        ↓
Best Model Selection (Based on F1 Score)
        ↓
Model Serialization (Pickle)
        ↓
Streamlit Web App
        ↓
Real-time Fraud Prediction
⚙️ Features
🧠 Machine Learning Pipeline
Multiple model training and comparison
Handles class imbalance using class_weight="balanced"
Automatic best model selection based on F1-score
Model persistence using Pickle
📊 Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
🌐 Web Application (Streamlit)
Real-time fraud prediction
Fraud probability scoring
Interactive transaction input system
Demo modes:
Manual input
Real legitimate transaction sample
Real fraudulent transaction sample
🧪 Models Used
Model	Type
Logistic Regression	Linear classifier
Decision Tree	Tree-based model
Random Forest	Ensemble model (Best Performer)
📊 Model Training Logic
Dataset split: 80/20 (stratified)
Evaluation metrics computed for each model
Best model selected using F1-score optimization
Final model saved as:
models/fraud_model.pkl
🖥️ Web Application Features

Built using Streamlit:

🔍 Real-time fraud prediction
📊 Fraud probability visualization
⚡ Fast inference using trained ML model
🧪 Multiple testing modes:
Manual transaction input
Real dataset samples (fraud & legitimate)
📁 Project Structure
fraud-detection-system/
│
├── app.py                  # Streamlit web application
├── model.py               # ML training + model selection
│
├── models/
│   └── fraud_model.pkl    # Saved trained model
│
├── data/
│   └── fraud.csv          # Dataset (not included in repo)
│
└── README.md
▶️ How to Run the Project
1. Install dependencies
pip install numpy pandas scikit-learn streamlit
2. Add dataset

Place dataset as:

data/fraud.csv
3. Train the model
python model.py
4. Run the web app
streamlit run app.py
🔍 How It Works
Model training script compares multiple ML algorithms
Best model is selected using F1-score
Model is saved using Pickle
Streamlit app loads trained model
User inputs transaction data
System predicts fraud probability
Output is classified:
≥ 30% → 🚨 Fraudulent Transaction
< 30% → ✅ Legitimate Transaction
📌 Key Highlights
End-to-end ML lifecycle implementation
Automated model selection pipeline
Handles imbalanced dataset using class weighting
Real-time deployment using Streamlit
Production-style model persistence
Probability-based fraud scoring system
⚠️ Limitations
No real-time streaming data ingestion
No API backend (Streamlit only)
Model trained on static dataset
🚀 Future Improvements
Add XGBoost / LightGBM for better accuracy
Deploy using FastAPI + Docker
Add SHAP explainability for model decisions
Add database logging system
Real-time transaction API integration
👨‍💻 Author

Nayan Padath Murali

Aspiring Cybersecurity & Machine Learning Engineer
Focused on building real-world AI + security systems.

🔐 Disclaimer

This project is for educational purposes only and should not be used in production financial systems without proper validation and regulatory compliance.
