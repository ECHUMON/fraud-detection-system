# 🔐 Financial Fraud Detection System (End-to-End Machine Learning Project)

A complete machine learning-based fraud detection system that trains multiple classification models, selects the best-performing model based on F1-score, and deploys it using a Streamlit web application for real-time fraud prediction.

---

## 🚀 Project Overview

Financial fraud detection is a critical problem in banking and digital transactions.  
This system builds an end-to-end ML pipeline that:

- Trains multiple machine learning models  
- Evaluates and compares performance  
- Selects the best model automatically  
- Deploys it in a real-time web application  
- Predicts fraud probability for financial transactions  

---

## 🧠 System Architecture

```text
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
