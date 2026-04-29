# fraud-detection-system
(Machine learning-based financial fraud detection system using data preprocessing and classification models).
# 🔐 Financial Fraud Detection System (Machine Learning + Streamlit)

## Overview
This project is a machine learning-based system that detects fraudulent financial transactions. It compares multiple classification models and selects the best-performing model, which is then deployed in an interactive web application using Streamlit.

---

## 🚀 Features
- Data preprocessing and feature engineering  
- Multiple ML model training and comparison:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
- Best model selected based on F1 Score  
- Model saved using Pickle  
- Interactive Streamlit web application  
- Real-time fraud prediction with probability score  

---

## 🧠 Machine Learning Pipeline

### Dataset
- Credit Card Fraud Detection dataset  
- Target column: `Class`  
  - `0` → Legitimate transaction  
  - `1` → Fraudulent transaction  

⚠️ Note: The dataset file is not included in this repository due to large size. Users should download the dataset separately and place it in the `data/` folder as `fraud.csv`.

---

### Model Training (`model.py`)
- Train-test split (80/20, stratified)  
- Models used:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
- Evaluation metrics:
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
- Best model selected automatically based on F1 Score  
- Model saved as: models/fraud_model.pkl
==================================================================================================================================
- ## 🌐 Web Application (`app.py`)

Built using Streamlit, the application provides:

### Input Modes
- Manual transaction input (Time + Amount)  
- Real legitimate transaction (dataset sample)  
- Real fraudulent transaction (dataset sample)  

### Features
- Fraud probability prediction  
- Progress bar visualization  
- Clear output:
  - 🚨 Fraudulent Transaction  
  - ✅ Legitimate Transaction  
- Custom styled UI  

---

## 📊 Prediction Logic
- Model outputs fraud probability  
- Threshold:
  - ≥ 30% → Fraudulent  
  - < 30% → Legitimate  

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Pickle

- ==================================================================================================================

- ## 📁 Project Structure

fraud-detection-system/
│
├── app.py
├── model.py
├── README.md
│
├── models/
│ └── fraud_model.pkl
│
└── data/
└── fraud.csv (not included)

====================================================================================================================


---

## ▶️ How to Run

### 1. Install dependencies
pip install pandas numpy scikit-learn streamlit

### 2. Train the model
python model.py

### 3. Run the web app
streamlit run app.py

===================================================================================================================
---

## 📌 Key Highlights
- End-to-end ML pipeline  
- Model comparison and evaluation  
- Real-time fraud detection system  
- Interactive web interface  
- Professional project structure  

---

## 👨‍💻 Author
Nayan Padath Murali 
