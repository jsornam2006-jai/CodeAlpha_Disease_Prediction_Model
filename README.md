# 🫀 Disease Prediction Model

## 📌 Project Overview

The Disease Prediction Model is a Machine Learning project developed as part of the CodeAlpha Internship.

The project predicts the possibility of heart disease based on patient health-related attributes using classification algorithms.

The system compares multiple Machine Learning models and selects the best-performing model for prediction.

---

## 🎯 Objectives

- Predict the possibility of heart disease from patient data.
- Apply Machine Learning classification algorithms.
- Compare the performance of different models.
- Select the best-performing model.
- Build an interactive patient prediction system.

---

## 🧠 Machine Learning Algorithms

The following classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Model Performance

| Model | Accuracy |
|---|---:|
| Logistic Regression | 83.33% |
| Decision Tree | 70.00% |
| Random Forest | **85.00%** |

Random Forest achieved the highest accuracy of **85%** and was selected as the best-performing model.

---

## 📊 Dataset

The project uses the **UCI Cleveland Heart Disease dataset**.

The cleaned dataset contains:

- 297 patient records
- 13 input features
- 1 target variable

### Features

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- ST Depression
- Slope
- Number of Major Vessels
- Thalassemia

### Target

- `0` → No Heart Disease
- `1` → Heart Disease

---

## ⚙️ Methodology

The project follows these steps:

1. Dataset Collection
2. Data Cleaning
3. Data Preprocessing
4. Feature and Target Separation
5. Train-Test Split
6. Feature Scaling
7. Model Training
8. Model Evaluation
9. Best Model Selection
10. Patient Data Prediction

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pickle
- VS Code
- Git & GitHub

---

## 📁 Project Structure

```text
Disease_Prediction_Model/
│
├── dataset/
│   ├── heart_disease.csv
│   └── processed.cleveland.data
│
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── disease_prediction.py
├── predict_disease.py
├── .gitignore
└── README.md