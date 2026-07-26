# 🫀 Heart Disease Prediction System

A Machine Learning web application that predicts the likelihood of heart disease based on patient health information.

The project covers the complete ML workflow — from data preprocessing and model comparison to building an interactive web application using Streamlit.

## 🚀 Project Overview

The goal of this project is to build a binary classification model for predicting heart disease.

The dataset contains health-related features such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

Target variable:

- `0` → No Heart Disease
- `1` → Heart Disease

## 🧠 Machine Learning Workflow

The project includes:

- Data exploration and cleaning
- Categorical feature encoding
- Feature analysis using Pearson correlation
- Train-test splitting
- Feature scaling using StandardScaler
- Training multiple classification algorithms
- Model evaluation using Accuracy and F1 Score
- Saving the trained model using Joblib
- Building an interactive Streamlit interface

## 🤖 Models Tested

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | **86.96%** | **88.57%** |
| K-Nearest Neighbors | 86.41% | 88.15% |
| Naive Bayes | 85.33% | 86.83% |
| Support Vector Machine | 84.78% | 86.79% |
| Decision Tree | 78.26% | 80.20% |

Based on the evaluation results, **Logistic Regression** was selected as the final model.

## 🖥️ Streamlit Web App

The project includes an interactive Streamlit application where users can enter health-related information and receive the model's prediction.

The application:

1. Collects user input
2. Converts categorical inputs into the required encoded format
3. Matches the feature structure used during training
4. Applies the saved StandardScaler
5. Sends the processed input to the trained model
6. Displays the predicted risk category

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## 📂 Project Structure

```text
Heart-Disease-Prediction/
│
├── app.py
├── le_heart.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
├── README.md
└── notebook.ipynb
