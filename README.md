# ❤️ Heart Disease Risk Prediction

A machine learning project that predicts the risk of heart disease using **Logistic Regression**, trained on the **UCI Cleveland Heart Disease dataset**. Built as a course project for **CSE366 (Artificial Intelligence)**.

## 📌 Overview

This project analyzes patient health data to predict the likelihood of heart disease. It includes:
- Data preprocessing and exploratory analysis on the UCI Cleveland dataset
- Logistic Regression model for classification
- **Genetic Algorithm (GA)-based feature selection** to identify the most relevant predictors
- An interactive **Streamlit web app** for real-time predictions

## 🗂️ Project Structure

```
heart_disease_prediction/
├── app.py                          # Streamlit web application
├── Heart_Disease_Training.ipynb    # Model training & experimentation notebook
├── heart_disease_model.pkl         # Trained Logistic Regression model
├── heart+disease/                  # Raw UCI dataset files
├── requirements.txt                # Python dependencies
├── run_app.bat                     # Windows script to launch the app
└── CSE366_Project_Report.docx      # Project report
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/Zahid074/heart-disease-prediction-model.git
cd heart-disease-prediction-model
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```
Or on Windows, simply double-click `run_app.bat`.

## 🧠 Model Details

- **Algorithm:** Logistic Regression
- **Dataset:** [UCI Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Feature Selection:** Genetic Algorithm (GA) used to optimize the feature subset for better accuracy
- **Output:** Predicted risk of heart disease based on patient clinical parameters

## 📊 Dataset

The dataset includes patient attributes such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, ECG results, max heart rate, and more — sourced from the UCI Machine Learning Repository.

## 👤 Author

**Md Zahid Hossain (Zahid074)**
CSE, East West University
[GitHub](https://github.com/Zahid074)
**Marjan Hasan (Marjan15H)**
CSE, East West University
[GitHub](https://github.com/Marjan15H)
**Marzia Hasan (Marzia-H)**
CSE, East West University
[GitHub](https://github.com/Marzia-H)
**Md Hasan Al Mamun**
CSE, East West University
**Sazzatul Islam**
CSE, East West University

## 📄 License

This project is for academic purposes as part of the CSE366 course.
