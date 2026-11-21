# 🎓 Student Performance Prediction Using KNN

## 📌 Project Overview

This is a Machine Learning project that predicts student academic performance using the K-Nearest Neighbors (KNN) classification algorithm.

The model predicts whether a student will Pass or Fail based on three important features:

- Study Hours
- Attendance
- Previous Marks

## 🎯 Project Objectives

- Understand and implement the KNN algorithm.
- Explore and analyze student performance data.
- Train a Machine Learning classification model.
- Predict student results using new input data.
- Evaluate model performance using accuracy.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- NumPy
- Joblib
- Streamlit

## 📊 Dataset Features

| Feature | Description |
|---|---|
| Hours | Student study hours |
| Attendance | Student attendance percentage |
| PrevMarks | Previous examination marks |
| Result | Pass or Fail (Target) |

## ⚙️ Machine Learning Workflow

1. Dataset Loading
2. Data Exploration (EDA)
3. Feature Selection
4. Train-Test Split
5. Feature Scaling using StandardScaler
6. KNN Model Training (K=3)
7. Prediction
8. Model Evaluation
9. Model Saving using Joblib

## 📈 Model Performance

**Model:** K-Nearest Neighbors (KNN)

**K Value:** 3

**Model Accuracy:** 95.45%

**Classification Type:** Binary Classification

**Target Classes:** Pass / Fail

## 📁 Project Structure

```text
Student-Performance-Prediction-KNN/
│
├── project1.csv
├── KNN_algorithm.py
├── KNN_model.pkl
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Install Required Libraries

```bash
pip install pandas scikit-learn joblib streamlit
```

### 3. Run the Machine Learning Model

```bash
python KNN_algorithm.py
```

### 4. Run the Streamlit Dashboard

```bash
python -m streamlit run app.py
```

## 🔮 Future Improvements

- Improve model performance.
- Add interactive data visualizations.
- Build a professional Streamlit dashboard.
- Add more
