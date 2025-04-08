# 📉 Customer Churn Prediction

This project focuses on predicting whether a customer is likely to churn (leave the service) using various machine learning algorithms. By analyzing customer behavior and historical service data, the model helps identify at-risk customers so businesses can take proactive retention measures.

---

## 📚 Table of Contents

- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Data Preprocessing](#-data-preprocessing)
- [Machine Learning Models](#-machine-learning-models)
- [Training & Evaluation](#-training--evaluation)
- [Environment Setup](#-environment-setup)
- [File Structure](#-file-structure)
- [Conclusion](#-conclusion)
- [References](#-references)
- [Authors](#-authors)

---

## 🧠 Project Overview

Churn prediction helps companies identify which customers are at risk of leaving. This project uses a dataset of 7,043 customer records with 21 features, and evaluates four machine learning models:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Gaussian Mixture Models (GMM)

The goal is to determine the most suitable model for early churn prediction.

---

## 🛠️ Tech Stack

- Python 3.8+
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn

---

## 📂 Dataset

- **Source**: [Kaggle – Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Records**: 7,043
- **Features**: 21 attributes including demographics, services used, and billing info
- **Target Variable**: `Churn` (Yes/No)

---

## 🔄 Data Preprocessing

1. **Null & "No Service" Handling**: Replaced with appropriate values (e.g., "No").
2. **Encoding**: Used label and one-hot encoding for categorical variables.
3. **Scaling**: Applied `MinMaxScaler` to numerical features (`tenure`, `monthly_charges`, `total_charges`).
4. **Train-Test Split**: 75% training and 25% testing.
5. **EDA**: Explored churn distribution and feature relevance.

### ✨ Final Features After Encoding:

`gender_Male`, `senior_citizen_1`, `partner_Yes`, `dependents_Yes`, `phone_service_Yes`, `multiple_lines_Yes`, `internet_service_Fiber optic`, `internet_service_No`, `online_security_Yes`, `online_backup_Yes`, `device_protection_Yes`, `tech_support_Yes`, `streaming_tv_Yes`, `streaming_movies_Yes`, `contract_One year`, `contract_Two year`, `paperless_billing_Yes`, `payment_method_Credit card (automatic)`, `payment_method_Electronic check`, `payment_method_Mailed check`

---

## 🧪 Machine Learning Models

### 1. **Logistic Regression**
- Binary classification algorithm.
- Uses sigmoid activation for probability prediction.
- Optimized using cross-entropy loss.
- Best performance in this project.

### 2. **K-Nearest Neighbors (KNN)**
- Instance-based learning.
- Classifies based on majority class among nearest neighbors.
- Distance metrics like Euclidean used.

### 3. **Support Vector Machines (SVM)**
- Finds optimal hyperplane for classification.
- Works well with high-dimensional data.
- Uses margin maximization and kernel tricks.

### 4. **Gaussian Mixture Models (GMM)**
- Unsupervised clustering.
- Uses Expectation-Maximization algorithm.
- Assumes data is generated from Gaussian distributions.

---

## 🏋️ Training & Evaluation

### 🧠 Training Strategy

- All models were trained on the encoded dataset with 23 features.
- Evaluated using accuracy, precision, recall, and F1-score.

### 📊 Results Summary

| Model               | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|---------------------|--------------|----------------|-------------|---------------|
| Logistic Regression | 81.09        | 64.5           | 57.8        | 60.96         |
| SVM                 | 81.03        | 66.5           | 52.0        | 58.22         |
| KNN                 | 76.77        | 57.0           | 50.0        | 53.78         |
| GMM                 | 54.57        | Very Low       | Very Low    | Very Low      |

> ✅ **Best Model**: Logistic Regression (highest accuracy & balanced F1-score)

### 🔧 Improvement Suggestions

- **SMOTE** for balancing class distribution.
- **Regularization** (L1, L2) for better generalization.
- **Feature Engineering** to enhance feature set.
- **Cross-validation** to ensure model robustness.
- **Metric tuning in KNN** for better distance measures.

---

## 🧪 Environment Setup

### ⚙️ Python Version
- Python 3.8 or later

### 📦 Required Packages

You can install dependencies using:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
