# 🔥 Customer Churn Prediction using ANN (Keras & TensorFlow)

## 📌 Project Overview
This project predicts whether a customer will leave the bank (churn) or stay using an Artificial Neural Network (ANN).  
The model is built using Keras and TensorFlow.

---

## 🎯 Objective
- Predict customer churn (0 = No churn, 1 = Churn)
- Learn deep learning workflow (EDA → preprocessing → ANN → evaluation)
- Understand backpropagation and model optimization

---

## 📊 Dataset Information
- Dataset: Churn Modelling Dataset
- Total records: ~10,000 customers
- Target variable: `Exited`

### Features:
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

---

## 🧹 Data Preprocessing
- Removed irrelevant columns (RowNumber, CustomerId, Surname)
- Encoded categorical variables:
  - Gender → Label Encoding
  - Geography → One Hot Encoding
- Feature Scaling using StandardScaler
- Train-test split (80-20)

---

## 🧠 Model Architecture (ANN)

```python
model = Sequential()

model.add(Dense(11, activation='relu', input_dim=11))  # Hidden Layer
model.add(Dense(1, activation='sigmoid'))             # Output Layer
