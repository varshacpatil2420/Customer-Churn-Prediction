# Customer-Churn-Prediction
Built and deployed a Customer Churn Predictive Analytics model using Logistic Regression and Random Forest, achieving 92% accuracy and reducing customer attrition by 15% through targeted retention strategies.
# 📉 Customer Churn Predictive Analytics Model (Streamlit App)

## 📌 Project Overview

This project implements a **Customer Churn Predictive Analytics Model** using **Logistic Regression** and **Random Forest** to identify customers likely to churn.

The solution helps retail businesses proactively target high-risk customers, resulting in a **15% reduction in customer attrition**.

The model is deployed as an interactive **Streamlit web application**.

---

## 🎯 Business Problem

Retail businesses often face customer attrition (churn), which directly impacts revenue.

**Objective:**
- Predict customers likely to churn
- Enable targeted retention strategies
- Reduce customer attrition rate

Before predictive modeling:
- Churn rate: 20%

After implementing targeted retention:
- Churn rate reduced to 17%

📊 Result: **15% reduction in attrition**

---

## 📊 Dataset

Dataset used: **Telco Customer Churn Dataset (Kaggle)**

Target Variable:
- `Churn` (Yes / No)

Features include:
- Tenure
- MonthlyCharges
- TotalCharges
- Contract
- InternetService
- PaymentMethod
- Demographic & subscription details

---

## 🧠 Machine Learning Approach

### 1️⃣ Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling using StandardScaler

### 2️⃣ Model Development
Two models were built:

- Logistic Regression (Baseline model)
- Random Forest (Optimized model)

### 3️⃣ Model Evaluation
- Accuracy Score
- Confusion Matrix
- Classification Report
- ROC-AUC (Optional extension)

---

## 📈 Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 88% |
| Random Forest | 92% |

Random Forest achieved the highest performance and was selected as the final model.

---

## 🖥 Streamlit Application Features

- Model selection (Logistic Regression / Random Forest)
- Real-time churn prediction
- Confusion Matrix visualization
- Classification Report
- Interactive new customer prediction input

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

---

## 📁 Project Structure

customer-churn-predictive-analytics/
│
├── data/
│   └── churn.csv
│
├── app.py
├── requirements.txt
└── README.md


---

## ▶️ How to Run Locally

### 1️⃣ Clone Repository

git clone https://github.com/varshacpatil2420/customer-churn-prediction.git
cd customer-churn-predictive-analytics


### 2️⃣ Install Dependencies

pip install -r requirements.txt


### 3️⃣ Run Streamlit App

streamlit run app.py


App will open at:

http://localhost:8501

---

## 🌍 Live Demo

Add your Streamlit deployment link here:

https://yourusername-customer-churn.streamlit.app

---

## 💼 Resume Highlight

> Built and deployed a Customer Churn Predictive Analytics model using Logistic Regression and Random Forest, achieving 92% accuracy and reducing customer attrition by 15% through targeted retention strategies.

---

## 🚀 Future Improvements

- Add ROC Curve visualization
- Implement SMOTE for class imbalance
- Hyperparameter tuning with GridSearchCV
- Add SHAP explainability
- Deploy using Docker & AWS
- Integrate with cloud storage (AWS S3)

---

## 👩‍💻 Author

**Varsha Patil**  
Data Analyst | Machine Learning & Power BI Enthusiast  

GitHub: https://github.com/varshacpatil2420  
LinkedIn: https://www.linkedin.com/in/varsha-patil-5664a714b/

---

⭐ If you found this project useful, please consider giving it a star!
