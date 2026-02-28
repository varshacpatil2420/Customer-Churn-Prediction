import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📉 Customer Churn Predictive Analytics Model")
st.write("Predict customer churn using Logistic Regression & Random Forest")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Telco-Customer-Churn.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Preprocessing
df.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Sidebar Model Selection
st.sidebar.header("Model Selection")
model_option = st.sidebar.selectbox(
    "Choose Model", 
    ("Logistic Regression", "Random Forest")
)

if model_option == "Logistic Regression":
    model = LogisticRegression()
else:
    model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.sidebar.write(f"Model Accuracy: {round(accuracy*100,2)}%")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# New Customer Prediction
st.subheader("Predict New Customer")

input_data = []
for col in X.columns:
    value = st.number_input(f"Enter {col}", value=0.0)
    input_data.append(value)

if st.button("Predict Churn"):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Customer is likely to Churn ⚠️")
    else:
        st.success("Customer is likely to Stay ✅")

st.markdown("---")
st.write("Developed by Varsha Patil")