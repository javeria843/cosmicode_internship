import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page config
st.set_page_config(page_title="Breast Cancer CSV Classifier", layout="centered")
st.title("🔬 Breast Cancer Prediction from CSV")
st.markdown("No `.pkl` files used – everything is done from CSV")

# Load CSV
@st.cache_data
def load_data():
    df = pd.read_csv("breast_cancer_data.csv")
    return df

df = load_data()
st.success("✅ CSV Loaded Successfully!")

# Sidebar Navigation
option = st.sidebar.radio("Select Option", ["EDA", "Train & Predict"])

# EDA Section
if option == "EDA":
    st.subheader("📊 Exploratory Data Analysis")
    st.write(df.head())

    st.write("### Target Value Counts")
    st.bar_chart(df["target"].value_counts())

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Prediction Section
elif option == "Train & Predict":
    st.subheader("🧪 Train & Predict")

    X = df.drop("target", axis=1)
    y = df["target"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"### 📈 Accuracy: `{acc:.2f}`")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Manual Prediction
    st.write("### 🧍 Enter Feature Values Manually:")
    input_data = []
    for col in X.columns:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        input_data.append(val)

    if st.button("Predict"):
        user_scaled = scaler.transform([input_data])
        pred = model.predict(user_scaled)[0]
        result = "Benign (No Cancer)" if pred == 1 else "Malignant (Cancer Detected)"
        st.success(f"🎯 Prediction: {result}")
