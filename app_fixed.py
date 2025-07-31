
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.title("💓 Heart Disease Risk Predictor")

# Upload CSV
uploaded_file = st.file_uploader("Upload your heart.csv file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("### Data Shape")
    st.write(df.shape)

    st.write("### Info Summary")
    st.text(str(df.dtypes))

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    st.write("### Statistical Summary")
    st.write(df.describe())

    # One-hot encoding
    cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Visualizations
    st.write("### Target Variable Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='HeartDisease', data=df, ax=ax)
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("### Pairplot of Key Features")
    selected_cols = ['Age', 'Cholesterol', 'RestingBP', 'MaxHR', 'HeartDisease']
    if all(col in df.columns for col in selected_cols):
        st.pyplot(sns.pairplot(df[selected_cols], hue='HeartDisease'))

    st.write("### Cholesterol by Heart Disease")
    fig, ax = plt.subplots()
    sns.boxplot(x='HeartDisease', y='Cholesterol', data=df, ax=ax)
    st.pyplot(fig)

    # Split and scale
    x = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Model
    model = SVC(kernel='rbf')
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    # Results
    st.subheader("📊 Model Performance (SVM)")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred):.2f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred):.2f}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
else:
    st.info("👆 Upload a CSV file to begin.")
