import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("bmi_data.csv")

df = load_data()

# Train Model
X = df[['Age', 'Weight', 'Height']]
y = df['Category']

# Encode target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("BMI Calculator & Health Prediction")

# User Inputs
age = st.slider("Select your Age", 18, 65, 25)
weight = st.slider("Enter your Weight (kg)", 40, 120, 70)
height = st.slider("Enter your Height (cm)", 140, 200, 170) / 100  # Convert to meters

# Calculate BMI
bmi = weight / (height ** 2)
st.write(f"Your **BMI** is: **{bmi:.2f}**")

# Predict BMI Category
input_data = np.array([[age, weight, height]])
prediction_encoded = model.predict(input_data)[0]
prediction = encoder.inverse_transform([prediction_encoded])[0]

# Display Result
st.subheader("Health Status")
st.write(f"Based on your BMI, you are classified as: **{prediction}**")

# Provide Health Advice
st.subheader("Health Recommendations")
if prediction == "Underweight":
    st.write("You should focus on gaining weight by eating a balanced diet and increasing calorie intake.")
elif prediction == "Normal weight":
    st.write("Great! Maintain a healthy lifestyle with a balanced diet and regular exercise.")
elif prediction == "Overweight":
    st.write("Consider monitoring your diet and engaging in regular physical activity.")
else:  # Obese
    st.write("It is advisable to consult a healthcare professional for guidance on weight management.")

st.info("BMI is a general indicator and does not account for muscle mass or other health factors. Please consult a doctor for a comprehensive health assessment.")
