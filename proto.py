import streamlit as st
import numpy as np
import joblib

# Load model and preprocessors
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

st.set_page_config(page_title="Crop Recommendation System", layout="centered")
st.title("🌾 Crop Recommendation System")

st.markdown("Adjust the values below based on your soil and weather conditions:")

# User input sliders and selects
n = st.selectbox('Nitrogen (N)', range(0, 150, 5))
p = st.selectbox('Phosphorus (P)', range(0, 150, 5))
k = st.selectbox('Potassium (K)', range(0, 150, 5))

temperature = st.slider('Temperature (°C)', 0.0, 50.0, 25.0)
humidity = st.slider('Humidity (%)', 10.0, 100.0, 50.0)
ph = st.slider('pH Level', 3.5, 9.5, 6.5)
rainfall = st.slider('Rainfall (mm)', 0.0, 300.0, 100.0)

# Prepare and scale input
input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict Crop"):
    prediction = model.predict(input_scaled)
    predicted_crop = label_encoder.inverse_transform(prediction)
    st.success(f"✅ Recommended Crop: **{predicted_crop[0].title()}**")
