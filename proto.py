import streamlit as st
import numpy as np
import joblib

# Load model, scaler, label encoder (adjust paths as needed)
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

st.title("🌾 Crop Recommendation System")

# Input sliders for N, P, K with emojis
N = st.slider("🧪 Nitrogen (N)", min_value=0, max_value=140, value=40, step=1)
P = st.slider("🧪 Phosphorus (P)", min_value=0, max_value=140, value=40, step=1)
K = st.slider("🧪 Potassium (K)", min_value=0, max_value=140, value=40, step=1)

# Input sliders for temperature, humidity, pH, and rainfall with emojis
temperature = st.slider("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
humidity = st.slider("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
ph = st.slider("🧴 pH value", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
rainfall = st.slider("☔ Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0, step=0.1)

if st.button("🌟 Predict Crop"):
    # Prepare input data as numpy array
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Scale inputs
    input_scaled = scaler.transform(input_data)

    # Predict using the trained model
    pred_encoded = model.predict(input_scaled)

    # Decode the predicted label
    predicted_crop = label_encoder.inverse_transform(pred_encoded)[0]

    st.success(f"🌿 The recommended crop is: {predicted_crop}")
