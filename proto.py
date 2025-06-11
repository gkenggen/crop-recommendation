import streamlit as st
import numpy as np
import joblib

# Load saved model, scaler, and label encoder
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

st.title("🌾 Crop Recommendation System pogi ako")

# Input sliders for soil nutrients and environmental factors
N = st.slider("Nitrogen (N) level", 0, 140, 40)
P = st.slider("Phosphorus (P) level", 5, 145, 5)
K = st.slider("Potassium (K) level", 5, 205, 15)
temperature = st.slider("Temperature (°C)", 8.0, 45.0, 25.0)
humidity = st.slider("Humidity (%)", 14.0, 100.0, 50.0)
ph = st.slider("Soil pH", 3.5, 9.5, 6.5)
rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

if st.button("Predict Crop"):
    # Prepare input data
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    
    # Predict crop
    prediction_encoded = model.predict(input_scaled)
    crop_prediction = label_encoder.inverse_transform(prediction_encoded)[0]

    crop_explanations = {
        'rice': "Rice thrives in areas with high rainfall and requires moderate temperatures.",
        'maize': "Maize grows well in well-drained soils with balanced nutrient levels and moderate climate.",
        'chickpea': "Chickpea prefers drier conditions and can grow well in soils with lower nitrogen levels.",
        'kidneybeans': "Kidneybeans need well-drained soils and moderate temperatures with balanced nutrients.",
        'pigeonpeas': "Pigeonpeas tolerate drought and grow well in slightly acidic to neutral soils.",
        'mothbeans': "Mothbeans are drought-resistant and thrive in sandy soils with less water.",
        'mungbean': "Mungbeans grow best in warm weather and require moderate nutrient levels.",
        'blackgram': "Blackgram prefers warm climates with moderate rainfall and nutrient balance.",
        'lentil': "Lentils require cooler climates and soils with adequate moisture but good drainage.",
        'pomegranate': "Pomegranates grow best in semi-arid regions with warm temperatures and less humidity.",
        'banana': "Bananas require tropical climates with high humidity and rich soil nutrients.",
        'mango': "Mango trees flourish in warm tropical climates with well-drained soil.",
        'grapes': "Grapes need warm days and cool nights, thriving in soils with good drainage.",
        'watermelon': "Watermelons require warm temperatures and sandy loam soils with moderate nutrients.",
        'muskmelon': "Muskmelons prefer warm, sunny conditions and well-drained soils.",
        'apple': "Apples grow best in temperate climates with cold winters and well-drained soil.",
        'orange': "Oranges require warm subtropical climates with adequate rainfall and nutrient-rich soil.",
        'papaya': "Papayas thrive in tropical climates with plenty of sunlight and well-drained soils.",
        'coconut': "Coconuts need coastal tropical climates with high humidity and sandy soils.",
        'cotton': "Cotton grows well in warm climates with moderate rainfall and well-drained soils.",
        'jute': "Jute requires warm and humid climates with plenty of rainfall.",
        'coffee': "Coffee plants grow best in tropical highlands with moderate temperatures and rainfall."
    }

    st.success(f"Recommended Crop: {crop_prediction.capitalize()} 🌱")
    explanation = crop_explanations.get(crop_prediction.lower(), 
                                         "This crop is well-suited to the given soil and environmental conditions.")
    st.info(explanation)
