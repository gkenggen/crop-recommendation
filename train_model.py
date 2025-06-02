import joblib

# Save the trained model
joblib.dump(model, 'model.joblib')

# Save the scaler and label encoder if used
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
