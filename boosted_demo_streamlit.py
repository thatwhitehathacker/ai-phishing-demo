import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("AI Phishing Email Detector")

subject = st.text_input("Email Subject:")
body = st.text_area("Email Body:")

if st.button("Detect"):
    text = subject + " " + body
    features = vectorizer.transform([text])
    pred_prob = model.predict_proba(features)[0][1]
    pred_label = "🚨 PHISHING" if pred_prob >= 0.5 else "✅ LEGIT"
    st.write(f"Prediction: {pred_label} (Probability: {pred_prob:.2f})")
