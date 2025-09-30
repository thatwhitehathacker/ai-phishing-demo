# app.py — Streamlit phishing demo (paste/replace entire file)
import streamlit as st
import joblib
from scipy.sparse import hstack
import numpy as np

st.set_page_config(page_title="AI Phishing Email Detector", layout="centered")

st.title("AI Phishing Email Detector")
st.write("Paste an email subject and body below and click **Detect**. This demo uses a TF‑IDF + model pipeline.")

# Load model & vectorizers (files must be next to app.py)
@st.cache_resource
def load_objects():
    model = joblib.load("phishing_model.pkl")
    vec_subj = joblib.load("vectorizer_subject.pkl")
    vec_body = joblib.load("vectorizer_body.pkl")
    return model, vec_subj, vec_body

model, vectorizer_subject, vectorizer_body = load_objects()

# Example emails
examples = {
    "Choose an example": ("",""),
    "Phishing: account update scam": (
        "WARNING! Your account will be deleted soon unless you update via this link:",
        "Please click the link to verify your account immediately."
    ),
    "Legit: course selection (short)": (
        "Rider Spring 2026 Course Selection Information",
        "Please review your course selection for the upcoming semester. Registration open next week."
    ),
    "Phishing: fake invoice": (
        "Urgent: Invoice #8729 Overdue — Pay Immediately",
        "Invoice attached. Click here to pay now: http://fake-pay.example"
    )
}

col1, col2 = st.columns([3,1])
with col1:
    choice = st.selectbox("Or select an example:", list(examples.keys()))
with col2:
    if st.button("Clear"):
        choice = "Choose an example"

if choice and choice != "Choose an example":
    subj_default, body_default = examples[choice]
else:
    subj_default, body_default = ("","")

subject_input = st.text_input("Email subject:", value=subj_default)
body_input = st.text_area("Email body:", value=body_default, height=220)

def predict_confidence(subject, body):
    subj_features = vectorizer_subject.transform([subject])
    body_features = vectorizer_body.transform([body])
    features = hstack([subj_features, body_features])
    # predict_proba may not exist for all models; handle gracefully
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features.toarray() if hasattr(features, "toarray") else features)[0][1]
    else:
        # fallback: use decision_function if available and convert to a pseudo-prob
        if hasattr(model, "decision_function"):
            score = model.decision_function(features)
            proba = 1 / (1 + np.exp(-score))[0]
        else:
            pred = model.predict(features)[0]
            proba = float(pred)
    pred_label = model.predict(features)[0]
    return pred_label, float(proba)

if st.button("Detect"):
    if subject_input.strip() == "" and body_input.strip() == "":
        st.warning("Please enter a subject and/or body text (or pick an example).")
    else:
        label, confidence = predict_confidence(subject_input, body_input)
        if label == 1:
            st.error(f"🚨 Phishing (Confidence {confidence:.2f})")
        else:
            st.success(f"✅ Legit (Confidence {confidence:.2f})")

st.markdown("---")
st.caption("Note: This is a demo model for a class project. Avoid uploading real private emails or PII.")
