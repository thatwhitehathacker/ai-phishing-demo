import streamlit as st
import joblib
from scipy.sparse import hstack
import re
import numpy as np

st.set_page_config(page_title="AI Phishing Email Detector Demo", layout="centered")

# --------- Load model and vectorizers ----------
model = joblib.load("phishing_model.pkl")
vectorizer_subject = joblib.load("vectorizer_subject.pkl")
vectorizer_body = joblib.load("vectorizer_body.pkl")

# --------- Preprocessing helper ----------
URL_RE = re.compile(r"(https?://\S+)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_ALPHANUM_RE = re.compile(r"[^a-z0-9@:/.\-_ ]+")

def preprocess_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.lower()
    urls = URL_RE.findall(text)
    text = URL_RE.sub(" URLTOKEN ", text)
    text = NON_ALPHANUM_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if urls:
        text += " URLTOKEN"
    return text

# --------- Prediction function ----------
def get_prediction_and_confidence(subject, body):
    subj_clean = preprocess_text(subject)
    body_clean = preprocess_text(body)
    subj_features = vectorizer_subject.transform([subj_clean])
    body_features = vectorizer_body.transform([body_clean])
    features = hstack([subj_features, body_features])

    if features.nnz == 0:
        # Unknown vocabulary fallback
        pred_label = 1  # default to phishing
        phishing_prob = 0.5
        return pred_label, phishing_prob, True
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        phishing_prob = float(probs[0][1])
        pred_label = int(probs[0].argmax())
        return pred_label, phishing_prob, False
    # fallback
    pred_label = int(model.predict(features)[0])
    return pred_label, float(pred_label), False

# --------- Demo Examples (guaranteed to work) ----------
examples = {
    "— pick an example —": ("", ""),
    "Legit — Course Info": (
        "Rider Spring 2026 Course Selection Information",
        "Please review your course selection for the upcoming semester."
    ),
    "Legit — University Admin": (
        "Important: Graduation Application & Deadlines",
        "Complete your graduation application by Nov 1. Visit the registrar page for required forms."
    ),
    "Phishing — Account Scam": (
        "URGENT: Your account will be locked",
        "We detected suspicious activity. Click https://fake-login.example to verify now."
    ),
    "Phishing — Fake Invoice": (
        "Invoice #8729 overdue — pay immediately",
        "Invoice attached. Click here to pay now: http://fake-pay.example"
    )
}

st.title("AI Phishing Email Detector Stage 1 - :)")
st.write("Select an example or paste your own email subject and body below. Hope you enjoy our project!")

# --------- Example Dropdown ----------
example_choice = st.selectbox("Choose an example email:", list(examples.keys()))
if example_choice and example_choice != "— pick an example —":
    default_subj, default_body = examples[example_choice]
else:
    default_subj, default_body = "", ""

subject_input = st.text_input("Email Subject:", value=default_subj)
body_input = st.text_area("Email Body:", value=default_body, height=240)

if st.button("Detect"):
    if subject_input.strip() == "" and body_input.strip() == "":
        st.warning("Please enter a subject and/or body (or choose an example).")
    else:
        label, conf, unknown_vocab = get_prediction_and_confidence(subject_input, body_input)
        conf_display = f"{conf:.2f}"
        if unknown_vocab:
            st.warning("⚠️ Input contains words unknown to the model — result may be low-confidence.")
            st.info(f"Predicted: {'Phishing' if label==1 else 'Legit'} (Confidence ≈ {conf_display})")
        else:
            if label == 1:
                st.error(f"🚨 Phishing (Confidence {conf_display})")
            else:
                st.success(f"✅ Legit (Confidence {conf_display})")

st.markdown("---")
st.caption("Created by Jeremy Burrell and Evan Cho")
