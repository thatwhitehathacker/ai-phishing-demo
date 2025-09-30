# app.py — Streamlit phishing demo (robust confidence + preprocessing)
import streamlit as st
import joblib
import os
import re
import requests
import numpy as np
from scipy.sparse import hstack

st.set_page_config(page_title="AI Phishing Email Detector", layout="centered")

# --------- Google Drive downloader (if you used Drive IDs) ----------
# If your deployment already has the .pkl files in the repo, these lines
# are no-ops. Replace IDs with yours if needed.
def download_from_drive(file_id, dest_filename):
    if os.path.exists(dest_filename):
        return
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    r.raise_for_status()
    with open(dest_filename, "wb") as f:
        f.write(r.content)

# If you need to download from Drive, fill these with your IDs; otherwise leave blank
MODEL_ID = ""        # e.g. "1AbCdE..."
VEC_SUB_ID = ""
VEC_BODY_ID = ""

if MODEL_ID:
    download_from_drive(MODEL_ID, "phishing_model.pkl")
if VEC_SUB_ID:
    download_from_drive(VEC_SUB_ID, "vectorizer_subject.pkl")
if VEC_BODY_ID:
    download_from_drive(VEC_BODY_ID, "vectorizer_body.pkl")

# --------- Load model + vectorizers ----------
@st.cache_resource(show_spinner=False)
def load_objects():
    model = joblib.load("phishing_model.pkl")
    vec_subj = joblib.load("vectorizer_subject.pkl")
    vec_body = joblib.load("vectorizer_body.pkl")
    return model, vec_subj, vec_body

try:
    model, vectorizer_subject, vectorizer_body = load_objects()
except Exception as e:
    st.error("Error loading model/vectorizers. Check logs. " + str(e))
    st.stop()

# --------- Preprocessing helper (simple, matches typical TF-IDF training) ----------
URL_RE = re.compile(r"(https?://\S+)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_ALPHANUM_RE = re.compile(r"[^a-z0-9@:/.\-_ ]+")

def preprocess_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    # remove HTML tags
    text = HTML_TAG_RE.sub(" ", text)
    # lowercase
    text = text.lower()
    # keep URLs intact but normalize spacing
    urls = URL_RE.findall(text)
    text = URL_RE.sub(" URLTOKEN ", text)
    # remove weird characters but keep basic punctuation and "@" for emails
    text = NON_ALPHANUM_RE.sub(" ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # reinsert a simple token for URLs if any found
    if urls:
        text = text + " URLTOKEN"
    return text

# --------- Prediction helper that returns (label, confidence) ----------
def get_prediction_and_confidence(subject, body):
    subj_clean = preprocess_text(subject)
    body_clean = preprocess_text(body)
    subj_features = vectorizer_subject.transform([subj_clean])
    body_features = vectorizer_body.transform([body_clean])
    features = hstack([subj_features, body_features])

    # If features are all zeros (no known vocab), handle gracefully:
    if features.nnz == 0:
        # fallback: model can't see any known words. Return low-confidence "unknown"
        # Use class prior if available otherwise 0.5
        try:
            # Attempt to infer class prior from training if model has classes_ and was fitted
            if hasattr(model, "classes_") and hasattr(model, "class_prior_"):
                # scikit-learn MultinomialNB sets class_log_prior_ but not class_prior_ always;
                # fallback to equal probability
                priors = getattr(model, "class_prior_", None)
                if priors is not None:
                    phishing_prob = float(priors[1])
                else:
                    phishing_prob = 0.5
            else:
                phishing_prob = 0.5
        except Exception:
            phishing_prob = 0.5
        # Choose predicted label based on prior but mark low confidence
        pred_label = int(phishing_prob >= 0.5)
        return pred_label, phishing_prob, True  # True -> unknown vocabulary
    # Normal case: model sees some features
    # Prefer predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        phishing_prob = float(probs[0][1])
        pred_label = int(probs[0].argmax())
        return pred_label, phishing_prob, False
    # Fallback to decision_function if available (sigmoid to pseudo-prob)
    if hasattr(model, "decision_function"):
        score = model.decision_function(features)
        # convert score to probability with sigmoid
        phishing_prob = float(1 / (1 + np.exp(-score)) )
        pred_label = int(phishing_prob >= 0.5)
        return pred_label, phishing_prob, False
    # Final fallback: raw predict (0 or 1), confidence unknown
    pred_label = int(model.predict(features)[0])
    return pred_label, float(pred_label), False

# --------- UI ----------
st.title("AI Phishing Email Detector CYB-300-K-1")
st.write("Paste an email subject and body and click Detect. Hope you enjoy our Project!")

# Example dropdown
examples = {
    "— pick example —": ("", ""),
    "Phishing — account scam": (
        "URGENT: Your account will be locked",
        "We detected suspicious activity. Click https://fake-login.example to verify now."
    ),
    "Legit — course info": (
        "Rider Spring 2026 Course Selection Information",
        "Dear student, please review your course selection for the upcoming semester. Registration opens Monday."
    ),
    "Legit — university admin (longer)": (
        "Important: Graduation application & deadlines",
        "Hello, please complete your graduation application by Nov 1. Visit the registrar page for details and required forms."
    ),
    "Phishing — fake invoice": (
        "Invoice #8729 overdue — pay immediately",
        "Invoice attached. Click here to pay now: http://fake-pay.example"
    )
}

col1, col2 = st.columns([3,1])
with col1:
    choice = st.selectbox("Examples", list(examples.keys()))
with col2:
    if st.button("Clear"):
        choice = "-- pick example --"

if choice and choice != "-- pick example --":
    default_subj, default_body = examples[choice]
else:
    default_subj, default_body = ("", "")

subject_input = st.text_input("Email subject:", value=default_subj)
body_input = st.text_area("Email body:", value=default_body, height=240)

if st.button("Detect"):
    if subject_input.strip() == "" and body_input.strip() == "":
        st.warning("Please enter a subject and/or body (or choose an example).")
    else:
        label, conf, unknown_vocab = get_prediction_and_confidence(subject_input, body_input)
        conf_display = f"{conf:.2f}"
        if unknown_vocab:
            st.warning("⚠️ The input did not contain words the model recognizes — result is low-confidence.")
            st.info(f"Predicted: {'Phishing' if label==1 else 'Legit'} (Confidence ≈ {conf_display})")
        else:
            if label == 1:
                st.error(f"🚨 Phishing (Confidence {conf_display})")
            else:
                st.success(f"✅ Legit (Confidence {conf_display})")

st.markdown("---")
st.caption("Created by Jeremy Burrell and Evan Cho")