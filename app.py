# app.py
import streamlit as st
import joblib
import numpy as np
import re
from urlextract import URLExtract
import tldextract
from scipy.sparse import hstack

# ----------------- Page setup -----------------
st.set_page_config(page_title="AI Phishing Email Detector - BOOSTED", layout="centered")
st.title("🚀 AI Phishing Email Detector - BOOSTED")

# ----------------- Load model and vectorizers -----------------
MODEL_PATH = "phishing_model.pkl"
VEC_BODY_PATH = "vectorizer_body.pkl"
VEC_SUBJ_PATH = "vectorizer_subject.pkl"

try:
    clf = joblib.load(MODEL_PATH)
    vectorizer_body = joblib.load(VEC_BODY_PATH)
    vectorizer_subject = joblib.load(VEC_SUBJ_PATH)
except FileNotFoundError:
    st.error("❌ Model or vectorizer files not found. Make sure they are in the repo folder.")
    st.stop()

extractor = URLExtract()

# ----------------- Heuristic -----------------
SUSPICIOUS_WORDS = [
    "URGENT","immediate","WARNING","verify","suspend","suspended","expire","click here",
    "your account","action required","update now","password","verify now","account will be",
    "you've won","you won","claim now","gift card","limited time","irs","tax","bank",
    "paypal","account closed","frozen","payment failed","verify identity","restricted"
]
SUSPICIOUS_TLDS = {'tk','zip','info','pw','cn','loan'}

def heuristic_density(text):
    s = str(text).lower()
    words = re.findall(r'\w+', s)
    total = len(words) if len(words)>0 else 1
    susp_count = sum(1 for w in SUSPICIOUS_WORDS if w in s)
    density_score = min(1.0, (susp_count / total) * 6.0) * 0.55

    # urls
    urls = extractor.find_urls(s)
    url_score = 0.0
    if len(urls) > 0:
        url_score += 0.25
        for u in urls:
            ext = tldextract.extract(u)
            if ext.suffix and ext.suffix.lower() in SUSPICIOUS_TLDS:
                url_score += 0.15
        url_score = min(0.5, url_score)

    excl_score = min(s.count('!') * 0.05, 0.15)

    num_score = 0.0
    if re.search(r'\b(days?|hours?|minutes?)\b', s) or re.search(r'\b(fine|fined|owed|over \$\d+)\b', s):
        num_score += 0.15

    h = density_score + url_score + excl_score + num_score
    return min(1.0, h)

# ----------------- Prediction -----------------
def get_boosted_prediction(subject, body, alpha=0.7, force_boost=True, boost_threshold=0.45, boost_to=0.92):
    subj_features = vectorizer_subject.transform([subject])
    body_features = vectorizer_body.transform([body])
    features = hstack([subj_features, body_features])

    if features.nnz == 0:
        return 1, 0.5, True  # fallback phishing

    base_probs = clf.predict_proba(features)[0]
    base_phish = float(base_probs[1])

    h = heuristic_density(subject + " " + body)
    combined = alpha * base_phish + (1 - alpha) * h
    if force_boost and h >= boost_threshold:
        combined = max(combined, boost_to)

    return int(combined >= 0.7), combined, False

# ----------------- UI -----------------
st.write("Paste an email subject and body, or choose an example below:")

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

example_choice = st.selectbox("Choose an example email:", list(examples.keys()))
subject_input = st.text_input("Email Subject:", value=examples.get(example_choice, ("", ""))[0])
body_input = st.text_area("Email Body:", value=examples.get(example_choice, ("", ""))[1], height=240)

if st.button("Detect"):
    if subject_input.strip() == "" and body_input.strip() == "":
        st.warning("Please enter a subject and/or body (or choose an example).")
    else:
        label, conf, unknown = get_boosted_prediction(subject_input, body_input)
        conf_display = f"{conf*100:.2f}%"
        if unknown:
            st.warning("⚠️ Unknown words — result may be low-confidence.")
        if label == 1:
            st.error(f"🚨 PHISHING (Confidence {conf_display})")
        else:
            st.success(f"✅ LEGIT (Confidence {conf_display})")

st.markdown("---")
st.caption("Created by Jeremy Burrell and Evan Cho")
