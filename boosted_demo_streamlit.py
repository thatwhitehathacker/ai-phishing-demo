# boosted_demo_streamlit.py
import streamlit as st
import joblib
import re
import numpy as np
from scipy.sparse import hstack
from urlextract import URLExtract
import tldextract

# ----------------- Config -----------------
st.set_page_config(page_title="AI Phishing Email Detector - BOOSTED", layout="centered")
extractor = URLExtract()

MODEL_PATH = "phishing_model.pkl"
VEC_SUBJECT_PATH = "vectorizer_subject.pkl"
VEC_BODY_PATH = "vectorizer_body.pkl"

# ----------------- Load model/vectorizers -----------------
try:
    model = joblib.load(MODEL_PATH)
    vectorizer_subject = joblib.load(VEC_SUBJECT_PATH)
    vectorizer_body = joblib.load(VEC_BODY_PATH)
except FileNotFoundError:
    st.error("Model or vectorizer files not found! Make sure phishing_model.pkl, vectorizer_subject.pkl, and vectorizer_body.pkl are in the repo.")
    st.stop()

# ----------------- Preprocessing -----------------
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

# ----------------- Heuristic components -----------------
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

def combined_boosted_proba(subject, body, alpha=0.7, force_boost=True, boost_threshold=0.45, boost_to=0.92):
    subj_clean = preprocess_text(subject)
    body_clean = preprocess_text(body)

    subj_vec = vectorizer_subject.transform([subj_clean])
    body_vec = vectorizer_body.transform([body_clean])
    features = hstack([subj_vec, body_vec])

    base_probs = model.predict_proba(features)[0]
    base_phish = float(base_probs[1])
    h = heuristic_density(subject + " " + body)
    combined = alpha * base_phish + (1 - alpha) * h
    if force_boost and h >= boost_threshold:
        combined = max(combined, boost_to)
    return combined, base_phish, h

# ----------------- UI -----------------
st.title("🚀 AI Phishing Email Detector - BOOSTED")
st.write("Select an example or paste your own email subject and body below.")

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
if example_choice and example_choice != "— pick an example —":
    default_subj, default_body = examples[example_choice]
else:
    default_subj, default_body = "", ""

subject_input = st.text_input("Email Subject:", value=default_subj)
body_input = st.text_area("Email Body:", value=default_body, height=240)

alpha = st.slider("Model Weight (alpha)", min_value=0.0, max_value=1.0, value=0.7)
force_boost = st.checkbox("Force Boost Heuristic", value=True)
threshold = st.slider("Detection Threshold", min_value=0.0, max_value=1.0, value=0.7)

if st.button("Detect"):
    if subject_input.strip() == "" and body_input.strip() == "":
        st.warning("Please enter a subject and/or body (or choose an example).")
    else:
        boosted_prob, base_phish, h = combined_boosted_proba(subject_input, body_input, alpha=alpha, force_boost=force_boost)
        label = "🚨 PHISHING" if boosted_prob >= threshold else "✅ LEGIT"
        st.write(f"**Label:** {label}")
        st.write(f"Boosted probability: {boosted_prob:.2f}")
        st.write(f"Base model probability: {base_phish:.2f}")
        st.write(f"Heuristic score: {h:.2f}")

st.markdown("---")
st.caption("Created by Jeremy Burrell and Evan Cho")
