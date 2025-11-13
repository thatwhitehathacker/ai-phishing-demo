import streamlit as st
import joblib
from scipy.sparse import hstack
import numpy as np
import re
from urlextract import URLExtract
import tldextract

st.set_page_config(page_title="AI Phishing Email Detector (Boosted)", layout="centered")

extractor = URLExtract()

# -------- Load model and vectorizers ----------
MODEL_PATH = "phishing_model.pkl"
VECTOR_SUBJECT_PATH = "vectorizer_subject.pkl"
VECTOR_BODY_PATH = "vectorizer_body.pkl"

try:
    clf = joblib.load(MODEL_PATH)
    vectorizer_subject = joblib.load(VECTOR_SUBJECT_PATH)
    vectorizer_body = joblib.load(VECTOR_BODY_PATH)
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Make sure phishing_model.pkl, vectorizer_subject.pkl, and vectorizer_body.pkl are in the repo.")
    st.stop()

# -------- Heuristic components ----------
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
    total = len(words) if len(words) > 0 else 1
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

    return min(1.0, density_score + url_score + excl_score + num_score)

# -------- Prediction function ----------
def combined_boosted_proba(subject, body, alpha=0.7, force_boost=True, boost_threshold=0.45, boost_to=0.92):
    subj_features = vectorizer_subject.transform([subject])
    body_features = vectorizer_body.transform([body])
    features = hstack([subj_features, body_features])

    base_probs = clf.predict_proba(features)[0]
    base_phish = float(base_probs[1])
    h = heuristic_density(subject + " " + body)
    combined = alpha * base_phish + (1 - alpha) * h

    if force_boost and h >= boost_threshold:
        combined = max(combined, boost_to)

    return combined, base_phish, h

# -------- Streamlit UI ----------
st.title("🚀 AI Phishing Email Detector (Boosted)")
st.write("Paste an email subject and body below or select an example:")

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
body_input = st.text_area("Email Body:", value=default_body, height=200)

alpha = st.slider("Model weight (alpha)", 0.0, 1.0, 0.7)
boost_threshold = st.slider("Boost threshold", 0.0, 1.0, 0.45)
force_boost = st.checkbox("Enable force boost", True)

if st.button("Detect"):
    if not subject_input.strip() and not body_input.strip():
        st.warning("Please enter a subject and/or body (or select an example).")
    else:
        boosted_prob, base_prob, h_score = combined_boosted_proba(
            subject_input, body_input,
            alpha=alpha,
            force_boost=force_boost,
            boost_threshold=boost_threshold
        )
        label = "🚨 PHISHING" if boosted_prob >= 0.7 else "✅ LEGIT"
        st.markdown(f"**Label:** {label}")
        st.markdown(f"Boosted phishing probability: {boosted_prob*100:.2f}%")
        st.markdown(f"Base model probability: {base_prob*100:.2f}%")
        st.markdown(f"Heuristic score: {h_score:.3f}")
