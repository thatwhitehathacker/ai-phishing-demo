import streamlit as st
import joblib, os, numpy as np, re
from urlextract import URLExtract
import tldextract

# File paths
MODEL_PATH = "phishing_model.pkl"
VEC_SUBJECT_PATH = "vectorizer_subject.pkl"
VEC_BODY_PATH = "vectorizer_body.pkl"

# Check files exist
for f in [MODEL_PATH, VEC_SUBJECT_PATH, VEC_BODY_PATH]:
    if not os.path.exists(f):
        st.error(f"File not found: {f}. Make sure it is in the same folder as this app.")
        st.stop()

# Load model and vectorizers
clf = joblib.load(MODEL_PATH)
vectorizer_subject = joblib.load(VEC_SUBJECT_PATH)
vectorizer_body = joblib.load(VEC_BODY_PATH)

# Heuristic components
SUSPICIOUS_WORDS = [
    "URGENT","immediate","WARNING","verify","suspend","suspended","expire","click here",
    "your account","action required","update now","password","verify now","account will be",
    "you've won","you won","claim now","gift card","limited time","irs","tax","bank",
    "paypal","account closed","frozen","payment failed","verify identity","restricted"
]
SUSPICIOUS_TLDS = {'tk','zip','info','pw','cn','loan'}
extractor = URLExtract()

def heuristic_density(text):
    s = str(text).lower()
    words = re.findall(r'\w+', s)
    total = len(words) if len(words) > 0 else 1
    susp_count = sum(1 for w in SUSPICIOUS_WORDS if w in s)
    density_score = min(1.0, (susp_count / total) * 6.0) * 0.55

    urls = extractor.find_urls(s)
    url_score = 0.0
    if urls:
        url_score += 0.25
        for u in urls:
            ext = tldextract.extract(u)
            if ext.suffix and ext.suffix.lower() in SUSPICIOUS_TLDS:
                url_score += 0.15
        url_score = min(0.5, url_score)

    excl_score = min(s.count('!') * 0.05, 0.15)
    num_score = 0.15 if re.search(r'\b(days?|hours?|minutes?|fine|fined|owed|over \$\d+)\b', s) else 0.0

    return min(1.0, density_score + url_score + excl_score + num_score)

def combined_boosted_proba(text, alpha=0.7, force_boost=True, boost_threshold=0.45, boost_to=0.92):
    vec = vectorizer_body.transform([text])
    base_probs = clf.predict_proba(vec)[0]
    base_phish = float(base_probs[1])
    h = heuristic_density(text)
    combined = alpha * base_phish + (1 - alpha) * h
    if force_boost and h >= boost_threshold:
        combined = max(combined, boost_to)
    return np.array([1.0 - combined, combined]), base_phish, h

# ---- Streamlit UI ----
st.title("🚀 BOOSTED AI Phishing Detector Demo")

subject_input = st.text_input("Email Subject")
body_input = st.text_area("Email Body")

alpha = st.slider("Model Weight (alpha)", 0.0, 1.0, 0.7)
threshold = st.slider("Label Threshold", 0.0, 1.0, 0.7)
force_boost = st.checkbox("Force Boost Heuristic", True)

if st.button("Check Email"):
    email_text = f"{subject_input}\n{body_input}"
    probs, base_phish, h = combined_boosted_proba(email_text, alpha, force_boost, boost_threshold=threshold)
    boosted_phish = float(probs[1])
    label = "🚨 PHISHING" if boosted_phish >= threshold else "✅ LEGIT"

    st.write("---")
    st.write(f"**Label:** {label}")
    st.write(f"Boosted Phishing Probability: {boosted_phish*100:.2f}%")
    st.write(f"Base Model Probability: {base_phish*100:.2f}%")
    st.write(f"Heuristic Score: {h:.3f}")
