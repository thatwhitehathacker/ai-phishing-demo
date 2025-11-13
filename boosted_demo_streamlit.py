# ===== BOOSTED Phishing Email Detector - Streamlit Version =====
import streamlit as st
import joblib, os, numpy as np, re
from urlextract import URLExtract
import tldextract

st.set_page_config(page_title="BOOSTED Phishing Email Detector", layout="centered")

extractor = URLExtract()

# ===== Load model and vectorizer =====
MODEL_PATH = "phishing_model.pkl"
VEC_PATH = "tfidf_vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
    st.error("Model or vectorizer file not found. Check working directory.")
    st.stop()

clf = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# ===== Heuristic settings =====
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
    if urls:
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

def combined_boosted_proba(text, alpha=0.7, force_boost=True, boost_threshold=0.45, boost_to=0.92):
    vec = vectorizer.transform([text])
    base_probs = clf.predict_proba(vec)[0]
    base_phish = float(base_probs[1])

    h = heuristic_density(text)
    combined = alpha * base_phish + (1 - alpha) * h

    if force_boost and h >= boost_threshold:
        combined = max(combined, boost_to)

    return np.array([1.0 - combined, combined]), base_phish, h

# ===== Streamlit UI =====
st.title("🚀 BOOSTED Phishing Email Detector")

st.write("Paste an email subject and body below to check if it is phishing or legit. Adjust settings to see how the model reacts.")

subject_input = st.text_input("Email Subject:")
body_input = st.text_area("Email Body:", height=240)

alpha = st.slider("Alpha (model weight)", 0.0, 1.0, 0.7, 0.05)
threshold = st.slider("Label Threshold", 0.0, 1.0, 0.7, 0.05)
force_boost = st.checkbox("Force Boost Heuristics", value=True)

if st.button("Detect"):
    if not subject_input.strip() and not body_input.strip():
        st.warning("Please enter a subject and/or body.")
    else:
        text = subject_input + " " + body_input
        probs, base_phish, h = combined_boosted_proba(text, alpha=alpha, force_boost=force_boost)
        boosted_phish = float(probs[1])
        label = "🚨 PHISHING" if boosted_phish >= threshold else "✅ LEGIT"

        st.subheader("Result")
        st.write(f"**Label:** {label}")
        st.write(f"**Boosted phishing probability:** {boosted_phish:.2f}")
        st.write(f"**Base model probability:** {base_phish:.2f}")
        st.write(f"**Heuristic score:** {h:.3f}")

st.markdown("---")
st.caption("Created by Jeremy Burrell")
