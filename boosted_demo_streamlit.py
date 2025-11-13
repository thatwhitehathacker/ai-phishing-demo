# ===== Streamlit Boosted Phishing Demo =====
import streamlit as st
import joblib, os, numpy as np, re
from scipy.sparse import hstack
from urlextract import URLExtract
import tldextract

st.set_page_config(page_title="AI Phishing Email Detector (Boosted)", layout="centered")
st.title("AI Phishing Email Detector")

# ----- Paths for your model/vectorizers -----
MODEL_PATH = "phishing_model.pkl"
VEC_SUBJECT_PATH = "vectorizer_subject.pkl"
VEC_BODY_PATH = "vectorizer_body.pkl"

# ----- Check files exist -----
for f in [MODEL_PATH, VEC_SUBJECT_PATH, VEC_BODY_PATH]:
    if not os.path.exists(f):
        st.error(f"❌ File not found: {f}")
        st.stop()

# ----- Load model/vectorizers -----
model = joblib.load(MODEL_PATH)
vectorizer_subject = joblib.load(VEC_SUBJECT_PATH)
vectorizer_body = joblib.load(VEC_BODY_PATH)

# ----- Heuristic components -----
SUSPICIOUS_WORDS = [
    "urgent","immediate","warning","verify","suspend","suspended","expire","click here",
    "your account","action required","update now","password","verify now","account will be",
    "you've won","you won","claim now","gift card","limited time","irs","tax","bank",
    "paypal","account closed","frozen","payment failed","verify identity","restricted"
]
SUSPICIOUS_TLDS = {'tk','zip','info','pw','cn','loan'}
extractor = URLExtract()

def preprocess_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9@:/.\-_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def heuristic_density(text):
    s = str(text).lower()
    words = re.findall(r'\w+', s)
    total = len(words) if len(words)>0 else 1
    # suspicious word density
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

    # exclamations
    excl_score = min(s.count('!') * 0.05, 0.15)

    # numerical threats
    num_score = 0.0
    if re.search(r'\b(days?|hours?|minutes?)\b', s) or re.search(r'\b(fine|fined|owed|over \$\d+)\b', s):
        num_score += 0.15

    return min(1.0, density_score + url_score + excl_score + num_score)

def get_boosted_prediction(subject, body, alpha=0.7, force_boost=True, boost_threshold=0.45, boost_to=0.92):
    subj_clean = preprocess_text(subject)
    body_clean = preprocess_text(body)
    subj_features = vectorizer_subject.transform([subj_clean])
    body_features = vectorizer_body.transform([body_clean])
    features = hstack([subj_features, body_features])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        base_phish = float(probs[1])
    else:
        base_phish = float(model.predict(features)[0])
    
    # heuristic
    h = heuristic_density(subject + " " + body)
    combined = alpha * base_phish + (1 - alpha) * h
    if force_boost and h >= boost_threshold:
        combined = max(combined, boost_to)

    label = "🚨 PHISHING" if combined >= 0.7 else "✅ LEGIT"
    return label, combined, base_phish, h

# ----- Example dropdown -----
examples = {
    "— pick an example —": ("",""),
    "Legit — Course Info": ("Rider Spring 2026 Course Selection", "Please review your course selection."),
    "Phishing — Account Scam": ("URGENT: Your account will be locked", "Click https://fake-login.example to verify."),
}

example_choice = st.selectbox("Choose an example email:", list(examples.keys()))
default_subj, default_body = examples.get(example_choice, ("",""))

subject_input = st.text_input("Email Subject:", value=default_subj)
body_input = st.text_area("Email Body:", value=default_body, height=200)

if st.button("Detect"):
    if subject_input.strip() == "" and body_input.strip() == "":
        st.warning("Please enter subject and/or body (or choose an example).")
    else:
        label, boosted_prob, base_prob, h_score = get_boosted_prediction(subject_input, body_input)
        st.write(f"Label: {label}")
        st.write(f"Boosted phishing probability: {boosted_prob:.2f}")
        st.write(f"Base model probability: {base_prob:.2f}")
        st.write(f"Heuristic score: {h_score:.2f}")
