import streamlit as st
import joblib, re, numpy as np
from scipy.sparse import hstack
from urlextract import URLExtract
import tldextract

st.set_page_config(page_title="AI Phishing Email Detector Demo", layout="centered")

# --------- Load model and vectorizers ----------
MODEL_PATH = "phishing_model.pkl"
VEC_SUBJ_PATH = "vectorizer_subject.pkl"
VEC_BODY_PATH = "vectorizer_body.pkl"

try:
    clf = joblib.load(MODEL_PATH)
    vectorizer_subject = joblib.load(VEC_SUBJ_PATH)
    vectorizer_body = joblib.load(VEC_BODY_PATH)
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Make sure all .pkl files are in the repo.")
    st.stop()

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

# --------- Heuristic booster ----------
extractor = URLExtract()
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
    subj_feat = vectorizer_subject.transform([subj_clean])
    body_feat = vectorizer_body.transform([body_clean])
    features = hstack([subj_feat, body_feat])
    base_probs = clf.predict_proba(features)[0]
    base_phish = float(base_probs[1])
    h = heuristic_density(subject + " " + body)
    combined = alpha * base_phish + (1 - alpha) * h
    if force_boost and h >= boost_threshold:
        combined = max(combined, boost_to)
    return np.array([1.0 - combined, combined]), base_phish, h

# --------- Streamlit UI ----------
st.title("🚀 AI Phishing Email Detector (Boosted)")

subject_input = st.text_input("Email Subject:")
body_input = st.text_area("Email Body:", height=240)

if st.button("Detect"):
    if not subject_input and not body_input:
        st.warning("Enter a subject or body.")
    else:
        probs, base, h_score = combined_boosted_proba(subject_input, body_input)
        label = "🚨 PHISHING" if probs[1] >= 0.7 else "✅ LEGIT"
        st.write(f"**Prediction:** {label}")
        st.write(f"Boosted phishing probability: {probs[1]*100:.2f}%")
        st.write(f"Base model probability: {base*100:.2f}%")
        st.write(f"Heuristic score: {h_score:.3f}")
