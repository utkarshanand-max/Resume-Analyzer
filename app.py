import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

# -----------------------------------
# 🧹 Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation/numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# -----------------------------------
# 📄 PDF Text Extraction
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text

# -----------------------------------
# 🧠 Compute Similarity
def compute_similarity(resume_text, job_description):
    texts = [clean_text(job_description), clean_text(resume_text)]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(texts)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return round(float(similarity[0][0] * 100), 2)

# -----------------------------------
# 🧾 Suggest missing keywords
def suggest_keywords(resume_text, job_description):
    resume_words = set(clean_text(resume_text).split())
    jd_words = set(clean_text(job_description).split())
    missing = jd_words - resume_words
    return list(missing)[:10]  # Limit to 10

# -----------------------------------
# 🖥️ Streamlit App
st.set_page_config(page_title="Resume Analyzer", layout="wide")
st.title("📄 AI-Powered Resume Analyzer")
st.write("Upload your resume and enter a job description to see how well it matches!")

uploaded_files = st.file_uploader("Upload PDF Resume(s)", type="pdf", accept_multiple_files=True)
job_description = st.text_area("Paste the Job Description here")

if uploaded_files and job_description and st.button("Analyze"):
    st.subheader("📊 Results")

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        resume_text = extract_text_from_pdf(tmp_path)
        score = compute_similarity(resume_text, job_description)
        suggestions = suggest_keywords(resume_text, job_description)

        st.markdown(f"### 🧾 {uploaded_file.name}")
        st.write(f"**🔍 Match Score:** `{score}%`")

        if score >= 75:
            st.success("✅ Great match!")
        elif score >= 50:
            st.warning("⚠️ Moderate match. Consider improving your resume.")
        else:
            st.error("❌ Poor match. Resume needs improvement.")

        st.markdown("**📌 Suggested Keywords to Add:**")
        st.write(", ".join(suggestions))

        os.remove(tmp_path)
