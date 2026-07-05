import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Company Policy RAG Chatbot", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc, #ecfeff);
}
.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 8px;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #475569;
    margin-bottom: 28px;
}
.card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}
.answer {
    background: #ffffff;
    padding: 22px;
    border-radius: 14px;
    border-left: 6px solid #2563eb;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    font-size: 17px;
    color: #0f172a;
}
.stButton > button {
    background: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 700;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Company Policy Q&A Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Free RAG app using Streamlit, Sentence Transformers, and FAISS</div>',
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def read_sample_file():
    with open("docs/sample_policy.txt", "r", encoding="utf-8") as file:
        return file.read()

def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_index(chunks, model):
    embeddings = model.encode(chunks, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_policy(question, chunks, model, index):
    query_embedding = model.encode([question], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")
    scores, indexes = index.search(query_embedding, 3)
    results = []
    for score, idx in zip(scores[0], indexes[0]):
        results.append((chunks[idx], score))
    return results

left, right = st.columns([1, 2])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Document")
    uploaded_file = st.file_uploader("Upload policy PDF", type=["pdf"])
    use_sample = st.checkbox("Use sample policy document", value=True)

    st.markdown("---")
    st.subheader("Example Questions")
    st.write("What is the vacation policy?")
    st.write("How many sick leaves are allowed?")
    st.write("What is the remote work policy?")
    st.write("What benefits are available?")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Ask a Policy Question")

    question = st.text_input("Enter your question", placeholder="Example: What is the remote work policy?")
    button = st.button("Get Answer")

    if button:
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching document using RAG..."):
                if uploaded_file:
                    text = read_pdf(uploaded_file)
                elif use_sample:
                    text = read_sample_file()
                else:
                    st.error("Please upload a PDF or use the sample document.")
                    st.stop()

                model = load_model()
                chunks = split_text(text)
                index = build_index(chunks, model)
                results = search_policy(question, chunks, model, index)

                st.markdown("### Answer")
                st.markdown(f'<div class="answer">{results[0][0]}</div>', unsafe_allow_html=True)

                st.markdown("### Source Matches")
                for i, (chunk, score) in enumerate(results, start=1):
                    with st.expander(f"Source Match {i} | Score: {score:.2f}"):
                        st.write(chunk)

    st.markdown("</div>", unsafe_allow_html=True)