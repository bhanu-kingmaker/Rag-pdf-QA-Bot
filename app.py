import os
import shutil
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(working_dir, "doc_vectorstore")

st.title("🦙 Llama-3.3-70B - Document RAG")

# Track file upload state
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Only clear vector DB if a new file is uploaded
    if uploaded_file.name != st.session_state.last_uploaded_file:
        # Delete vector DB
        shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)

        # Delete all old PDFs in working_dir
        for fname in os.listdir(working_dir):
            if fname.lower().endswith(".pdf") and fname != uploaded_file.name:
                try:
                    os.remove(os.path.join(working_dir, fname))
                except Exception as e:
                    st.warning(f"Could not delete {fname}: {e}")

        # Save uploaded file
        save_path = os.path.join(working_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process new file
        process_document_to_chroma_db(uploaded_file.name)
        st.session_state.last_uploaded_file = uploaded_file.name
        st.success("✅ Document processed, old PDFs deleted, and vector DB updated.")

# Question input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    answer = answer_question(user_question)
    st.markdown("### Llama-3.3-70B Response")
    st.markdown(answer)