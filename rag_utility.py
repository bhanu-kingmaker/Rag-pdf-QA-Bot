import os
import shutil
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment variables
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(working_dir, "doc_vectorstore")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def process_document_to_chroma_db(file_name: str):
    """Process a PDF into Chroma DB, using OCR if no text is extractable."""
    shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)

    file_path = os.path.join(working_dir, file_name)

    # Try OCR directly (for scanned/handwritten PDFs)
    pages = convert_from_path(file_path)
    documents = []
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page)
        if text.strip():
            documents.append(Document(page_content=text, metadata={"page": i+1}))

    if not documents:
        raise ValueError(f"No text could be extracted from {file_name} even with OCR.")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create vector DB
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=VECTOR_STORE_PATH
    )
    return vectordb

def answer_question(user_question: str):
    vectordb = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embedding)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    response = qa_chain.invoke({"query": user_question})
    return response.get("result", "").strip()

if __name__ == "__main__":
    try:
        vectordb = process_document_to_chroma_db("ENVIRONMENTAL EDUCATION.pdf")
        response = answer_question("List the important questions in the notes")
        print(response)
    except ValueError as e:
        print("❌ Error:", e)