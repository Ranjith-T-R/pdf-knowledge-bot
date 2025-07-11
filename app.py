# knowledge Bot
# Import necessary libraries
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import os

# Load .env file for API keys 
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please enter GOOGLE_API_KEY in .env file.")

# Load and process uploaded PDF documents
def load_docs(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Extract text
        text_content = ""
        try:
            reader = PdfReader(tmp_file_path)
            for page in reader.pages:
                text_content += page.extract_text() or ""
        except Exception as e:
            print("Text extraction error:", e)

        # OCR
        ocr_text = ""
        try:
            images = convert_from_path(tmp_file_path)
            for img in images:
                ocr_text += pytesseract.image_to_string(img)
        except Exception as e:
            print("OCR error:", e)

        full_text = text_content + "\n" + ocr_text
        doc = Document(page_content=full_text, metadata={"source": uploaded_file.name})
        docs.append(doc)

        os.unlink(tmp_file_path)

    return docs

# Chunk documents
def split_documents(docs, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

# Embed documents
def embed_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding=embeddings)

# Create LLM
def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

# Build retrieval QA chain
def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # increased k for better context
    llm = get_llm()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

#  Main Streamlit App Function
def main():
    st.set_page_config(page_title="Knowledge Bot")
    st.title("Chat with Your PDFs with RAG and OCR Support")

    # Sidebar
    with st.sidebar:
        st.header(" Upload PDF Files")
        uploaded_files = st.file_uploader("Choose one or more PDFs", type="pdf", accept_multiple_files=True)
        load_btn = st.button("Load PDFs")

    # Handle PDF Loading
    if load_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing uploaded PDFs..."):
                raw_docs = load_docs(uploaded_files)
                split_docs = split_documents(raw_docs)
                vectorstore = embed_chunks(split_docs)
                st.session_state.qa_chain = build_qa_chain(vectorstore)
            st.success("Documents processed and ready to chat!")

    # Chat Interface
    if "qa_chain" in st.session_state:
        query = st.chat_input("Ask a question about your uploaded Docs...")
        if query:
            response = st.session_state.qa_chain.run(query)
            st.chat_message("user").markdown(query)
            if not response or len(response.strip()) < 10:
                st.warning("I couldn’t find that information in the provided documents.")
            else:
                st.chat_message("assistant").markdown(response)
    else:
        st.info(" Upload Docs in the sidebar and click 'Load Docs' to start.")


# Entry point
if __name__ == "__main__":
    main()
