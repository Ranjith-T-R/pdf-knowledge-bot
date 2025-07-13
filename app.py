# knowledge_bot.py
# A Streamlit app for chatting with PDFs using RAG and OCR support
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import tempfile
import os
import gc
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please enter GOOGLE_API_KEY in your .env file.")

# Embedding Model (cached for performance)
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # huggingface embedding model

# Load and process PDFs
def load_docs(uploaded_files, enable_ocr=True): # Enable OCR for scanned PDFs
    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file: # Create a temporary file
            tmp_file.write(uploaded_file.read()) 
            tmp_file_path = tmp_file.name  

        text_content = ""
        try:
            reader = PdfReader(tmp_file_path)
            for page in reader.pages:
                text_content += page.extract_text() or ""
        except Exception as e:
            st.error(f"Text extraction failed: {e}") 

        ocr_text = ""
        if enable_ocr:
            try:
                images = convert_from_path(tmp_file_path)
                for img in images:
                    ocr_text += pytesseract.image_to_string(img)
                    del img 
                del images 
                gc.collect() # Clean up memory
            except Exception as e:
                st.error(f"OCR failed for {uploaded_file.name}: {e}") # Handle OCR errors

        full_text = text_content + "\n" + ocr_text # Combine text and OCR results
        doc = Document(page_content=full_text.strip(), metadata={"source": uploaded_file.name})
        docs.append(doc) 

        os.unlink(tmp_file_path)  # Clean temp file

    return docs

# Split large documents into chunks
def split_documents(docs, chunk_size=1000, overlap=200): # Adjust chunk size and overlap as needed
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

# Embed documents and build vectorstore
def embed_chunks(chunks):
    embeddings = get_embeddings() 
    return FAISS.from_documents(chunks, embedding=embeddings) 

# Load Gemini LLM
def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=0.3, # Adjust temperature for response creativity (0.0 to 1.0)
        google_api_key=GOOGLE_API_KEY
    )

# Create QA chain
def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # Adjust k for more or fewer results
    llm = get_llm() # Load the LLM
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Main Streamlit App
def main():
    st.set_page_config(page_title="Knowledge Bot", layout="wide")
    st.title(" Chat with Your PDFs with RAG and OCR Support") # Title of the app

    with st.sidebar:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        enable_ocr = st.checkbox("Enable OCR for scanned PDFs", value=True) # Checkbox for OCR support
        load_btn = st.button(" Load PDFs") # Button to load PDFs

    if load_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
            with st.spinner("Processing documents..."):
                raw_docs = load_docs(uploaded_files, enable_ocr=enable_ocr) # Load and process documents
                split_docs = split_documents(raw_docs) # Split documents  chunks
                vectorstore = embed_chunks(split_docs) # Embed the split documents
                st.session_state.qa_chain = build_qa_chain(vectorstore)
            st.success(" Documents are processed and ready for Q&A!")

    # Chat interface
    if "qa_chain" in st.session_state:
        query = st.chat_input("Ask a question about your uploaded documents...")
        if query:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.run(query)
            st.chat_message("user").markdown(query)
            if not response or len(response.strip()) < 10: # Check if response is empty or too short
                st.chat_message("assistant").warning("I couldn’t find that information in the documents.")
            else:
                st.chat_message("assistant").markdown(response)
    else:
        st.info("Upload your documents and click **Load PDFs** to start chatting.")

# Run app
if __name__ == "__main__":
    main()
