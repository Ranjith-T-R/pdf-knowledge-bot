<<<<<<< HEAD
# Knowledge Bot 
# A Streamlit app to chat with your PDFs using RAG and OCR support.
=======
# knowledge Bot
# Import necessary libraries
>>>>>>> 35e341442d668bf97ee58ffe6a2e7efdfc7e1611
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
<<<<<<< HEAD
import tempfile
import os
import gc
=======
>>>>>>> 35e341442d668bf97ee58ffe6a2e7efdfc7e1611
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
<<<<<<< HEAD

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please enter GOOGLE_API_KEY in your .env file.")

# Embedding Model (cached for performance)
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # HuggingFace model for embeddings

# Load and process PDFs
def load_docs(uploaded_files, enable_ocr=True):
    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file: # Create a temporary file
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name 

=======
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
>>>>>>> 35e341442d668bf97ee58ffe6a2e7efdfc7e1611
        text_content = ""
        try:
            reader = PdfReader(tmp_file_path)
            for page in reader.pages:
                text_content += page.extract_text() or ""
        except Exception as e:
<<<<<<< HEAD
            st.error(f"Text extraction failed: {e}")

        ocr_text = ""
        if enable_ocr:
            try:
                images = convert_from_path(tmp_file_path)
                for img in images:
                    ocr_text += pytesseract.image_to_string(img)
                    del img
                del images # Clean up images
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
    return splitter.split_documents(docs) # Split documents into smaller chunks...

# Embed documents and build vectorstore
def embed_chunks(chunks):
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)

# Load Gemini LLM
def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=0.3, # Adjust temperature for response creativity (0.0 to 0.9)
        google_api_key=GOOGLE_API_KEY 
    )

# Create QA chain
def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # Adjust k for number of retrieved documents
    llm = get_llm()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Main Streamlit App
def main():
    st.set_page_config(page_title="Knowledge Bot", layout="wide") 
    st.title("Chat with Your PDFs with RAG and OCR Support") #title
    with st.sidebar:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        enable_ocr = st.checkbox("Enable OCR for scanned PDFs", value=True) #checkbox for OCR
        load_btn = st.button(" Load PDFs") #load button

=======
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
>>>>>>> 35e341442d668bf97ee58ffe6a2e7efdfc7e1611
    if load_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        else:
<<<<<<< HEAD
            with st.spinner("Processing documents..."):
                raw_docs = load_docs(uploaded_files, enable_ocr=enable_ocr) # Load and process the uploaded PDFs
                split_docs = split_documents(raw_docs) # Split documents into manageable chunks
                vectorstore = embed_chunks(split_docs) # Embed the split documents
                st.session_state.qa_chain = build_qa_chain(vectorstore)
            st.success("Documents are processed and ready for Q&A!")

    # Chat interface
    if "qa_chain" in st.session_state:
        query = st.chat_input("Ask a question about your uploaded documents...")
        if query:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.run(query) # QA chain with the user query
            st.chat_message("user").markdown(query)
            if not response or len(response.strip()) < 5: # Check if response is empty or too short
                st.chat_message("assistant").warning("I couldn’t find that information in the documents.")
            else:
                st.chat_message("assistant").markdown(response) # Display the response
    else:
        st.info(" Upload your documents and click **Load PDFs** to start chatting.")

# Run app
=======
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
>>>>>>> 35e341442d668bf97ee58ffe6a2e7efdfc7e1611
if __name__ == "__main__":
    main()
