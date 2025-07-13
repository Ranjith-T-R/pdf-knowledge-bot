# PDF Knowledge Bot – Chat with Your PDFs

**Knowledge Bot** is an AI-powered PDF question-answering system that allows users to **chat with uploaded documents** using a combination of OCR, text parsing, vector search, and large language models. It’s built with **LangChain’s Retrieval-Augmented Generation (RAG)** framework and powered by **Google Gemini** for real-time, intelligent responses.

---

##  What is Knowledge Bot?

Knowledge Bot enables users to query complex documents such as scanned PDFs, multi-page manuals, reports, or handbooks. It combines:

- **OCR (Tesseract)** – (when needed) To extract text from scanned/image-based PDFs  
- **PDF parsing (PyPDF2)** – To extract structured text  
- **Vector Search (FAISS)** – For semantic retrieval  
- **LLM (Gemini 1.5 Flash)** – For context-aware natural language responses  
- **LangChain (RAG)** – For orchestrating retrieval and reasoning  
- **Streamlit** – For a clean, interactive user interface

---

##  Features:

-  Upload multiple PDFs via Streamlit sidebar  
-  Automatically extract and chunk content using PyPDF2 and OCR  
-  Store document vectors in memory using FAISS  
-  Real-time Q&A powered by Gemini  
-  Graceful fallback for out-of-context questions  
-  Conversational support for follow-up queries within the session  

---

## Tech Stack:

| Layer               | Technology                          |
|--------------------|--------------------------------------|
| **Interface**       | Streamlit (Python)                   |
| **Document Parsing**| PyPDF2 + Tesseract OCR               |
| **Embeddings**      | HuggingFace (MiniLM-L6-v2)           |
| **Vector DB**       | FAISS (in-memory)                    |
| **LLM**             | Gemini 1.5 Flash (Google GenAI API)  |
| **Orchestration**   | LangChain                            |
| **Hosting**         | Localhost                            |

---

##  Limitations:

### 1. Upload Size Limit  
- Maximum total upload size is **200 MB**, due to Streamlit's file uploader limitations.  
- Uploading very large files or multiple PDFs at once may result in memory issues or timeouts.

### 2. PDF-Only Support  
- The application currently supports only `.pdf` files.  
- Other file types like `.docx`, `.txt`, `.pptx`, etc., are **not yet supported**.

### 3. Slow Processing for Large or Scanned Files  
- Processing performance may slow down for:
  - PDFs with **many pages**
  - **Scanned or image-based** PDFs requiring OCR  
- This delay is due to the combination of **PyPDF2 (text extraction)** and **Tesseract OCR** for image text.

### 4. In-Memory Vector Storage  
- The FAISS vector store is **not persistent**.  
- All data is stored **in-memory**,
  - for followup Questions: 
  - Chat history is lost when the app reloads or restarts  
  - Re-uploading files is required for new sessions

  
##  Setup and Step by Step Instructions:

### Clone the Repository: 

```bash
https://github.com/Ranjith-T-R/pdf-knowledge-bot.git
cd knowledge-bot
---

Install Dependencies:

Create and activate a virtual environment (optional but recommended):

python -m venv venv :

venv\Scripts\activate  # Windows

Then install the required packages:

pip install -r requirements.txt

Create .env File:

In the root directory, create a .env file with your Google GenAI API key:
Refer Support Docs for key,

GOOGLE_API_KEY=your_google_genai_api_key

Run the App:

streamlit run app.py

Additional System Requirements (not installable via pip):

You must install these on your system (outside Python)
1.Poppler (windows)
2.Tesseract (Windows)
refer support docs for guide 
