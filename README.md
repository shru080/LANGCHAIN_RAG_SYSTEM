RAG Pipeline — Document Q&A System
Retrieval-Augmented Generation with FastAPI, ChromaDB & Groq
Version 1.0.0
Overview
This project is a full-stack Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions about their content. The system retrieves the most relevant chunks from the document using vector similarity search, then passes them to a large language model to generate accurate, context-aware answers.

Technology Stack

Component	Technology
Embeddings	HuggingFace — sentence-transformers/all-mpnet-base-v2
Vector Store	ChromaDB (local persistent storage)
LLM	Groq — compound model
Backend API	FastAPI + Uvicorn
Frontend	HTML / CSS / JavaScript
PDF Loading	LangChain PyPDFDirectoryLoader
Text Splitting	LangChain RecursiveCharacterTextSplitter

Project Structure

project/
├── api.py              ← FastAPI backend server
├── functions.py        ← Embedding + document loading
├── populate_db.py      ← ChromaDB indexing logic
├── query_db.py         ← RAG query logic (CLI)
├── main.py             ← CLI entry point
├── demo.html           ← Browser demo UI
├── requirements.txt    ← Python dependencies
└── data/               ← Folder where PDFs are stored

Setup & Installation
Prerequisites
•	Python 3.9 or higher
•	pip package manager
•	HuggingFace account (for embedding API token)
•	Groq account (for LLM API key)

Step 1 — Clone or download the project
Place all project files in the same folder and create the data directory:

mkdir data

Step 2 — Install dependencies
pip install -r requirements.txt

Step 3 — Configure API keys
Update the following files with your own API keys:

functions.py — HuggingFace token
huggingfacehub_api_token="hf_YOUR_TOKEN_HERE"

Get your token at: https://huggingface.co/settings/tokens (Read permission is sufficient)

api.py — Groq API key
GROQ_API_KEY = "gsk_YOUR_KEY_HERE"

Get your key at: https://console.groq.com/keys

Step 4 — Fix functions.py
Remove the bare module-level call on line 27 of functions.py:

# REMOVE this line:
documents = load_documents()

This line runs on every import and crashes if the data/ folder is empty.

Step 5 — Start the API
python api.py

The server starts at http://localhost:8000. Visit http://localhost:8000/docs for the interactive Swagger UI.

Step 6 — Open the Demo UI
Serve the demo HTML locally to avoid browser CORS restrictions:

python -m http.server 3000

Then open http://localhost:3000/demo.html in your browser.

API Reference

Method	Endpoint	Description
GET	/	Health check — returns API status
GET	/health	Returns healthy status
GET	/documents	Lists all indexed PDF files
POST	/upload	Upload a PDF and auto-index into ChromaDB
POST	/query	Send a question, get an answer with sources
DELETE	/reset	Clear the entire ChromaDB vector store

Query Request Body
POST /query
{
  "query": "What are the main topics in this document?"
}

Query Response
{
  "answer": "The document covers...",
  "sources": ["data/file.pdf:0:0", "data/file.pdf:1:2"],
  "scores": [0.312, 0.445],
  "query": "What are the main topics?"
}

How It Works

The pipeline follows these steps:

1.	PDF Upload — File is saved to the data/ folder
2.	Chunking — Document is split into 800-character overlapping chunks
3.	Embedding — Each chunk is converted to a vector using HuggingFace
4.	Storage — Vectors are stored in ChromaDB on disk
5.	Query — User question is embedded and top 5 similar chunks are retrieved
6.	Generation — Chunks + question are sent to Groq LLM as context
7.	Response — Answer is returned with source chunk IDs and similarity scores

Troubleshooting

Error	Fix
pytestconda not found	Use the new requirements.txt — pytestconda is a typo in the original
conda not installable via pip	Skip conda; install packages individually with pip
HuggingFace 401 error	Generate a new Read token at huggingface.co/settings/tokens
Groq 401 AuthenticationError	Generate a new API key at console.groq.com/keys
Connection failed in demo UI	Serve demo.html via python -m http.server 3000
load_documents() crash on import	Remove the bare documents = load_documents() line in functions.py

Notes
•	API keys are hardcoded for demo purposes. For production, use environment variables.
•	ChromaDB stores data in the chroma/ folder. Delete it to reset the vector store.
•	The embedding model runs via HuggingFace API. For offline use, switch to HuggingFaceEmbeddings (local).
•	Chunk size is set to 800 characters with 80-character overlap. Adjust in populate_db.py for your use case.


Built with LangChain · ChromaDB · FastAPI · Groq
