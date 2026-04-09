from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
import uvicorn

# --- Import your existing modules ---
from functions import get_embedding_functions
from populate_db import add_to_chroma, split_documents, clear_database
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ---- Config ----
DATA_PATH = "data"
CHROMA_PATH = "chroma"
GROQ_API_KEY = "GROQ_API_KEY"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

os.makedirs(DATA_PATH, exist_ok=True)

app = FastAPI(title="RAG Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


# ---- Routes ----

@app.get("/")
def root():
    return {"message": "RAG Pipeline API is running", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF to the data directory and index it into ChromaDB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest = os.path.join(DATA_PATH, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Re-index
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

    return {"message": f"'{file.filename}' uploaded and indexed successfully.", "filename": file.filename}


@app.post("/query")
def query(request: QueryRequest):
    """Query the RAG pipeline."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    embedding_function = get_embedding_functions()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(request.query, k=5)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found. Please upload PDFs first.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=request.query)

    model = ChatGroq(
        model="groq/compound",
        api_key=GROQ_API_KEY
    )
    response = model.invoke(prompt)

    sources = [doc.metadata.get("id", "unknown") for doc, _score in results]
    scores = [round(float(score), 4) for _, score in results]

    return {
        "answer": response.content,
        "sources": sources,
        "scores": scores,
        "query": request.query
    }


@app.delete("/reset")
def reset_database():
    """Clear the ChromaDB vector store."""
    clear_database()
    return {"message": "Database cleared successfully."}


@app.get("/documents")
def list_documents():
    """List all PDFs currently in the data directory."""
    if not os.path.exists(DATA_PATH):
        return {"documents": []}
    pdfs = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    return {"documents": pdfs, "count": len(pdfs)}



if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
