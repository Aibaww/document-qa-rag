from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
import os
import uuid

from utils.embedding import extract_text_from_pdf, chunk_json_pages_by_tokens, embed_pdf_chunks_with_text, get_embedding
from utils.llm import get_rag_response
from db.models import DocumentChunk
from db.operations import search_similar

import tiktoken
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import openai
from dotenv import load_dotenv

# Load env variables and initialize globals
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("POSTGRES_URL")

tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_BATCH_SIZE = 10
CSV_DIR = "csv_outputs"
COMPANY = "Nvidia"  # Keep consistent with your existing code

# Setup DB session factory
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

app = FastAPI()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

def process_pdfs(pdf_paths, tokenizer, model="text-embedding-3-small", db_session=None):
    if len(pdf_paths) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Max {MAX_BATCH_SIZE} PDFs allowed at a time.")
    session = db_session or Session()

    all_results = []
    for pdf_path in pdf_paths:
        file_id = os.path.splitext(os.path.basename(pdf_path))[0]
        existing = session.query(DocumentChunk).filter_by(document_name=file_id).first()
        if existing:
            continue  # skip already processed PDFs

        pdf_json = extract_text_from_pdf(pdf_path)
        chunked_json = chunk_json_pages_by_tokens(pdf_json, tokenizer=tokenizer)
        embedded_data = embed_pdf_chunks_with_text(chunked_json, model=model)

        for page in embedded_data["pages"]:
            page_num = page["page_number"]
            for entry in page["chunks"]:
                chunk_text = entry["chunk"]
                embedding = entry["embedding"]
                session.add(DocumentChunk(
                    document_name=file_id,
                    page_number=page_num,
                    chunk=chunk_text,
                    embedding=embedding
                ))
        session.commit()

        flat_chunks = [entry["chunk"] for page in embedded_data["pages"] for entry in page["chunks"]]
        flat_embeddings = [entry["embedding"] for page in embedded_data["pages"] for entry in page["chunks"]]

        all_results.append({
            "pdf": pdf_path,
            "chunks": flat_chunks,
            "embeddings": flat_embeddings
        })

    if db_session is None:
        session.close()
    return all_results


@app.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"Max {MAX_BATCH_SIZE} files allowed.")

    # Save files temporarily
    saved_paths = []
    try:
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")

            temp_filename = f"temp_{uuid.uuid4().hex}.pdf"
            temp_path = os.path.join("temp_uploads", temp_filename)
            os.makedirs("temp_uploads", exist_ok=True)

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(temp_path)

        # Process PDFs and insert embeddings into DB
        results = process_pdfs(saved_paths, tokenizer)

    finally:
        # Cleanup temp files
        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path)

    return {"status": "success", "processed_pdfs": [os.path.basename(p) for p in saved_paths], "results_count": len(results)}


@app.post("/query")
async def query_pdf(request: QueryRequest):
    query = request.query
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    # Create DB session
    session = Session()

    try:
        # Get embedding for query
        query_embedding = get_embedding(query, model="text-embedding-3-small")

        # Search DB for similar chunks
        top_chunks = search_similar(query_embedding, top_k=5)

        if not top_chunks:
            return {"response": "No relevant chunks found."}

        # Generate RAG response
        response = get_rag_response(query, COMPANY, top_chunks)
    finally:
        session.close()

    return {"response": response}