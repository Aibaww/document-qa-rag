import os
import tiktoken
from dotenv import load_dotenv
import openai
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from utils.embedding import extract_text_from_pdf, chunk_json_pages_by_tokens, embed_pdf_chunks_with_text, get_embedding
from utils.llm import get_rag_response
from db.models import DocumentChunk
from db.operations import search_similar


# Setup
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
tokenizer = tiktoken.get_encoding("cl100k_base")
# Database URL from .env
DATABASE_URL = os.getenv("POSTGRES_URL")

# Constants
CSV_DIR = "csv_outputs"
COMPANY = "Nvidia" # test
os.makedirs(CSV_DIR, exist_ok=True)
MAX_BATCH_SIZE = 10

def process_pdfs(pdf_paths, tokenizer, model="text-embedding-3-small", db_url=DATABASE_URL):
    assert len(pdf_paths) <= MAX_BATCH_SIZE, f"Max {MAX_BATCH_SIZE} PDFs allowed at a time."

    # Setup SQLAlchemy connection
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    all_results = []

    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        abs_path = os.path.abspath(pdf_path)
        file_id = os.path.splitext(os.path.basename(abs_path))[0]

        # Check if this document is already in the DB
        existing = session.query(DocumentChunk).filter_by(document_name=file_id).first()
        if existing:
            print(f"{pdf_path} already processed. Skipping.")
            continue

        print(f"Extracting structured text from {pdf_path}...")
        pdf_json = extract_text_from_pdf(pdf_path)

        print("Chunking...")
        chunked_json = chunk_json_pages_by_tokens(pdf_json, tokenizer=tokenizer)

        print("Embedding...")
        embedded_data = embed_pdf_chunks_with_text(chunked_json, model=model)

        print("Inserting into database...")
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

    session.close()
    return all_results

# --- CLI ---
def run_cli():
    print("PDF Semantic Search CLI")

    pdf_paths = input("Enter paths to PDF files (comma-separated): ").strip().split(",")
    pdf_paths = [p.strip() for p in pdf_paths if p.strip().lower().endswith(".pdf")]

    if not pdf_paths:
        print("No valid PDF files provided.")
        return

    # Process each PDF and store in ragdb
    process_pdfs(pdf_paths, tokenizer)

    print("\nType a question to search. Type 'quit' or 'end' to exit.")

    while True:
        query = input("\nYour query: ").strip()
        if query.lower() in {"quit", "end"}:
            print("Exiting. Goodbye!")
            break

        # Generate embedding for the query
        query_embedding = get_embedding(query, model="text-embedding-3-small")

        # Search in ragdb using pgvector
        top_chunks = search_similar(query_embedding, top_k=5)

        if not top_chunks:
            print("No relevant chunks found.")
            continue

        # Pass results to RAG model
        response = get_rag_response(query, COMPANY, top_chunks)

        print("\n--- LLM Response ---\n")
        print(response)

if __name__ == "__main__":
    run_cli()
