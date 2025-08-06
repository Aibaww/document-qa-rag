import csv
import json
import os
import tiktoken
from dotenv import load_dotenv
import openai
from utils.embedding import extract_text_from_pdf, chunk_json_pages_by_tokens, embed_pdf_chunks_with_text
from utils.rag import get_rag_response
from utils.database import load_processed_db, save_processed_db, build_faiss_cosine_index, search_similar_paragraphs

# Setup
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
tokenizer = tiktoken.get_encoding("cl100k_base")

# Constant file paths
CSV_DIR = "csv_outputs"
COMPANY = "Nvidia" # test
os.makedirs(CSV_DIR, exist_ok=True)

# Save to CSV
def save_embeddings_to_csv(embedded_data, output_file="citations.csv"):
    """
    Saves the embedded data to a CSV file for citation.

    Args:
        embedded_data (dict): Output from embed_pdf_chunks_with_text.
        output_file (str): Path to the output CSV file.
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["document", "page_number", "chunk", "embedding"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        document_name = embedded_data.get("document", "Unknown")

        for page in embedded_data.get("pages", []):
            page_number = page.get("page_number", -1)
            for entry in page.get("chunks", []):
                writer.writerow({
                    "document": document_name,
                    "page_number": page_number,
                    "chunk": entry["chunk"],
                    "embedding": ",".join(map(str, entry["embedding"]))  # flatten list
                })

#
# PIPELINE
#
# --- Main pipeline ---
def process_pdf(pdf_path, tokenizer, model="text-embedding-3-small", csv_dir="citations"):
    print("Checking if already processed...")
    processed_db = load_processed_db()
    abs_path = os.path.abspath(pdf_path)

    if abs_path in processed_db:
        print("Already processed. Skipping embedding pipeline.")
        return processed_db[abs_path]

    print("Extracting structured PDF text...")
    pdf_json = extract_text_from_pdf(pdf_path)

    print("Chunking text by tokens...")
    chunked_json = chunk_json_pages_by_tokens(pdf_json, tokenizer=tokenizer)

    print("Generating embeddings with chunk text and page info...")
    embedded_data = embed_pdf_chunks_with_text(chunked_json, model=model)

    print("Saving to CSV...")
    file_id = os.path.splitext(os.path.basename(pdf_path))[0]
    csv_path = os.path.join(csv_dir, f"{file_id}_chunks.csv")
    os.makedirs(csv_dir, exist_ok=True)
    save_embeddings_to_csv(embedded_data, output_file=csv_path)

    print("Flattening embeddings for FAISS...")
    flat_chunks = []
    flat_embeddings = []
    for page in embedded_data["pages"]:
        for entry in page["chunks"]:
            flat_chunks.append(entry["chunk"])
            flat_embeddings.append(entry["embedding"])

    print("Building FAISS index...")
    index = build_faiss_cosine_index(flat_embeddings)

    print("Saving processed state...")
    processed_db[abs_path] = {
        "chunks": flat_chunks,
        "csv_path": csv_path
    }
    save_processed_db(processed_db)

    return {
        "chunks": flat_chunks,
        "csv_path": csv_path,
        "index": index,
        "embeddings": flat_embeddings
    }

# --- CLI ---
def run_cli():
    print("PDF Semantic Search CLI")
    pdf_path = input("Enter path to a PDF file: ").strip()

    if not os.path.isfile(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        print("Invalid PDF path.")
        return

    result = process_pdf(pdf_path, tokenizer, model="text-embedding-3-small", csv_dir=CSV_DIR)
    chunks = result["chunks"]
    csv_path = result["csv_path"]

    # If index not already present, rebuild from CSV
    if "index" in result:
        index = result["index"]
    else:
        print("Rebuilding index from CSV...")
        chunks = []
        embeddings = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                chunks.append(row["chunk"])
                embedding = list(map(float, row["embedding"].split(",")))
                embeddings.append(embedding)
        index = build_faiss_cosine_index(embeddings)

    # Load citation map: chunk text → (doc, page)
    citation_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            citation_map[row["chunk"]] = {
                "document": row["document"],
                "page_number": row["page_number"]
            }

    print(len(citation_map), "chunks loaded with citations.")

    print("\nType a question to search. Type 'quit' or 'end' to exit.")

    while True:
        query = input("\nYour query: ").strip()
        if query.lower() in {"quit", "end"}:
            print("Exiting. Goodbye!")
            break

        results = search_similar_paragraphs(tokenizer, index, query, chunks, threshold=0.5, top_k=5)

        # Annotate results with document name and page number before passing to RAG
        for r in results:
            citation = citation_map.get(r["text"], {})
            r["document"] = citation.get("document", "Unknown")
            r["page_number"] = citation.get("page_number", "?")

        response = get_rag_response(query, COMPANY, results)
        print("\n--- LLM Response ---\n")
        print(response)

        # if results:
        #     print("\n--- Top Matches with Citations ---")
        #     for i, r in enumerate(results):
        #         print(f"\nResult {i+1} (Similarity: {r['similarity']:.2f}) [Page {r['page_number']}] - {r['document']}")
        #         print(r["text"])
        # else:
        #     print("\nNo relevant paragraphs found above similarity threshold.")



if __name__ == "__main__":
    run_cli()
