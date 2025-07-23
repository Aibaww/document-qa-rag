import fitz
import tiktoken
import openai
import faiss
import csv
import json
import numpy as np
import os
from dotenv import load_dotenv

#
# SETUP
#
# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Constant file paths
PROCESSED_DB = "processed_files.json"
CSV_DIR = "csv_outputs"
os.makedirs(CSV_DIR, exist_ok=True)

#
# FUNCTIONS
#
# Load or create processed file database
def load_processed_db():
    if os.path.exists(PROCESSED_DB):
        with open(PROCESSED_DB, "r") as f:
            return json.load(f)
    return {}

def save_processed_db(db):
    with open(PROCESSED_DB, "w") as f:
        json.dump(db, f)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text.strip()

# Split text into chunks of max_token tokens
def chunk_text_by_tokens(text, max_tokens=300):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        token_count = len(tokenizer.encode(" ".join(current_chunk)))

        if token_count > max_tokens:
            # Remove last word and finalize current chunk
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]  # start new chunk with the word that caused overflow

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Generate embedding
def get_embedding(text_chunk, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text_chunk,
        model=model
    )
    return response.data[0].embedding

# Save to CSV
def save_to_csv(chunks, embeddings, file_path="pdf_chunks.csv"):
    with open(file_path, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "embedding", "n_tokens"])
        for chunk, embedding in zip(chunks, embeddings):
            n_tokens = len(tokenizer.encode(chunk))
            writer.writerow([chunk, str(embedding), n_tokens])

# Build cosine FAISS index
def build_faiss_cosine_index(embeddings):
    # Normalize for cosine similarity
    vectors = np.array(embeddings).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])  # IP = inner product = cosine if normalized
    index.add(vectors)
    return index

# Search FAISS with similarity filtering
def search_similar_paragraphs(index, query, chunks, threshold=0.5, top_k=5):
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_vector)

    D, I = index.search(query_vector, top_k)
    
    results = []
    for i, sim in zip(I[0], D[0]):
        if sim >= threshold:
            results.append({
                "text": chunks[i],
                "similarity": float(sim),
                "tokens": len(tokenizer.encode(chunks[i]))
            })
    return results


#
# PIPELINE
#
# --- Main pipeline ---
def process_pdf(pdf_path):
    print("Checking if already processed...")
    processed_db = load_processed_db()
    abs_path = os.path.abspath(pdf_path)

    if abs_path in processed_db:
        print("Already processed. Skipping embedding pipeline.")
        return processed_db[abs_path]

    print("Extracting text...")
    text = extract_text_from_pdf(pdf_path)
    print("Chunking...")
    chunks = chunk_text_by_tokens(text)
    print(f"Total chunks: {len(chunks)}")

    print("Generating embeddings...")
    embeddings = [get_embedding(chunk) for chunk in chunks]

    print("Saving to CSV...")
    file_id = os.path.splitext(os.path.basename(pdf_path))[0]
    csv_path = os.path.join(CSV_DIR, f"{file_id}_chunks.csv")
    save_to_csv(chunks, embeddings, csv_path)

    # Save FAISS index to memory (could persist later)
    print("Building FAISS index...")
    index = build_faiss_cosine_index(embeddings)

    # Save processed state
    processed_db[abs_path] = {
        "chunks": chunks,
        "csv_path": csv_path
    }
    save_processed_db(processed_db)

    return {"chunks": chunks, "csv_path": csv_path, "index": index, "embeddings": embeddings}

# --- CLI ---
def run_cli():
    print("PDF Semantic Search CLI")
    pdf_path = input("Enter path to a PDF file: ").strip()

    if not os.path.isfile(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        print("Invalid PDF path.")
        return

    result = process_pdf(pdf_path)
    chunks = result["chunks"]

    # If index not already present, rebuild from CSV
    if "index" in result:
        index = result["index"]
    else:
        # Rebuild index from CSV
        print("Rebuilding index from CSV...")
        with open(result["csv_path"], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            chunks = []
            embeddings = []
            for row in reader:
                chunks.append(row["text"])
                embedding = json.loads(row["embedding"].replace("'", '"'))  # ensure proper format
                embeddings.append(embedding)
        index = build_faiss_cosine_index(embeddings)

    print("\nType a question to search. Type 'quit' or 'end' to exit.")

    while True:
        query = input("\nYour query: ").strip()
        if query.lower() in {"quit", "end"}:
            print("Exiting. Goodbye!")
            break

        results = search_similar_paragraphs(index, query, chunks, threshold=0.5, top_k=5)
        if not results:
            print("No relevant paragraphs found above similarity threshold.")
        else:
            for i, r in enumerate(results):
                print(f"\nResult {i+1} (Similarity: {r['similarity']:.2f}) [{r['tokens']} tokens]:\n{r['text']}")


if __name__ == "__main__":
    run_cli()
