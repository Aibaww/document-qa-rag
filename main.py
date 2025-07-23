import pdfplumber
import tiktoken
import openai
import faiss
import csv
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

#
# FUNCTIONS
#
# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text.strip()

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
def search_similar_paragraphs(index, query, chunks, threshold=0.7, top_k=5):
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
pdf_path = "test.pdf"

print("Extracting text...")
text = extract_text_from_pdf(pdf_path)

print("Chunking...")
chunks = chunk_text_by_tokens(text, max_tokens=300)

print("Embedding...")
embeddings = [get_embedding(chunk) for chunk in chunks]

print("Saving to CSV...")
save_to_csv(chunks, embeddings)

print("Building index...")
index = build_faiss_cosine_index(embeddings)

# Example query
query = "What are the main findings of the document?"
results = search_similar_paragraphs(index, query, chunks, threshold=0.7, top_k=5)

print("\nTop similar paragraphs:")
for r in results:
    print(f"\n(Similarity: {r['similarity']:.2f}) [{r['tokens']} tokens]")
    print(r['text'])