import os
import json
import numpy as np
import faiss
from utils.embedding import get_embedding

PROCESSED_DB = "processed_files.json"

# Load or create processed file database
def load_processed_db():
    if os.path.exists(PROCESSED_DB):
        with open(PROCESSED_DB, "r") as f:
            return json.load(f)
    return {}

def save_processed_db(db):
    with open(PROCESSED_DB, "w") as f:
        json.dump(db, f)

# Build cosine FAISS index
def build_faiss_cosine_index(embeddings):
    # Normalize for cosine similarity
    vectors = np.array(embeddings).astype("float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])  # IP = inner product = cosine if normalized
    index.add(vectors)
    return index

# Search FAISS with similarity filtering
def search_similar_paragraphs(tokenizer, index, query, chunks, threshold=0.5, top_k=5):
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