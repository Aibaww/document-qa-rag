import pdfplumber
import tiktoken
import os
import openai

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

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

# Read pdf
pdf_path = "test.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text_by_tokens(text, max_tokens=300)

# # Print result
# for i, chunk in enumerate(chunks):
#     print(f"--- Chunk {i + 1} ({len(tokenizer.encode(chunk))} tokens) ---")
#     print(chunk)
#     print()

# Get embeddings
embeddings = []
for i, chunk in enumerate(chunks):
    embedding = get_embedding(chunk)
    embeddings.append({
        "chunk_index": i,
        "token_count": len(tokenizer.encode(chunk)),
        "text": chunk,
        "embedding": embedding
    })
    print(f"Chunk {i + 1} embedded.")