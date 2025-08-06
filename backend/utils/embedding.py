import os
import fitz
import openai



def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file, including page numbers and the document name.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text with page numbers and document name included.
    """

    doc_name = os.path.basename(pdf_path)
    pages = []

    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            pages.append({
                "page_number": page_number,
                "text": page.get_text().strip()
            })

    return {
        "document": doc_name,
        "pages": pages
    }


def chunk_json_pages_by_tokens(pdf_json, tokenizer, max_tokens=300):
    """
    Takes a JSON structure of PDF content and chunks the text on each page 
    into smaller pieces with a maximum number of tokens.

    Args:
        pdf_json (dict): The structured PDF content from extract_text_from_pdf().
        tokenizer (Tokenizer): A tokenizer with an .encode() method.
        max_tokens (int): Maximum number of tokens per chunk.

    Returns:
        dict: A new dictionary with document name and pages, each containing 
              page number and a list of text chunks.
    """
    chunked_pages = []

    for page in pdf_json.get("pages", []):
        text = page.get("text", "")
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            token_count = len(tokenizer.encode(" ".join(current_chunk)))

            if token_count > max_tokens:
                current_chunk.pop()
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        chunked_pages.append({
            "page_number": page["page_number"],
            "chunks": chunks
        })

    return {
        "document": pdf_json.get("document"),
        "pages": chunked_pages
    }

def embed_pdf_chunks_with_text(chunked_pdf_json, model="text-embedding-3-small"):
    """
    Takes a chunked PDF JSON structure and returns embeddings for each chunk,
    along with the original chunk text.

    Args:
        chunked_pdf_json (dict): Output from chunk_json_pages_by_tokens.
        model (str): OpenAI embedding model name.

    Returns:
        dict: A structured dictionary with document name and pages.
              Each page contains a list of {"chunk", "embedding"} entries.
    """
    embedded_pages = []

    for page in chunked_pdf_json.get("pages", []):
        chunk_entries = []
        for chunk in page.get("chunks", []):
            embedding = get_embedding(chunk, model=model)
            chunk_entries.append({
                "chunk": chunk,
                "embedding": embedding
            })

        embedded_pages.append({
            "page_number": page["page_number"],
            "chunks": chunk_entries
        })

    return {
        "document": chunked_pdf_json.get("document"),
        "pages": embedded_pages
    }

def get_embedding(text_chunk, model="text-embedding-3-small"):
    """
    Takes a text chunk and returns its embedding.

    Args:
        text_chunk (str): The text chunk to embed.
        model (str): OpenAI embedding model name.

    Returns:
        list: The embedding vector for the text chunk.
    """
    response = openai.embeddings.create(
        input=text_chunk,
        model=model
    )
    return response.data[0].embedding