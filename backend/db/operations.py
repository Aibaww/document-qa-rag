from db.database import SessionLocal
from db.models import DocumentChunk

def insert_chunk(doc_name, page_num, chunk_text, embedding):
    session = SessionLocal()
    try:
        chunk = DocumentChunk(
            document_name=doc_name,
            page_number=page_num,
            chunk=chunk_text,
            embedding=embedding
        )
        session.add(chunk)
        session.commit()
    finally:
        session.close()

def search_similar(query_embedding, top_k=5):
    session = SessionLocal()
    try:
        results = session.query(DocumentChunk).order_by(
            DocumentChunk.embedding.l2_distance(query_embedding)
        ).limit(top_k).all()
        return results
    finally:
        session.close()