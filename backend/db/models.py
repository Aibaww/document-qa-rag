from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.dialects.postgresql import VARCHAR
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_name = Column(String, nullable=False)
    page_number = Column(Integer, nullable=False)
    chunk = Column(String, nullable=False)
    embedding = Column(Vector(1536))