"""
Vector store setup and management for RAG agents
"""

import os
import hashlib
from typing import List, Optional
from dataclasses import dataclass
from langchain.schema import Document, BaseRetriever
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Config


class SimpleEmbeddings:
    """Simple deterministic embeddings using hash-based approach"""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for documents"""
        return [self._hash_to_vector(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query"""
        return self._hash_to_vector(text)
    
    def _hash_to_vector(self, text: str, dim: int = 384) -> List[float]:
        """Convert text to vector using hash-based approach"""
        # Create deterministic hash
        hash_obj = hashlib.sha256(text.lower().encode())
        hash_bytes = hash_obj.digest()
        
        # Expand to desired dimension
        vector = []
        for i in range(dim):
            byte_idx = i % len(hash_bytes)
            vector.append((hash_bytes[byte_idx] / 255.0) - 0.5)
        
        # Normalize
        magnitude = sum(x**2 for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector


class TFIDFRetriever(BaseRetriever):
    docs: List[Document]
    tfidf: Any
    matrix: Any
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        q_vec = self.tfidf.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idxs = sims.argsort()[::-1][: self.top_k]
        return [self.docs[i] for i in idxs]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


class SimpleVectorStore:
    def __init__(self, documents: List[Document], embeddings: SimpleEmbeddings):
        self.documents = documents
        self.embeddings = embeddings
        self._tfidf = TfidfVectorizer(max_features=4096)
        self._matrix = self._tfidf.fit_transform([d.page_content for d in documents])

    def as_retriever(self, search_kwargs: Optional[dict] = None) -> TFIDFRetriever:
        top_k = (search_kwargs or {}).get("k", 5)
        return TFIDFRetriever(docs=self.documents, tfidf=self._tfidf, matrix=self._matrix, top_k=top_k)



class VectorStoreManager:
    """Manages vector stores for different domains"""
    
    def __init__(self, embedding_model: Optional[str] = None):
        """Initialize vector store manager with simple embeddings"""
        # Use simple hash-based embeddings (no dependencies)
        self.embeddings = SimpleEmbeddings()
        self.vector_stores = {}
    
    def create_vector_store(
        self,
        documents: List[Document],
        collection_name: str,
        persist_directory: Optional[str] = None
    ) -> SimpleVectorStore:
        """Create an in-memory TF-IDF vector store from documents"""
        print(f"DEBUG: Creating vector store for {collection_name} with {len(documents)} documents")
        vectorstore = SimpleVectorStore(documents=documents, embeddings=self.embeddings)
        print(f"DEBUG: In-memory vector store ready: {collection_name}")
        self.vector_stores[collection_name] = vectorstore
        return vectorstore
    
    def load_vector_store(
        self,
        collection_name: str,
        persist_directory: Optional[str] = None
    ) -> Optional[SimpleVectorStore]:
        """
        Load an existing vector store
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory where vector store is persisted
            
        Returns:
            VectorStore instance or None if not found
        """
        # No persistence for simple in-memory store; always return None to rebuild
        return None
    
    def get_vector_store(self, collection_name: str) -> Optional[SimpleVectorStore]:
        """
        Get a vector store by collection name
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            VectorStore instance or None
        """
        return self.vector_stores.get(collection_name)

