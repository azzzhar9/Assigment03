"""
Configuration management for the multi-agent system
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration for the multi-agent system"""
    
    # API Configuration (supports OpenAI or OpenRouter)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    # If using OpenRouter key, default the base URL to OpenRouter unless explicitly overridden
    OPENAI_BASE_URL = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or ("https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else "https://api.openai.com/v1")
    )
    # Default to an OpenRouter-routable alias if not provided
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openrouter/auto")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Langfuse Configuration
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE = "chroma"  # Options: chroma, faiss
    CHROMA_PERSIST_DIR = "./chroma_db"
    
    # Document Processing Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # RAG Configuration
    TOP_K_RETRIEVAL = 5
    TEMPERATURE = 0.0
    
    # Data Directories
    DATA_DIR = "./data"
    HR_DOCS_DIR = os.path.join(DATA_DIR, "hr_docs")
    TECH_DOCS_DIR = os.path.join(DATA_DIR, "tech_docs")
    FINANCE_DOCS_DIR = os.path.join(DATA_DIR, "finance_docs")
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY is required")
        if os.getenv("OPENROUTER_API_KEY") and "openrouter.ai" not in cls.OPENAI_BASE_URL:
            print("Warning: OPENROUTER_API_KEY detected but base URL is not OpenRouter; using", cls.OPENAI_BASE_URL)
        # Langfuse is optional for observability
        if not cls.LANGFUSE_PUBLIC_KEY or not cls.LANGFUSE_SECRET_KEY:
            print("Warning: Langfuse keys not set - observability disabled")
        return True

