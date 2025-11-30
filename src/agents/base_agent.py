"""
Base agent class with shared functionality for specialized RAG agents
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma

from src.config import Config
from src.utils.langfuse_setup import get_langfuse_callback


class BaseRAGAgent(ABC):
    """Base class for specialized RAG agents"""
    
    def __init__(
        self,
        vector_store: "Chroma",
        agent_name: str,
        domain: str,
        temperature: float = 0.0
    ):
        """
        Initialize base RAG agent
        
        Args:
            vector_store: Vector store for document retrieval
            agent_name: Name of the agent
            domain: Domain this agent handles (e.g., "HR", "Tech", "Finance")
            temperature: LLM temperature
        """
        self.vector_store = vector_store
        self.agent_name = agent_name
        self.domain = domain
        self.temperature = temperature
        
        # Initialize LLM (supports OpenAI or OpenRouter)
        callback = get_langfuse_callback()
        callbacks = [callback] if callback else []
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=temperature,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            callbacks=callbacks,
            max_retries=2,
            timeout=15,
            request_timeout=15
        )
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create retrieval chain
        self.chain = self._create_chain()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for the agent"""
        template = f"""You are a specialized {self.domain} assistant for a SaaS company.
    Your role is to provide accurate, helpful, and actionable answers strictly based on the company's internal documentation.

    Instructions:
    - Answer ONLY using the provided context; do not invent details.
    - If the context lacks specifics, explicitly state what is unknown.
    - Prefer structured steps, bullets, and concrete details (numbers, timelines, contacts) when available.
    - End with a short, clear next-step if appropriate.

    Context:
    {{context}}

    Question:
    {{question}}

    Now provide a precise, complete, and actionable answer based on the context above."""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_chain(self) -> RetrievalQA:
        """Create the retrieval QA chain"""
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": Config.TOP_K_RETRIEVAL}
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True,
            verbose=False
        )
        
        return chain
    
    @abstractmethod
    def answer(self, query: str) -> dict:
        """
        Answer a query using RAG
        
        Args:
            query: User query
            
        Returns:
            Dictionary with 'answer', 'source_documents', and 'agent' keys
        """
        pass

