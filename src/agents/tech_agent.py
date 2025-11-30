"""
IT/Tech specialized RAG agent for handling technical support queries
"""

from typing import Dict, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma

from src.agents.base_agent import BaseRAGAgent
from src.utils.langfuse_setup import observe


class TechAgent(BaseRAGAgent):
    """Specialized agent for IT/Tech domain queries"""
    
    def __init__(self, vector_store: "Chroma"):
        """
        Initialize Tech agent
        
        Args:
            vector_store: Vector store containing IT/Tech documents
        """
        super().__init__(
            vector_store=vector_store,
            agent_name="Tech Agent",
            domain="IT/Tech"
        )
    
    @observe(name="tech_agent_query")
    def answer(self, query: str) -> Dict:
        """
        Answer IT/Tech-related query using RAG
        
        Args:
            query: User query
            
        Returns:
            Dictionary with answer, source documents, and agent info
        """
        if os.getenv("DISABLE_LLM", "0") == "1":
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})
            docs = retriever.invoke(query)
            sections = {
                "Password Reset": [],
                "Account & Access": [],
                "Security Practices": [],
                "Support Contact": []
            }
            keywords = {
                "Password Reset": ["password", "reset", "credential"],
                "Account & Access": ["account", "login", "access", "locked"],
                "Security Practices": ["security", "secure", "monitor", "device", "lock"],
                "Support Contact": ["support", "ticket", "helpdesk", "report"]
            }
            def clean_line(line: str) -> str:
                line = line.strip().replace("\t", " ")
                while "  " in line:
                    line = line.replace("  ", " ")
                line = line.lstrip("-•*0123456789. ")
                return line[:240]
            seen = {sec: set() for sec in sections}
            for d in docs:
                for raw in d.page_content.splitlines():
                    line = clean_line(raw)
                    low = line.lower()
                    for sec, kws in keywords.items():
                        if any(k in low for k in kws):
                            norm = line.lower()
                            if norm not in seen[sec]:
                                sections[sec].append(line)
                                seen[sec].add(norm)
            def fmt(sec, lines):
                if not lines:
                    return f"{sec}: (No explicit details found in retrieved excerpts.)"
                return f"{sec}:\n" + "\n".join(f"- {l}" for l in lines[:5])
            answer = "IT/Tech Guidance Summary (offline extraction):\n\n" + "\n\n".join(
                fmt(sec, lines) for sec, lines in sections.items()
            )
            sources = docs
        else:
            try:
                result = self.chain.invoke({"query": query})
                answer = result["result"]
                sources = result.get("source_documents", [])
            except Exception as e:
                print(f"Warning: LLM call failed ({type(e).__name__}: {e}), using fallback")
                retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(query)
                bullets = []
                for d in docs:
                    snippet = d.page_content.strip().splitlines()[0][:200]
                    bullets.append(f"- {snippet}")
                answer = "Based on IT/Tech documentation:\n" + "\n".join(bullets)
                sources = docs
        
        return {
            "answer": answer,
            "source_documents": sources,
            "agent": self.agent_name,
            "domain": self.domain
        }

