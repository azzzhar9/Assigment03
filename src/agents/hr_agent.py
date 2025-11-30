"""
HR specialized RAG agent for handling HR-related queries
"""

from typing import Dict, TYPE_CHECKING
import os

if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma

from src.agents.base_agent import BaseRAGAgent
from src.utils.langfuse_setup import observe


class HRAgent(BaseRAGAgent):
    """Specialized agent for HR domain queries"""
    
    def __init__(self, vector_store: "Chroma"):
        """
        Initialize HR agent
        
        Args:
            vector_store: Vector store containing HR documents
        """
        super().__init__(
            vector_store=vector_store,
            agent_name="HR Agent",
            domain="HR"
        )
    
    @observe(name="hr_agent_query")
    def answer(self, query: str) -> Dict:
        """
        Answer HR-related query using RAG
        
        Args:
            query: User query
            
        Returns:
            Dictionary with answer, source documents, and agent info
        """
        if os.getenv("DISABLE_LLM", "0") == "1":
            # Enhanced structured fallback (offline summarization)
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 8})
            # Use new invoke API to avoid deprecation warnings
            docs = retriever.invoke(query)
            sections = {
                "Vacation / PTO": [],
                "Request & Approval": [],
                "Extended / Special Leave": [],
                "Contacts": []
            }
            keywords = {
                "Vacation / PTO": ["vacation", "pto", "paid time", "holiday", "time off"],
                "Request & Approval": ["request", "approval", "approve", "manager", "submit"],
                "Extended / Special Leave": ["extended", "medical", "family", "bereavement", "parental"],
                "Contacts": ["hr", "email", "contact"]
            }
            def clean_line(line: str) -> str:
                line = line.strip().replace("\t", " ")
                while "  " in line:
                    line = line.replace("  ", " ")
                line = line.lstrip("-•*0123456789. ")
                return line[:240]
            # Track normalized uniqueness per section to avoid duplicate lines
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
            answer = "HR Policy Summary (offline extraction):\n\n" + "\n\n".join(
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
                docs = retriever.get_relevant_documents(query)
                bullets = []
                for d in docs:
                    snippet = d.page_content.strip().splitlines()[0][:200]
                    bullets.append(f"- {snippet}")
                answer = "Based on HR documentation:\n" + "\n".join(bullets)
                sources = docs
        
        return {
            "answer": answer,
            "source_documents": sources,
            "agent": self.agent_name,
            "domain": self.domain
        }

