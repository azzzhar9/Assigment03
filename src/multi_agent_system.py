"""
Main entry point for the multi-agent system
"""

import os
import re
from typing import Dict, Optional

from src.config import Config
from src.utils.document_loader import DocumentLoader
from src.utils.vector_store import VectorStoreManager
from src.agents.orchestrator import OrchestratorAgent
from src.agents.hr_agent import HRAgent
from src.agents.tech_agent import TechAgent
from src.agents.finance_agent import FinanceAgent
from src.evaluator.evaluator_agent import EvaluatorAgent
from src.utils.langfuse_setup import get_langfuse_manager


class MultiAgentSystem:
    """Main multi-agent system orchestrator"""
    
    def __init__(self, rebuild_vector_stores: bool = False):
        """
        Initialize the multi-agent system
        
        Args:
            rebuild_vector_stores: Whether to rebuild vector stores from documents
        """
        # Validate configuration
        Config.validate()
        
        # Initialize components
        self.document_loader = DocumentLoader(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.vector_store_manager = VectorStoreManager()
        
        # Load or create vector stores
        self._setup_vector_stores(rebuild=rebuild_vector_stores)
        
        # Initialize specialized agents
        hr_vector_store = self.vector_store_manager.get_vector_store("hr_docs")
        tech_vector_store = self.vector_store_manager.get_vector_store("tech_docs")
        finance_vector_store = self.vector_store_manager.get_vector_store("finance_docs")
        
        if not hr_vector_store or not tech_vector_store or not finance_vector_store:
            raise ValueError("Vector stores not properly initialized")
        
        self.hr_agent = HRAgent(hr_vector_store)
        self.tech_agent = TechAgent(tech_vector_store)
        self.finance_agent = FinanceAgent(finance_vector_store)
        
        # Initialize orchestrator
        self.orchestrator = OrchestratorAgent(
            hr_agent=self.hr_agent,
            tech_agent=self.tech_agent,
            finance_agent=self.finance_agent
        )
        
        # Initialize evaluator (automatic evaluation enabled)
        self.evaluator = EvaluatorAgent()
    
    def _setup_vector_stores(self, rebuild: bool = False):
        """Setup vector stores for each domain"""
        collections = {
            "hr_docs": Config.HR_DOCS_DIR,
            "tech_docs": Config.TECH_DOCS_DIR,
            "finance_docs": Config.FINANCE_DOCS_DIR
        }
        
        for collection_name, docs_dir in collections.items():
            # Try to load existing vector store
            if not rebuild:
                vector_store = self.vector_store_manager.load_vector_store(collection_name)
                if vector_store:
                    print(f"Loaded existing vector store: {collection_name}")
                    continue
            
            # Create new vector store from documents
            if not os.path.exists(docs_dir):
                print(f"Warning: Directory {docs_dir} does not exist. Creating empty vector store.")
                # Create empty vector store
                from langchain.schema import Document
                empty_docs = [Document(page_content="No documents available", metadata={})]
                vector_store = self.vector_store_manager.create_vector_store(
                    documents=empty_docs,
                    collection_name=collection_name
                )
            else:
                print(f"Loading documents from {docs_dir}...")
                documents = self.document_loader.load_and_chunk(docs_dir)
                
                if len(documents) < 50:
                    print(f"Warning: Only {len(documents)} chunks found for {collection_name}. Minimum 50 recommended.")
                
                vector_store = self.vector_store_manager.create_vector_store(
                    documents=documents,
                    collection_name=collection_name
                )
                print(f"Created vector store: {collection_name} with {len(documents)} chunks")
    
    def process_query(
        self,
        query: str,
        evaluate: bool = True
    ) -> Dict:
        """
        Process a user query through the multi-agent system
        
        Args:
            query: User query
            evaluate: Whether to evaluate the response quality (default: True - automatic evaluation)
            
        Returns:
            Dictionary with answer, routing info, and evaluation scores
        """
        # Check for multi-domain / combined query (simple keyword heuristic)
        if self._is_multi_domain_query(query):
            result = self._process_multi_domain_query(query)
        else:
            # Route query through orchestrator (single intent)
            result = self.orchestrator.route_and_answer(query)
        
        # Automatically evaluate response quality (bonus feature - always enabled)
        if evaluate:
            try:
                # Evaluator will automatically get trace_id from Langfuse context
                evaluation = self.evaluator.evaluate(
                    query=query,
                    response=result["answer"]
                )
                result["evaluation"] = evaluation
            except Exception as e:
                # If evaluation fails, continue without it but log the error
                print(f"Warning: Evaluation failed: {e}")
                result["evaluation"] = None
        
        # Flush Langfuse events
        get_langfuse_manager().flush()
        
        return result

    def _is_multi_domain_query(self, query: str) -> bool:
        """Determine if the query appears to span multiple domains."""
        q = query.lower()
        hr_kw = ["leave", "vacation", "benefits", "onboarding", "performance", "hr"]
        tech_kw = ["password", "reset", "system", "access", "network", "software"]
        fin_kw = ["expense", "invoice", "budget", "payment", "reimbursement", "finance"]

        def contains_any(text: str, kws) -> bool:
            for k in kws:
                if len(k) <= 3:
                    if re.search(rf"\\b{re.escape(k)}\\b", text):
                        return True
                else:
                    if k in text:
                        return True
            return False

        hits = 0
        if contains_any(q, hr_kw):
            hits += 1
        if contains_any(q, tech_kw):
            hits += 1
        if contains_any(q, fin_kw):
            hits += 1
        # Consider multi-domain if two or more domain keyword groups hit
        return hits >= 2

    def _process_multi_domain_query(self, query: str) -> Dict:
        """Process a query that spans multiple domains by invoking all matching agents."""
        q = query.lower()
        domains = []
        responses = []
        source_docs = []

        # Invoke each agent conditionally
        if self._contains_any(q, ["leave", "vacation", "benefits", "onboarding", "performance", "hr"]):
            r = self.hr_agent.answer(query)
            domains.append("HR")
            responses.append("HR:\n" + r["answer"].strip())
            source_docs.extend(r.get("source_documents", []))
        if self._contains_any(q, ["password", "reset", "system", "access", "network", "software"]):
            r = self.tech_agent.answer(query)
            domains.append("Tech")
            responses.append("Tech:\n" + r["answer"].strip())
            source_docs.extend(r.get("source_documents", []))
        if self._contains_any(q, ["expense", "invoice", "budget", "payment", "reimbursement", "finance"]):
            r = self.finance_agent.answer(query)
            domains.append("Finance")
            responses.append("Finance:\n" + r["answer"].strip())
            source_docs.extend(r.get("source_documents", []))

        if not domains:
            # Fallback to orchestrator if heuristic failed
            return self.orchestrator.route_and_answer(query)

        combined_answer = "\n\n".join(responses)
        return {
            "answer": combined_answer,
            "source_documents": source_docs,
            "agent": "MultiAgentAggregator",
            "intent": ",".join(domains),
            "original_query": query
        }

    @staticmethod
    def _contains_any(text: str, keywords) -> bool:
        for k in keywords:
            if len(k) <= 3:
                if re.search(rf"\\b{re.escape(k)}\\b", text):
                    return True
            else:
                if k in text:
                    return True
        return False
    
    def get_system_info(self) -> Dict:
        """Get information about the system"""
        return {
            "agents": {
                "orchestrator": "OrchestratorAgent",
                "hr": "HRAgent",
                "tech": "TechAgent",
                "finance": "FinanceAgent",
                "evaluator": "EvaluatorAgent"
            },
            "vector_stores": list(self.vector_store_manager.vector_stores.keys()),
            "config": {
                "model": Config.OPENAI_MODEL,
                "embedding_model": Config.EMBEDDING_MODEL,
                "chunk_size": Config.CHUNK_SIZE,
                "top_k": Config.TOP_K_RETRIEVAL
            }
        }


def main():
    """Main entry point for command-line usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.multi_agent_system <query> [--no-evaluate]")
        print("Note: Evaluation is enabled by default. Use --no-evaluate to disable.")
        sys.exit(1)
    
    query = sys.argv[1]
    evaluate = "--no-evaluate" not in sys.argv  # Evaluation enabled by default
    
    # Initialize system
    print("Initializing multi-agent system...")
    system = MultiAgentSystem()
    
    # Process query
    print(f"\nProcessing query: {query}")
    result = system.process_query(query, evaluate=evaluate)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Intent: {result['intent']}")
    print(f"Agent: {result['agent']}")
    print(f"\nAnswer:\n{result['answer']}")
    
    if evaluate and 'evaluation' in result:
        eval_data = result['evaluation']
        print(f"\nEvaluation:")
        if eval_data:
            print(f"  Overall Score: {eval_data.get('overall_score', 'N/A')}/10")
            print(f"  Relevance: {eval_data.get('relevance', 'N/A')}/10 - {eval_data.get('relevance_comment', '')}")
            print(f"  Completeness: {eval_data.get('completeness', 'N/A')}/10 - {eval_data.get('completeness_comment', '')}")
            print(f"  Accuracy: {eval_data.get('accuracy', 'N/A')}/10 - {eval_data.get('accuracy_comment', '')}")
        else:
            print("  Evaluation unavailable (network or service error).")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

