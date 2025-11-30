"""
Orchestrator agent for intent classification and routing
"""

from typing import Dict, Optional
import os
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from src.config import Config
from src.agents.hr_agent import HRAgent
from src.agents.tech_agent import TechAgent
from src.agents.finance_agent import FinanceAgent
from src.utils.langfuse_setup import observe, get_langfuse_callback


class OrchestratorAgent:
    """Orchestrator that classifies intent and routes to specialized agents"""
    
    INTENT_CATEGORIES = ["HR", "Tech", "Finance", "Unknown"]
    
    def __init__(
        self,
        hr_agent: HRAgent,
        tech_agent: TechAgent,
        finance_agent: FinanceAgent
    ):
        """
        Initialize orchestrator with specialized agents
        
        Args:
            hr_agent: HR specialized agent
            tech_agent: Tech specialized agent
            finance_agent: Finance specialized agent
        """
        self.hr_agent = hr_agent
        self.tech_agent = tech_agent
        self.finance_agent = finance_agent
        
        # Initialize LLM for intent classification (supports OpenAI or OpenRouter)
        callback = get_langfuse_callback()
        callbacks = [callback] if callback else []
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=0.0,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            callbacks=callbacks,
            max_retries=2,
            timeout=15,
            request_timeout=15
        )
        
        # Create intent classification prompt
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classification system for a customer support routing system.
Your task is to classify user queries into one of these categories:
- HR: Questions about benefits, leave policies, employee policies, hiring, onboarding, performance reviews, workplace issues
- Tech: Questions about software, hardware, IT support, system access, technical troubleshooting, software licenses
- Finance: Questions about expenses, invoices, budgets, payments, financial policies, reimbursement

Respond with ONLY the category name (HR, Tech, or Finance). If the query doesn't clearly fit any category, respond with "Unknown"."""),
            ("human", "{query}")
        ])
    
    @observe(name="intent_classification")
    def classify_intent(self, query: str) -> str:
        """
        Classify the intent of a user query
        
        Args:
            query: User query
            
        Returns:
            Intent category (HR, Tech, Finance, or Unknown)
        """
        if os.getenv("DISABLE_LLM", "0") == "1":
            q = query.lower()
            def contains_any(text: str, kws) -> bool:
                for k in kws:
                    # Use word boundaries for short tokens; substring match for longer domain terms
                    if len(k) <= 3:
                        if re.search(rf"\\b{re.escape(k)}\\b", text):
                            return True
                    else:
                        if k in text:
                            return True
                return False

            if contains_any(q, ["leave", "vacation", "benefits", "onboarding", "performance", "hr"]):
                intent = "HR"
            elif contains_any(q, ["password", "reset", "system", "access", "network", "software"]):
                intent = "Tech"
            elif contains_any(q, ["expense", "invoice", "budget", "payment", "reimbursement", "finance"]):
                intent = "Finance"
            else:
                intent = "Unknown"
        else:
            try:
                messages = self.intent_prompt.format_messages(query=query)
                response = self.llm.invoke(messages)
                intent = response.content.strip()
            except Exception as e:
                print(f"Warning: Intent LLM call failed ({type(e).__name__}: {e}), using keyword fallback")
                q = query.lower()
                if any(k in q for k in ["leave", "vacation", "benefits", "onboarding", "performance", "hr"]):
                    intent = "HR"
                elif any(k in q for k in ["password", "reset", "system", "access", "network", "software", "it"]):
                    intent = "Tech"
                elif any(k in q for k in ["expense", "invoice", "budget", "payment", "reimbursement", "finance"]):
                    intent = "Finance"
                else:
                    intent = "Unknown"
        
        if intent not in self.INTENT_CATEGORIES:
            intent = "Unknown"
        
        return intent
    
    @observe(name="route_query")
    def route_and_answer(self, query: str) -> Dict:
        """
        Classify intent and route query to appropriate agent
        
        Args:
            query: User query
            
        Returns:
            Dictionary with answer, routing info, and metadata
        """
        # Classify intent
        intent = self.classify_intent(query)
        
        # Route to appropriate agent
        if intent == "HR":
            agent_response = self.hr_agent.answer(query)
        elif intent == "Tech":
            agent_response = self.tech_agent.answer(query)
        elif intent == "Finance":
            agent_response = self.finance_agent.answer(query)
        else:
            # Unknown intent - provide generic response
            agent_response = {
                "answer": "I'm not sure which department can best help with your query. Could you provide more details? You can contact HR for employee-related questions, IT Support for technical issues, or Finance for payment and expense questions.",
                "source_documents": [],
                "agent": "Orchestrator",
                "domain": "Unknown"
            }
        
        return {
            **agent_response,
            "intent": intent,
            "original_query": query
        }

