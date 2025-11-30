"""
Evaluator agent for quality assessment of RAG responses
"""

from typing import Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from src.config import Config
from src.utils.langfuse_setup import get_langfuse_manager, get_langfuse_callback, observe


class EvaluatorAgent:
    """Agent that evaluates the quality of RAG responses"""
    
    def __init__(self):
        """Initialize evaluator agent"""
        callback = get_langfuse_callback()
        callbacks = [callback] if callback else []
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=0.0,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
            callbacks=callbacks
        )
        
        self.langfuse_manager = get_langfuse_manager()
        
        # Create evaluation prompt
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality evaluator for customer support responses.
Evaluate the response based on the original query and provide scores (1-10) for:
1. Relevance: How well does the answer address the query?
2. Completeness: Does the answer fully address all aspects of the query?
3. Accuracy: Is the answer factually correct and based on the provided context?

Respond in JSON format with scores and a brief comment for each dimension:
{{
    "relevance": <score 1-10>,
    "relevance_comment": "<brief explanation>",
    "completeness": <score 1-10>,
    "completeness_comment": "<brief explanation>",
    "accuracy": <score 1-10>,
    "accuracy_comment": "<brief explanation>",
    "overall_score": <average of three scores>
}}"""),
            ("human", """Original Query: {query}

Response: {response}

Evaluate the response quality:""")
        ])
    
    @observe(name="evaluate_response")
    def evaluate(
        self,
        query: str,
        response: str,
        trace_id: Optional[str] = None
    ) -> Dict:
        """
        Evaluate the quality of a response
        
        Args:
            query: Original user query
            response: Agent response to evaluate
            trace_id: Optional Langfuse trace ID for scoring (auto-retrieved if not provided)
            
        Returns:
            Dictionary with evaluation scores and comments
        """
        import os
        # Skip evaluation entirely if LLM or evaluation disabled
        if os.getenv("DISABLE_LLM") == "1" or os.getenv("DISABLE_EVALUATION") == "1":
            return None
        # Automatically get trace_id from Langfuse context if not provided
        if trace_id is None:
            try:
                from langfuse.decorators import langfuse_context
                trace_id = langfuse_context.get_current_trace_id()
            except Exception:
                # If context not available, try to get from observe decorator
                try:
                    trace_id = getattr(self, '_current_trace_id', None)
                except Exception:
                    trace_id = None
        
        # Get evaluation from LLM
        messages = self.evaluation_prompt.format_messages(
            query=query,
            response=response
        )
        
        llm_response = self.llm.invoke(messages)
        
        # Parse JSON response (simple extraction, could be improved)
        import json
        import re
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response.content, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
            else:
                # Fallback: create default evaluation
                evaluation = {
                    "relevance": 5,
                    "relevance_comment": "Could not parse evaluation",
                    "completeness": 5,
                    "completeness_comment": "Could not parse evaluation",
                    "accuracy": 5,
                    "accuracy_comment": "Could not parse evaluation",
                    "overall_score": 5
                }
        except json.JSONDecodeError:
            # Fallback evaluation
            evaluation = {
                "relevance": 5,
                "relevance_comment": "JSON parsing error",
                "completeness": 5,
                "completeness_comment": "JSON parsing error",
                "accuracy": 5,
                "accuracy_comment": "JSON parsing error",
                "overall_score": 5
            }
        
        # Automatically score in Langfuse (always attempt, even without trace_id)
        try:
            # Try to get trace_id from current observation if not already set
            if trace_id is None:
                try:
                    from langfuse.decorators import langfuse_context
                    trace_id = langfuse_context.get_current_trace_id()
                except:
                    pass
            
            if trace_id:
                try:
                    overall_score = evaluation.get("overall_score", 5)
                    self.langfuse_manager.score(
                        trace_id=trace_id,
                        name="response_quality",
                        value=overall_score,
                        comment=f"Relevance: {evaluation.get('relevance', 'N/A')}, "
                               f"Completeness: {evaluation.get('completeness', 'N/A')}, "
                               f"Accuracy: {evaluation.get('accuracy', 'N/A')}"
                    )
                    
                    # Also score individual dimensions
                    self.langfuse_manager.score(
                        trace_id=trace_id,
                        name="relevance",
                        value=evaluation.get("relevance", 5)
                    )
                    self.langfuse_manager.score(
                        trace_id=trace_id,
                        name="completeness",
                        value=evaluation.get("completeness", 5)
                    )
                    self.langfuse_manager.score(
                        trace_id=trace_id,
                        name="accuracy",
                        value=evaluation.get("accuracy", 5)
                    )
                except Exception as e:
                    print(f"Warning: Could not score in Langfuse (trace_id may not be available): {e}")
        except Exception as e:
            # Continue even if scoring fails - evaluation still returned
            pass
        
        return evaluation

