"""
Langfuse tracing configuration and setup
"""

from typing import Optional
try:
    from langfuse import Langfuse, observe
except Exception:  # pragma: no cover
    Langfuse = None
    def observe(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

from src.config import Config


class LangfuseManager:
    """Manages Langfuse tracing and observability"""
    
    def __init__(self):
        """Initialize Langfuse client"""
        self.client = None
        if Langfuse and Config.LANGFUSE_PUBLIC_KEY and Config.LANGFUSE_SECRET_KEY and not __import__('os').getenv('DISABLE_LANGFUSE'):
            try:
                self.client = Langfuse(
                    public_key=Config.LANGFUSE_PUBLIC_KEY,
                    secret_key=Config.LANGFUSE_SECRET_KEY,
                    host=Config.LANGFUSE_HOST
                )
            except Exception as e:  # network or init failure
                print(f"Warning: Langfuse disabled (init failed: {e})")
        # LangChain callback handler not available in current setup
        self.callback_handler = None
    
    def get_callback_handler(self):
        """Get LangChain callback handler for tracing"""
        return self.callback_handler
    
    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None
    ):
        """
        Create a score for a trace
        
        Args:
            trace_id: ID of the trace to score
            name: Name of the score metric
            value: Score value (1-10)
            comment: Optional comment
        """
        if not self.client:
            return
        try:
            self.client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment
            )
        except Exception as e:
            print(f"Warning: Langfuse score failed: {e}")
    
    def flush(self):
        """Flush pending events to Langfuse"""
        if not self.client:
            return
        try:
            self.client.flush()
        except Exception as e:
            print(f"Warning: Langfuse flush failed: {e}")


# Global Langfuse manager instance
_langfuse_manager: Optional[LangfuseManager] = None


def get_langfuse_manager() -> LangfuseManager:
    """Get or create global Langfuse manager"""
    global _langfuse_manager
    if _langfuse_manager is None:
        _langfuse_manager = LangfuseManager()
    return _langfuse_manager


def get_langfuse_callback():
    """Get Langfuse callback handler for LangChain"""
    return get_langfuse_manager().get_callback_handler()

