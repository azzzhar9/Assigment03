"""Quick test to verify OpenRouter LLM connectivity"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

print("Testing OpenRouter LLM connection...")
print(f"API Key: {os.getenv('OPENROUTER_API_KEY')[:20]}...")
print(f"Base URL: {os.getenv('OPENROUTER_BASE_URL')}")
print(f"Model: {os.getenv('OPENAI_MODEL', 'openrouter/auto')}")

try:
    llm = ChatOpenAI(
        model=os.getenv('OPENAI_MODEL', 'openrouter/auto'),
        temperature=0.0,
        openai_api_key=os.getenv('OPENROUTER_API_KEY'),
        openai_api_base=os.getenv('OPENROUTER_BASE_URL'),
        max_retries=1,
        timeout=10,
        request_timeout=10
    )
    
    print("\nSending test query...")
    response = llm.invoke("Say 'Hello, working!' if you can read this.")
    print(f"\nSuccess! Response: {response.content}")
    
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
