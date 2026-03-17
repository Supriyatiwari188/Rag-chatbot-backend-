# filepath: backend/services/intent_classifier.py
import os
import functools
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Using LRU cache to save API calls for repeated queries
@functools.lru_cache(maxsize=100)
def classify_intent(query: str) -> str:
    """
    Classify the intent of the user's query into exactly one of: "nec", "wattmonk", or "general".
    Uses GPT-4o-mini as requested for fast/cheap classification.
    """
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.0,
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1000
    )
    
    system_prompt = """Classify the following user query into exactly one of these three categories:
1. "nec": If the query is about electrical codes, wiring rules, NEC standards, circuit breakers, ampacity, or general electrical engineering regulations.
2. "wattmonk": If the query is about Wattmonk company info, services, pricing, policies, team, or specific solar design services offered by Wattmonk.
3. "general": If the query does not fit into "nec" or "wattmonk" (e.g., greetings, general knowledge, non-related topics).

You must output exactly one word from the choices: "nec", "wattmonk", or "general". Do not output anything else.

User Query: {query}"""

    prompt = PromptTemplate.from_template(system_prompt)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"query": query}).strip().lower()
        if result in ["nec", "wattmonk", "general"]:
            return result
        return "general"
    except Exception as e:
        print(f"Error classifying intent: {e}")
        return "general"  # Fallback
