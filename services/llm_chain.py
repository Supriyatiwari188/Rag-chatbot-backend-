# filepath: backend/services/llm_chain.py
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

async def generate_response(query: str, context_chunks: list, sources: list, chat_history: list, intent: str):
    """
    Streams the response using GPT-4o. Yields token by token.
    At the end, yields a JSON object with suggested_questions.
    """
    
    llm = ChatOpenAI(
        model="openai/gpt-4o",
        temperature=0.2,
        streaming=True,
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=2000
    )
    
    # Determine System Prompt
    if intent == "nec":
        base_system_prompt = "You are an NEC electrical code expert. Answer carefully according to the NEC provisions."
    elif intent == "wattmonk":
        base_system_prompt = "You are a Wattmonk company assistant. Write professionally and clearly about Wattmonk's services and policies."
    else:
        base_system_prompt = "You are a helpful general assistant. Answer the user's questions clearly and accurately."

    context_text = "\n\n".join(context_chunks)
    
    if not context_chunks and intent in ["nec", "wattmonk"]:
        base_system_prompt += "\n\nNo specific documents found for this query based on the database. Respond accurately with your base LLM knowledge and note that 'No specific documents were found for this query'."
    elif context_chunks:
        base_system_prompt += f"\n\nContext to use for your answer:\n{context_text}"
        
    messages = [SystemMessage(content=base_system_prompt)]
    
    # Convert chat\_history to langchain messages (last 6 only)
    history_to_use = chat_history[-6:] if len(chat_history) > 6 else chat_history
    for msg in history_to_use:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content")))
            
    messages.append(HumanMessage(content=query))
    
    full_content = ""
    async for chunk in llm.astream(messages):
        if chunk.content:
            full_content += chunk.content
            yield chunk.content
            
    # Try generating suggested questions
    try:
        suggestion_prompt = f"Based on the conversation and the assistant's answer, provide exactly 3 short follow-up questions the user might ask next. Provide them as a JSON array of strings e.g. [\"question 1\", \"question 2\", \"question 3\"]. Do not provide any markdown, just the JSON array. \n\nAssistant's last answer: {full_content}"
        
        suggestion_llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0.5,
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1000
        )
        suggestion_result = await suggestion_llm.ainvoke([HumanMessage(content=suggestion_prompt)])
        # Parse JSON array
        suggestions = json.loads(suggestion_result.content)
        if isinstance(suggestions, list) and len(suggestions) <= 3:
            yield {"suggested_questions": suggestions}
        else:
            yield {"suggested_questions": ["Tell me more about this", "Can you give an example?", "What are the next steps?"]}
    except Exception as e:
        yield {"suggested_questions": ["Could you explain further?", "What does that imply?", "How can I proceed?"]}
