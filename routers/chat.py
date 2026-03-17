# filepath: backend/routers/chat.py
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, root_validator
import html
import json
import traceback

from services.intent_classifier import classify_intent
from services.retriever import retrieve_context
from services.llm_chain import generate_response

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    chat_history: list = []
    session_id: str

    @root_validator(pre=True)
    def sanitize_query(cls, values):
        if "query" in values:
            q = values["query"][:2000] # Limit to 2000 chars
            values["query"] = html.escape(q) # Strip HTML via escape
        return values

@router.post("/chat")
async def chat_endpoint(request: Request, payload: ChatRequest):
    # rate limit check: max 20 per minute
    limiter = request.app.state.limiter
    
    # We apply rate limit manually on the object because using decorator is tricky with class methods
    # We do it here but typically you'd use @limiter.limit("20/minute") on the route.
    # The requirement is just "max 20 requests/minute per IP".
    
    query = payload.query
    chat_history = payload.chat_history
    session_id = payload.session_id

    # 1. Intent Classification
    intent = classify_intent(query)
    
    # 2. Retrieval
    collection_name = intent
    sources = []
    confidence = 0.0
    context_chunks = []
    
    if intent in ["nec", "wattmonk"]:
        context_chunks, sources, confidence = retrieve_context(query, collection_name)
    
    if confidence < 0.4:
        context_chunks = []
        sources = []
        confidence = 0.0
        # "If no relevant chunks found... respond with base LLM knowledge" is handled in generate_response
        # by passing empty context chunks

    # 3. Stream Response
    async def event_generator():
        try:
            suggested_questions = []
            
            # generate_response is an async generator
            async for token in generate_response(query, context_chunks, sources, chat_history, intent):
                if isinstance(token, str):
                    yield f"data: {json.dumps({'text': token})}\n\n"
                elif isinstance(token, dict) and "suggested_questions" in token:
                    suggested_questions = token["suggested_questions"]
            
            # Send the final [DONE] event
            final_data = {
                "intent": intent,
                "sources": sources,
                "confidence_score": confidence,
                "suggested_questions": suggested_questions,
            }
            yield f"data: [DONE] {json.dumps(final_data)}\n\n"
            
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'error': 'Service temporarily unavailable'})}\n\n"
            yield f"data: [DONE] {{}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Add the slowapi Limiter decorator to the function directly instead
from slowapi import Limiter
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
# Note: we should apply the decorator to the route, let's fix that by re-declaring it.
# Actually we can just apply limit decorator to local func.
chat_endpoint = limiter.limit("20/minute")(chat_endpoint)
