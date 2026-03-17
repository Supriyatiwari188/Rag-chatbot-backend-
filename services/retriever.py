# filepath: backend/services/retriever.py
from langchain_openai import OpenAIEmbeddings
from vectorstore.faiss_store import get_faiss_index

def retrieve_context(query: str, collection_name: str, top_k: int = 4):
    """
    Retrieve relevant chunks for the query from the specified FAISS collection.
    Returns: (chunks: list[str], sources: list[dict], confidence: float)
    """
    index = get_faiss_index(collection_name)
    if index is None:
        # Collection does not exist
        return [], [], 0.0

    # similarity_search_with_score in FAISS returns L2 distance. Lower is better.
    results = index.similarity_search_with_score(query, k=top_k)
    
    if not results:
        return [], [], 0.0
        
    chunks = []
    metadatas = []
    distances = []
    
    for doc, score in results:
        chunks.append(doc.page_content)
        metadatas.append(doc.metadata)
        distances.append(score)
    
    # Calculate confidence: 1 - min(distances) clamped between 0 and 1
    min_dist = min(distances) if distances else 1.0
    confidence = max(0.0, min(1.0, 1.0 - min_dist))
    
    # Fallback to zero if confidence is below threshold
    if confidence < 0.4:
         return [], [], confidence

    sources = []
    for meta in metadatas:
        sources.append({
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", 0)
        })
        
    # Deduplicate sources based on filename and page
    unique_sources = []
    seen = set()
    for s in sources:
        identifier = f"{s['source']}_{s['page']}"
        if identifier not in seen:
            seen.add(identifier)
            unique_sources.append(s)
            
    return chunks, unique_sources, confidence
