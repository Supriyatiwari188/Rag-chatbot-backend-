# filepath: backend/vectorstore/faiss_store.py
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Dictionary to hold loaded FAISS indices
_faiss_indices = {}

def get_faiss_index(collection_name: str):
    """
    Get a loaded FAISS index by name. If not loaded, attempt to load it from disk.
    Creates an empty one if it doesn't exist.
    """
    global _faiss_indices
    if collection_name in _faiss_indices:
        return _faiss_indices[collection_name]
        
    db_path = os.getenv("FAISS_DB_PATH", "./faiss_db")
    collection_path = os.path.join(db_path, collection_name)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(collection_path) and os.path.exists(os.path.join(collection_path, "index.faiss")):
        # Load existing index
        index = FAISS.load_local(collection_path, embeddings, allow_dangerous_deserialization=True)
        _faiss_indices[collection_name] = index
        return index
    
    # Return None if it doesn't exist
    return None

def save_faiss_index(index: FAISS, collection_name: str):
    """
    Save the FAISS index to the local directory.
    """
    global _faiss_indices
    db_path = os.getenv("FAISS_DB_PATH", "./faiss_db")
    collection_path = os.path.join(db_path, collection_name)
    os.makedirs(collection_path, exist_ok=True)
    
    index.save_local(collection_path)
    _faiss_indices[collection_name] = index

def collection_exists(name: str) -> bool:
    """Helper to check if a collection exists on disk."""
    db_path = os.getenv("FAISS_DB_PATH", "./faiss_db")
    collection_path = os.path.join(db_path, name)
    return os.path.exists(collection_path) and os.path.exists(os.path.join(collection_path, "index.faiss"))

