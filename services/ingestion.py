# filepath: backend/services/ingestion.py
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from vectorstore.faiss_store import get_faiss_index, save_faiss_index

def ingest_pdfs(folder_path: str, collection_name: str):
    """
    Reads PDFs from a folder and stores chunks in FAISS.
    """
    print(f"Starting ingestion from {folder_path} into collection {collection_name}")
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return 0

    pdf_files = glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
    if not pdf_files:
        print(f"No PDF files found in {folder_path}.")
        return 0
        
    documents = []
    
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file}...")
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()
        documents.extend(pages)
        
    print(f"Loaded {len(documents)} total pages.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    if not chunks:
        return 0
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Pre-process metadata to ensure compatibility
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk.metadata = {
            "source": os.path.basename(source),
            "page": page,
            "collection": collection_name,
            "chunk_index": i
        }

    print(f"Building FAISS index with {len(chunks)} chunks...")
    
    existing_index = get_faiss_index(collection_name)
    
    if existing_index is not None:
        print("Adding to existing FAISS index...")
        existing_index.add_documents(chunks)
        save_faiss_index(existing_index, collection_name)
    else:
        print("Creating new FAISS index...")
        new_index = FAISS.from_documents(chunks, embeddings)
        save_faiss_index(new_index, collection_name)
        
    print(f"Successfully ingested {len(chunks)} chunks into {collection_name}.")
    return len(chunks)
