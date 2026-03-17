# filepath: backend/scripts/ingest.py
import argparse
import os
import sys

# Add the 'backend' folder to sys.path so we can import from services and vectorstore
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ingestion import ingest_pdfs

def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB.")
    parser.add_argument(
        "--source", 
        type=str, 
        choices=["nec", "wattmonk", "all"], 
        required=True,
        help="Source collection to ingest: 'nec', 'wattmonk', or 'all'."
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nec_path = os.path.join(base_dir, "data", "nec_docs")
    wattmonk_path = os.path.join(base_dir, "data", "wattmonk_docs")

    if args.source in ["nec", "all"]:
        chunks = ingest_pdfs(nec_path, "nec")
        print(f"Total chunks ingested for NEC: {chunks}")

    if args.source in ["wattmonk", "all"]:
        chunks = ingest_pdfs(wattmonk_path, "wattmonk")
        print(f"Total chunks ingested for Wattmonk: {chunks}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
