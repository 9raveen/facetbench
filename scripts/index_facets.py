# scripts/index_facets.py
import sys
sys.path.insert(0, ".")

from src.vectordb.chroma_client import index_facets, collection

if __name__ == "__main__":
    print("Indexing facets into ChromaDB...")
    index_facets()
    print(f"Total facets in ChromaDB: {collection.count()}")