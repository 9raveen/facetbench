# src/vectordb/chroma_client.py
import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

CHROMA_PATH = "./data/chroma_db"
FACETS_PATH = "./data/facets/facets.json"

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="facets",
    embedding_function=embed_fn
)

def index_facets():
    with open(FACETS_PATH) as f:
        facets = json.load(f)

    existing = collection.get()["ids"]
    new_facets = [f for f in facets if f["facet_id"] not in existing]

    if not new_facets:
        print("All facets already indexed.")
        return

    collection.add(
        ids=[f["facet_id"] for f in new_facets],
        documents=[
            f"{f['name']}: {f['description']} | Category: {f['category']} | Group: {f['group']}"
            for f in new_facets
        ],
        metadatas=[
            {
                "facet_id": f["facet_id"],
                "name": f["name"],
                "category": f["category"],
                "group": f["group"],
                "weight": f["weight"]
            }
            for f in new_facets
        ]
    )
    print(f"Indexed {len(new_facets)} facets into ChromaDB.")

def retrieve_facets(turn_text: str, n: int = 40) -> list:
    results = collection.query(
        query_texts=[turn_text],
        n_results=min(n, collection.count())
    )
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]

    with open(FACETS_PATH) as f:
        all_facets = {fac["facet_id"]: fac for fac in json.load(f)}

    return [all_facets[fid] for fid in ids if fid in all_facets]

if __name__ == "__main__":
    index_facets()
    print(f"Total facets in DB: {collection.count()}")