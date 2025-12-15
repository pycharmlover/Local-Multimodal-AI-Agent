# /modules/search.py
#拆目录 + PersistentClient
import os
import chromadb

BASE_INDEX_DIR = "data/index"

PAPER_INDEX_DIR = os.path.join(BASE_INDEX_DIR, "papers")
IMAGE_INDEX_DIR = os.path.join(BASE_INDEX_DIR, "images")
PARA_INDEX_DIR  = os.path.join(BASE_INDEX_DIR, "paragraphs")

os.makedirs(PAPER_INDEX_DIR, exist_ok=True)
os.makedirs(IMAGE_INDEX_DIR, exist_ok=True)
os.makedirs(PARA_INDEX_DIR, exist_ok=True)

paper_client = chromadb.PersistentClient(path=PAPER_INDEX_DIR)
image_client = chromadb.PersistentClient(path=IMAGE_INDEX_DIR)
para_client  = chromadb.PersistentClient(path=PARA_INDEX_DIR)

paper_collection = paper_client.get_or_create_collection("papers")
image_collection = image_client.get_or_create_collection("images")
para_collection  = para_client.get_or_create_collection("paragraphs")

def add_paper_embedding(pid, embedding, metadata):
    emb = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    paper_collection.add(
        ids=[pid],
        embeddings=[emb],
        metadatas=[metadata]
    )


def search_paper(query_emb, top_k=5):
    q = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

    results = paper_collection.query(
        query_embeddings=[q],
        n_results=top_k,
        include=["metadatas"]
    )

    if not results or not results.get("metadatas"):
        return {"metadatas": [[]]}

    return results

def add_image_embedding(iid, embedding, metadata):
    emb = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    image_collection.add(
        ids=[iid],
        embeddings=[emb],
        metadatas=[metadata]
    )


def search_image(query_emb, top_k=10):
    q = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

    results = image_collection.query(
        query_embeddings=[q],
        n_results=top_k,
        include=["distances", "metadatas"]
    )

    hits = []
    if results.get("distances") and results.get("metadatas"):
        for dist, meta in zip(results["distances"][0], results["metadatas"][0]):
            hits.append({
                "path": meta.get("path", ""),
                "distance": dist
            })

    return hits

def add_paragraph(pid, para_id, embedding, metadata):
    emb = embedding.tolist() if hasattr(embedding, "tolist") else embedding

    metadata = dict(metadata)
    metadata["para_id"] = para_id  # ⭐关键

    para_collection.add(
        ids=[f"{pid}_{para_id}"],
        embeddings=[emb],
        metadatas=[metadata]
    )

def search_paragraph(query_emb, top_k=5):
    q = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

    results = para_collection.query(
        query_embeddings=[q],
        n_results=top_k,
        include=["metadatas"]
    )

    if not results or not results.get("metadatas"):
        return {"metadatas": [[]], "processed": []}

    processed = []
    for meta in results["metadatas"][0]:
        processed.append({
            "paper": meta.get("paper", "unknown"),
            "topic": meta.get("topic", "unknown"),
            "para_id": meta.get("para_id", -1),
            "text": meta.get("text", "")
        })

    results["processed"] = processed
    return results

