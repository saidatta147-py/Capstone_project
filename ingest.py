"""Local document chunking and ChromaDB indexing."""
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"; DB = ROOT / "chroma_db"; COLLECTION = "zepto_policies"

def collection():
    client = chromadb.PersistentClient(path=str(DB))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space":"cosine"})

def build_index(force: bool = False):
    col = collection()
    if col.count() and not force: return col
    if force and col.count(): col.delete(ids=col.get()["ids"])
    model = SentenceTransformer("all-MiniLM-L6-v2")
    files = sorted(DOCS.glob("doc_*.txt")); texts = [p.read_text(encoding="utf-8").strip() for p in files]
    ids = [p.stem for p in files]
    col.add(ids=ids, documents=texts, embeddings=model.encode(texts).tolist(), metadatas=[{"source":i} for i in ids])
    return col

def retrieve(query: str, limit: int = 3):
    col = build_index(); model = SentenceTransformer("all-MiniLM-L6-v2")
    result = col.query(query_embeddings=[model.encode(query).tolist()], n_results=limit, include=["documents", "metadatas", "distances"])
    return [{"id": ident, "text": text, "distance": distance} for ident,text,distance in zip(result["ids"][0],result["documents"][0],result["distances"][0])]

if __name__ == "__main__":
    print(f"Indexed {build_index(force=True).count()} documents in {DB}")
