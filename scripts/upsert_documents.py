#!/usr/bin/env python3
"""
Upsert documents (text/pdf/docx) into Pinecone index.
Place in project root and run: python scripts/upsert_documents.py
"""

import os
import time
import uuid
import hashlib
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# third-party libs
try:
    import pinecone
    from sentence_transformers import SentenceTransformer
    import PyPDF2
    import docx
except Exception as e:
    raise SystemExit(f"Missing dependency: {e}. Install required packages: pip install pinecone-client sentence-transformers PyPDF2 python-docx tqdm")

# ---------- CONFIG (reads from env, fallback defaults) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDICAL_LITERATURE_PATH = os.getenv("MEDICAL_LITERATURE_PATH", str(PROJECT_ROOT / "data" / "medical_literature"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV") or ""
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "medical-literature-index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "dmis-lab/biobert-base-cased-v1.2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
BATCH_SIZE = 100  # upsert batch size
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")  # optional

# ---------- utility functions ----------
def extract_text_from_pdf(path: Path) -> str:
    text_chunks = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in range(len(reader.pages)):
            try:
                text = reader.pages[p].extract_text() or ""
            except Exception:
                text = ""
            if text:
                text_chunks.append(text)
    return "\n".join(text_chunks)

def extract_text_from_docx(path: Path) -> str:
    doc = docx.Document(path)
    paras = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paras)

def read_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple sliding window chunker on whitespace tokens."""
    tokens = text.split()
    if not tokens:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks

def deterministic_id(text: str, prefix: str = "") -> str:
    """Create deterministic id from text (sha1), optionally prefixed."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{prefix}{h}"

# ---------- main ----------
def main():
    if not PINECONE_API_KEY:
        raise SystemExit("Set PINECONE_API_KEY environment variable before running.")

    # init pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if PINECONE_INDEX not in pinecone.list_indexes():
        raise SystemExit(f"Index '{PINECONE_INDEX}' not found in environment '{PINECONE_ENV}'. Create it first or change PINECONE_INDEX_NAME/PINECONE_ENVIRONMENT.")

    # describe index & dimension
    desc = pinecone.describe_index(PINECONE_INDEX)
    index_dim = desc.dimension
    print(f"Pinecone index '{PINECONE_INDEX}' dimension: {index_dim}")

    # load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    # test embedding shape
    sample_vec = embed_model.encode(["test"])
    embed_dim = len(sample_vec[0])
    print(f"Embedding model dimension: {embed_dim}")

    if embed_dim != index_dim:
        raise SystemExit(f"Dimension mismatch: embedding dim={embed_dim} != index dim={index_dim}. Recreate index or choose a matching model.")

    index = pinecone.Index(PINECONE_INDEX)

    # gather files to index
    doc_root = Path(MEDICAL_LITERATURE_PATH)
    if not doc_root.exists():
        raise SystemExit(f"Medical literature path not found: {doc_root}")

    files = list(doc_root.rglob("*.*"))
    # filter by extension
    supported_exts = {".txt", ".md", ".pdf", ".docx"}
    files = [f for f in files if f.suffix.lower() in supported_exts]
    if not files:
        print("No supported files to index. Put text/pdf/docx files into MEDICAL_LITERATURE_PATH.")
        return

    # prepare upsert batches
    upsert_batch = []
    total_chunks = 0

    for file_path in tqdm(files, desc="Files"):
        text = ""
        try:
            if file_path.suffix.lower() == ".pdf":
                text = extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() == ".docx":
                text = extract_text_from_docx(file_path)
            else:
                text = read_text_file(file_path)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue

        if not text.strip():
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        total_chunks += len(chunks)

        for i, chunk in enumerate(chunks):
            # metadata: keep minimal but useful fields
            metadata = {
                "source": str(file_path.name),
                "path": str(file_path.resolve()),
                "chunk_index": i,
                "chunk_size": len(chunk.split()),
            }
            vec = embed_model.encode(chunk)
            vec = vec.tolist()  # pinecone expects list[float]

            # deterministic id so re-running won't duplicate same text
            doc_id = deterministic_id(chunk, prefix=f"{file_path.stem}__")
            upsert_batch.append((doc_id, vec, metadata))

            # flush batch
            if len(upsert_batch) >= BATCH_SIZE:
                _upsert_to_pinecone(index, upsert_batch, namespace=NAMESPACE)
                upsert_batch = []

    # final flush
    if upsert_batch:
        _upsert_to_pinecone(index, upsert_batch, namespace=NAMESPACE)

    print(f"Done. Total chunks processed: {total_chunks}")

    # show index stats
    stats = index.describe_index_stats(namespace=NAMESPACE) if NAMESPACE else index.describe_index_stats()
    print("Index stats:", stats)


def _upsert_to_pinecone(index, batch, namespace=""):
    """
    batch: list of tuples (id, vector, metadata)
    """
    to_upsert = []
    for id_, vec, metadata in batch:
        to_upsert.append({"id": id_, "values": vec, "metadata": metadata})
    # pinecone upsert
    print(f"Upserting {len(to_upsert)} vectors...")
    index.upsert(vectors=to_upsert, namespace=namespace)
    time.sleep(0.1)  # slight pause to be polite

if __name__ == "__main__":
    main()
