from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "code_chunks")


@dataclass(frozen=True)
class CodeChunk:
    id: str
    text: str
    metadata: dict


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def split_code_into_chunks(
    code: str,
    *,
    min_chars: int = 500,
    max_chars: int = 800,
) -> List[str]:
    """
    Split code into chunks between min_chars and max_chars.

    Strategy:
    - Prefer splitting on newline boundaries close to max_chars.
    - Ensure chunks are at least min_chars (except possibly the final remainder).
    """
    if not isinstance(code, str) or not code.strip():
        return []
    if min_chars <= 0:
        raise ValueError("min_chars must be > 0")
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if min_chars > max_chars:
        raise ValueError("min_chars must be <= max_chars")

    s = code
    n = len(s)
    out: List[str] = []
    i = 0

    while i < n:
        remaining = n - i
        if remaining <= max_chars:
            tail = s[i:].strip("\n")
            if tail:
                out.append(tail)
            break

        window = s[i : i + max_chars]
        cut = window.rfind("\n")

        # If newline cut makes chunk too small, search for a later newline
        # that still keeps us under max_chars but above min_chars.
        if cut != -1 and cut + 1 < min_chars:
            cut2 = window.find("\n", min_chars)
            if cut2 != -1:
                cut = cut2

        # If still not a good newline cut, fall back to hard split.
        if cut == -1 or cut + 1 < min_chars:
            cut = max_chars
        else:
            cut = cut + 1  # include the newline

        chunk = s[i : i + cut].strip("\n")
        if chunk:
            out.append(chunk)
        i = i + cut

    return out


def chunk_code(
    code: str,
    *,
    filename: str = "uploaded",
    chunk_chars: int = 2000,
    overlap_chars: int = 200,
) -> List[CodeChunk]:
    """
    Split code into overlapping text chunks for embedding.
    This keeps retrieval robust for long files.
    """
    if not isinstance(code, str) or not code.strip():
        return []
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be < chunk_chars")

    chunks: List[CodeChunk] = []
    start = 0
    n = len(code)

    i = 0
    while start < n:
        end = min(n, start + chunk_chars)
        text = code[start:end]
        # Avoid storing extremely tiny trailing chunks unless it's the only one.
        if len(text.strip()) < 20 and chunks:
            break

        chunk_id = f"{filename}:{i}:{_sha1(text)[:12]}"
        chunks.append(
            CodeChunk(
                id=chunk_id,
                text=text,
                metadata={
                    "filename": filename,
                    "chunk_index": i,
                    "start": start,
                    "end": end,
                },
            )
        )
        i += 1
        start = end - overlap_chars
        if start < 0:
            start = 0
        if end == n:
            break

    return chunks


def get_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> SentenceTransformer:
    """
    Create (or reuse) a SentenceTransformers model.
    """
    return SentenceTransformer(model_name)


def embed_code_chunks(
    chunks: Sequence[str],
    *,
    embedder: Optional[SentenceTransformer] = None,
    model_name: str = DEFAULT_EMBED_MODEL,
    normalize: bool = True,
) -> List[List[float]]:
    """
    Use SentenceTransformers to generate embeddings for each code chunk.

    Returns a list of embedding vectors (one per chunk).
    """
    if not chunks:
        return []
    if any((not isinstance(c, str) or not c.strip()) for c in chunks):
        raise ValueError("chunks must be a sequence of non-empty strings")

    embedder = embedder or get_embedder(model_name)
    vectors = embedder.encode(list(chunks), normalize_embeddings=normalize)
    return vectors.tolist()


def get_chroma_collection(
    *,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> "chromadb.api.models.Collection.Collection":
    """
    Get a persistent ChromaDB collection on disk.
    """
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=collection_name)


def store_code_chunks_in_chroma(
    *,
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    ids: Optional[Sequence[str]] = None,
    metadatas: Optional[Sequence[dict]] = None,
    collection=None,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """
    Store code chunks and their embeddings in a ChromaDB collection.

    Returns number of stored chunks.
    """
    if not chunks:
        return 0
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    collection = collection or get_chroma_collection(
        persist_dir=persist_dir, collection_name=collection_name
    )

    resolved_ids = list(ids) if ids is not None else [f"chunk:{i}" for i in range(len(chunks))]
    if len(resolved_ids) != len(chunks):
        raise ValueError("ids must have the same length as chunks")

    resolved_metas = list(metadatas) if metadatas is not None else [{} for _ in range(len(chunks))]
    if len(resolved_metas) != len(chunks):
        raise ValueError("metadatas must have the same length as chunks")

    # Upsert-like behavior: delete any existing ids, then add.
    # This keeps the API stable across Chroma versions.
    try:
        collection.delete(ids=resolved_ids)
    except Exception:
        pass

    collection.add(
        ids=resolved_ids,
        documents=list(chunks),
        metadatas=resolved_metas,
        embeddings=[list(v) for v in embeddings],
    )
    return len(chunks)


def index_code(
    code: str,
    *,
    filename: str = "uploaded",
    embedder: Optional[SentenceTransformer] = None,
    collection=None,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    chunk_chars: int = 2000,
    overlap_chars: int = 200,
) -> int:
    """
    Convert code to embeddings and store in ChromaDB.

    Returns number of chunks stored.
    """
    chunks = chunk_code(
        code,
        filename=filename,
        chunk_chars=chunk_chars,
        overlap_chars=overlap_chars,
    )
    if not chunks:
        return 0

    embedder = embedder or get_embedder()
    collection = collection or get_chroma_collection(
        persist_dir=persist_dir, collection_name=collection_name
    )

    texts = [c.text for c in chunks]
    ids = [c.id for c in chunks]
    metadatas = [c.metadata for c in chunks]

    vectors = embed_code_chunks(texts, embedder=embedder, normalize=True)

    return store_code_chunks_in_chroma(
        chunks=texts,
        embeddings=vectors,
        ids=ids,
        metadatas=metadatas,
        collection=collection,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )


def search_similar(
    query: str,
    *,
    top_k: int = 5,
    embedder: Optional[SentenceTransformer] = None,
    collection=None,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> List[Tuple[str, dict, float]]:
    """
    Retrieve similar code chunks for a natural-language (or code) query.

    Returns list of (document_text, metadata, distance).
    Smaller distance means more similar (for Chroma's default distance).
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    embedder = embedder or get_embedder()
    collection = collection or get_chroma_collection(
        persist_dir=persist_dir, collection_name=collection_name
    )

    qvec = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=qvec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[Tuple[str, dict, float]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append((doc, meta, float(dist)))
    return out


def find_most_relevant_chunk(
    query: str,
    *,
    embedder: Optional[SentenceTransformer] = None,
    collection=None,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Optional[Tuple[str, dict, float]]:
    """
    Given a user query, return the single most relevant code chunk from ChromaDB.

    Returns (document_text, metadata, distance) or None if nothing is found.
    """
    hits = search_similar(
        query,
        top_k=1,
        embedder=embedder,
        collection=collection,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    return hits[0] if hits else None

