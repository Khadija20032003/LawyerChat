from __future__ import annotations

import hashlib
import math
import os
import datetime as dt
import logging
from typing import List, Sequence

logger = logging.getLogger(__name__)

# IMPORTANT: must be set BEFORE importing chromadb anywhere
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"


def make_chunk_ids(cik: str, accession: str, num_chunks: int) -> List[str]:
    return [
        hashlib.sha1(f"{cik}__{accession}__{i}".encode()).hexdigest()
        for i in range(num_chunks)
    ]


def init_chroma(
    persist_dir: str,
    collection_name: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    # Delay imports so env vars above take effect
    import chromadb
    from chromadb.config import Settings
    from langchain_huggingface import HuggingFaceEmbeddings

    # Prefer langchain-chroma if installed; fallback otherwise
    try:
        from langchain_chroma import Chroma as LCChroma
    except Exception:
        from langchain_community.vectorstores import Chroma as LCChroma

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(
            anonymized_telemetry=False,
            chroma_telemetry_impl="none",
        ),
    )

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    return LCChroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def upsert_batched(vectordb, docs: Sequence, ids: Sequence[str], batch_size: int = 10_000) -> int:
    if len(docs) != len(ids):
        raise ValueError("docs/ids length mismatch")

    n = len(docs)
    if n == 0:
        return 0

    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        vectordb.add_documents(docs[i:j], ids=ids[i:j])
        print(f"Upserted {j}/{n} chunks")

    return math.ceil(n / batch_size)


def check_date_in_vectordb(vectordb, d: dt.date) -> bool:
    """
    Check if filings from a specific date already exist in the vector database.
    
    Args:
        vectordb: ChromaDB collection instance
        d: Date to check
        
    Returns:
        True if filings from this date exist in the database
    """
    # Try both date formats
    date_str_1 = d.strftime("%Y-%m-%d")  # 2024-12-23
    date_str_2 = d.strftime("%Y%m%d")     # 20241223
    
    try:
        results_1 = vectordb._collection.get(
            where={"filingDate": date_str_1},
            limit=1
        )
        results_2 = vectordb._collection.get(
            where={"filingDate": date_str_2},
            limit=1
        )
        return len(results_1['ids']) > 0 or len(results_2['ids']) > 0
    except Exception as e:
        logger.warning(f"Error checking vectordb: {e}")
        return False