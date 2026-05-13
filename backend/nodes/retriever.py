"""
Retriever node with resilient Milvus connection.

The connection is lazily established and auto-reconnects if it drops,
preventing the scenario where a stale connection silently blocks all
retrieval and therefore all chat responses.
"""

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional
from state import FamilyLawState
import os
import logging
import threading

logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "family_law_cases"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

# Initialize model (loaded once)
model = SentenceTransformer(MODEL_NAME)
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

# Thread-safe lazy connection
_collection: Optional[Collection] = None
_lock = threading.Lock()


def _connect_and_load() -> Optional[Collection]:
    """Connect to Milvus using URI + token and load collection."""
    try:
        # Disconnect first to clear any stale connection
        try:
            connections.disconnect("default")
        except Exception:
            pass

        connections.connect(
            alias="default",
            uri=MILVUS_URI,
            token=MILVUS_TOKEN
        )
        coll = Collection(COLLECTION_NAME)
        coll.load()
        logger.info(f"✅ Connected to Milvus collection '{COLLECTION_NAME}'")
        return coll
    except Exception as e:
        logger.error(f"❌ Error connecting to Milvus: {e}")
        return None


def _get_collection() -> Optional[Collection]:
    """Get or reconnect Milvus collection (thread-safe)."""
    global _collection
    if _collection is not None:
        return _collection
    with _lock:
        # Double-check inside lock
        if _collection is None:
            _collection = _connect_and_load()
        return _collection


def _reset_collection():
    """Force reconnect on next access."""
    global _collection
    with _lock:
        _collection = None


# Eagerly attempt first connection at import time
_collection = _connect_and_load()

# Expose for health-check endpoint
collection = _collection


def retrieve_documents(state: FamilyLawState) -> Dict:
    """
    Retrieve relevant documents from Milvus based on the query.
    Auto-reconnects if the Milvus connection has dropped.
    """
    global collection

    root_query = state.get("root_query") or ""
    query = state.get("query") or ""
    combined_query = root_query + query

    coll = _get_collection()
    collection = coll  # keep module-level ref updated for health check

    if not coll:
        logger.warning("Milvus collection unavailable — returning empty results")
        return {
            "retrieved_chunks": [],
            "sources": []
        }

    # Generate query embedding
    query_embedding = model.encode([combined_query])[0].tolist()

    # Search in Milvus
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

    try:
        results = coll.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=TOP_K,
            output_fields=["content", "parent_id", "title", "query_text", "url", "category"]
        )
    except Exception as e:
        logger.error(f"❌ Milvus search failed, attempting reconnect: {e}")
        _reset_collection()
        coll = _get_collection()
        collection = coll
        if not coll:
            return {"retrieved_chunks": [], "sources": []}
        # Retry once
        results = coll.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=TOP_K,
            output_fields=["content", "parent_id", "title", "query_text", "url", "category"]
        )

    # Process results
    retrieved_chunks = []
    sources = []

    for hits in results:
        for hit in hits:
            chunk_data = {
                "content": hit.entity.get("content"),
                "score": hit.score,
                "metadata": {
                    "parent_id": hit.entity.get("parent_id"),
                    "title": hit.entity.get("title"),
                    "query_text": hit.entity.get("query_text"),
                    "url": hit.entity.get("url"),
                    "category": hit.entity.get("category")
                }
            }
            retrieved_chunks.append(chunk_data)

            # Add unique sources
            source = {
                "title": hit.entity.get("title"),
                "url": hit.entity.get("url"),
                "category": hit.entity.get("category")
            }
            if source not in sources:
                sources.append(source)

    logger.info(f"✅ Retrieved {len(retrieved_chunks)} chunks")

    return {
        "retrieved_chunks": retrieved_chunks,
        "sources": sources
    }