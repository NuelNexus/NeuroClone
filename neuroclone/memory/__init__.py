"""Long-term memory: embeddings, a SQLite vector store, and the memory manager."""

from .embeddings import (
    Embedder,
    HashingEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    create_embedder,
    embedder_for,
)
from .manager import MemoryManager, Recall, Turn, estimate_importance, humanize_age
from .store import MemoryRecord, MemoryStore, UserProfile

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "MemoryManager",
    "MemoryRecord",
    "MemoryStore",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "Recall",
    "Turn",
    "UserProfile",
    "create_embedder",
    "embedder_for",
    "estimate_importance",
    "humanize_age",
]
