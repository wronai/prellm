"""UserMemory — stores user interaction history and learned preferences.

Backend: SQLite (MVP) → ChromaDB (production) → Redis (enterprise).
Uses synchronous SQLite for MVP simplicity; async wrappers provided for pipeline integration.

ChromaDB backend enables semantic search over interaction history.
Install with: pip install prellm[memory]
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("prellm.context.user_memory")

_DEFAULT_DB_PATH = ".prellm/user_memory.db"
_DEFAULT_CHROMA_PATH = ".prellm/chroma_db"
_CHROMA_COLLECTION = "prellm_interactions"


class UserMemory:
    """Stores user query history and learned preferences.

    Usage:
        # SQLite (default, no extra deps)
        memory = UserMemory(backend="sqlite", path=".prellm/user_memory.db")

        # ChromaDB (semantic search, requires pip install prellm[memory])
        memory = UserMemory(backend="chromadb", path=".prellm/chroma_db")

        await memory.add_interaction("Deploy app", "Deployment plan...", {"intent": "deploy"})
        recent = await memory.get_recent_context("Deploy", limit=5)
        prefs = await memory.get_user_preferences()
    """

    def __init__(self, backend: str = "sqlite", path: str | Path = _DEFAULT_DB_PATH):
        self.backend = backend
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._chroma_client: Any = None
        self._chroma_collection: Any = None
        self._prefs_conn: sqlite3.Connection | None = None

        if backend == "sqlite":
            self._init_sqlite()
        elif backend == "chromadb":
            self._init_chromadb()

    def _init_sqlite(self) -> None:
        """Initialize SQLite database and tables."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response_summary TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                timestamp REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client and collection.

        Also creates a small SQLite DB for preferences (ChromaDB doesn't do key-value).
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for the ChromaDB backend. "
                "Install with: pip install prellm[memory]"
            )

        self.path.mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(path=str(self.path))
        self._chroma_collection = self._chroma_client.get_or_create_collection(
            name=_CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        # Preferences still in SQLite (simple key-value)
        prefs_path = self.path / "preferences.db"
        self._prefs_conn = sqlite3.connect(str(prefs_path))
        self._prefs_conn.row_factory = sqlite3.Row
        self._prefs_conn.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._prefs_conn.commit()
        logger.info(f"ChromaDB backend initialized at {self.path}")

    async def add_interaction(
        self, query: str, response_summary: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record a user interaction."""
        ts = time.time()
        meta = metadata or {}

        if self.backend == "chromadb" and self._chroma_collection is not None:
            doc_id = f"interaction_{int(ts * 1000)}"
            self._chroma_collection.add(
                documents=[f"{query}\n---\n{response_summary}"],
                metadatas=[{**{k: str(v) for k, v in meta.items()}, "timestamp": str(ts), "query": query}],
                ids=[doc_id],
            )
            logger.debug(f"Stored interaction (chromadb): {query[:80]}...")
            return

        if self._conn is None:
            return
        self._conn.execute(
            "INSERT INTO interactions (query, response_summary, metadata_json, timestamp) VALUES (?, ?, ?, ?)",
            (query, response_summary, json.dumps(meta), ts),
        )
        self._conn.commit()
        logger.debug(f"Stored interaction: {query[:80]}...")

    async def get_recent_context(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Get the most recent/relevant interactions.

        SQLite: returns last N interactions ordered by recency.
        ChromaDB: returns N most semantically similar interactions.
        """
        if self.backend == "chromadb" and self._chroma_collection is not None:
            try:
                results = self._chroma_collection.query(
                    query_texts=[query],
                    n_results=limit,
                )
                items = []
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    parts = doc.split("\n---\n", 1)
                    items.append({
                        "query": meta.get("query", parts[0] if parts else ""),
                        "response_summary": parts[1] if len(parts) > 1 else "",
                        "metadata": {k: v for k, v in meta.items() if k not in ("timestamp", "query")},
                        "timestamp": float(meta.get("timestamp", 0)),
                    })
                return items
            except Exception as e:
                logger.warning(f"ChromaDB query failed: {e}")
                return []

        if self._conn is None:
            return []

        cursor = self._conn.execute(
            "SELECT query, response_summary, metadata_json, timestamp "
            "FROM interactions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "query": row["query"],
                "response_summary": row["response_summary"],
                "metadata": json.loads(row["metadata_json"]),
                "timestamp": row["timestamp"],
            })
        return results

    async def get_user_preferences(self) -> dict[str, str]:
        """Get all learned user preferences."""
        conn = self._prefs_conn if self.backend == "chromadb" else self._conn
        if conn is None:
            return {}
        cursor = conn.execute("SELECT key, value FROM preferences")
        return {row["key"]: row["value"] for row in cursor.fetchall()}

    async def set_preference(self, key: str, value: str) -> None:
        """Set a user preference."""
        conn = self._prefs_conn if self.backend == "chromadb" else self._conn
        if conn is None:
            return
        conn.execute(
            "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        conn.commit()

    async def clear(self) -> None:
        """Clear all stored data (for testing)."""
        if self.backend == "chromadb":
            if self._chroma_collection is not None and self._chroma_client is not None:
                self._chroma_client.delete_collection(_CHROMA_COLLECTION)
                self._chroma_collection = self._chroma_client.get_or_create_collection(
                    name=_CHROMA_COLLECTION,
                    metadata={"hnsw:space": "cosine"},
                )
            if self._prefs_conn:
                self._prefs_conn.execute("DELETE FROM preferences")
                self._prefs_conn.commit()
            return

        if self._conn is None:
            return
        self._conn.execute("DELETE FROM interactions")
        self._conn.execute("DELETE FROM preferences")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._prefs_conn:
            self._prefs_conn.close()
            self._prefs_conn = None
        self._chroma_client = None
        self._chroma_collection = None
