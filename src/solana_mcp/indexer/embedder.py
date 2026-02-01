"""Build vector embeddings and store in LanceDB.

Creates a searchable vector index from chunked content.
"""

import json
import logging
from pathlib import Path
from typing import Callable

from .chunker import Chunk

logger = logging.getLogger(__name__)

# Try to import embedding dependencies
try:
    import lancedb
    from sentence_transformers import SentenceTransformer

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


# Default embedding model
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default data directory
DEFAULT_DATA_DIR = Path.home() / ".solana-mcp"


class Embedder:
    """Generate embeddings and manage LanceDB index."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        data_dir: Path | None = None,
    ):
        if not DEPS_AVAILABLE:
            raise ImportError(
                "Embedding dependencies not installed. "
                "Run: pip install lancedb sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.db_path = self.data_dir / "lancedb"

        # Initialize LanceDB
        self.db = lancedb.connect(str(self.db_path))

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.model.encode(text).tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts).tolist()

    def build_index(
        self,
        chunks: list[Chunk],
        table_name: str = "solana_index",
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict:
        """
        Build vector index from chunks.

        Args:
            chunks: List of content chunks
            table_name: Name of the LanceDB table
            progress_callback: Optional progress callback

        Returns:
            Statistics about the index
        """
        def log(msg: str):
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        if not chunks:
            log("No chunks to index")
            return {"chunks_indexed": 0}

        log(f"Generating embeddings for {len(chunks)} chunks...")

        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]
            embeddings = self.embed_texts(texts)
            all_embeddings.extend(embeddings)

            if (i + batch_size) % 100 == 0 or i + batch_size >= len(chunks):
                log(f"  Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

        # Build records for LanceDB
        records = []
        for chunk, embedding in zip(chunks, all_embeddings):
            record = {
                "content": chunk.content,
                "source_type": chunk.source_type,
                "source_file": chunk.source_file,
                "source_name": chunk.source_name,
                "line_number": chunk.line_number or 0,
                "metadata": json.dumps(chunk.metadata),
                "vector": embedding,
            }
            records.append(record)

        log(f"Writing {len(records)} records to LanceDB...")

        # Drop existing table if exists
        try:
            self.db.drop_table(table_name)
        except Exception as e:
            logger.debug("Table %s does not exist or could not be dropped: %s", table_name, e)

        # Create new table
        table = self.db.create_table(table_name, records)

        # Create index for faster search
        log("Building vector index...")
        try:
            table.create_index(
                metric="cosine",
                num_partitions=min(256, len(records) // 10 + 1),
                num_sub_vectors=min(96, len(records) // 100 + 1),
            )
        except Exception as e:
            log(f"  Index creation failed (will use brute force): {e}")

        log("Index built successfully")

        return {
            "chunks_indexed": len(chunks),
            "table_name": table_name,
            "db_path": str(self.db_path),
            "model": self.model_name,
        }

    def search(
        self,
        query: str,
        table_name: str = "solana_index",
        limit: int = 10,
        source_type: str | None = None,
    ) -> list[dict]:
        """
        Search the index for relevant content.

        Args:
            query: Search query
            table_name: Name of the LanceDB table
            limit: Maximum results to return
            source_type: Filter by source type (rust, simd, docs)

        Returns:
            List of matching results with scores
        """
        try:
            table = self.db.open_table(table_name)
        except Exception:
            return []

        # Generate query embedding
        query_embedding = self.embed_text(query)

        # Search
        results = table.search(query_embedding).limit(limit * 2 if source_type else limit)

        # Convert to list of dicts
        matches = []
        for row in results.to_list():
            # Filter by source type if specified
            if source_type and row.get("source_type") != source_type:
                continue

            matches.append({
                "content": row["content"],
                "source_type": row["source_type"],
                "source_file": row["source_file"],
                "source_name": row["source_name"],
                "line_number": row["line_number"],
                "metadata": json.loads(row["metadata"]) if row.get("metadata") else {},
                "score": float(row.get("_distance", 0)),
            })

            if len(matches) >= limit:
                break

        return matches

    def search_runtime(self, query: str, limit: int = 10) -> list[dict]:
        """Search only Rust runtime code."""
        return self.search(query, source_type="rust", limit=limit)

    def search_simds(self, query: str, limit: int = 10) -> list[dict]:
        """Search only SIMDs."""
        return self.search(query, source_type="simd", limit=limit)


def build_index(
    chunks: list[Chunk],
    data_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Convenience function to build the index.

    Args:
        chunks: List of content chunks
        data_dir: Base data directory
        model_name: Embedding model name
        progress_callback: Optional progress callback

    Returns:
        Statistics about the index
    """
    embedder = Embedder(model_name=model_name, data_dir=data_dir)
    return embedder.build_index(chunks, progress_callback=progress_callback)


def search(
    query: str,
    data_dir: Path | None = None,
    model_name: str = DEFAULT_MODEL,
    limit: int = 10,
    source_type: str | None = None,
) -> list[dict]:
    """
    Convenience function to search the index.

    Args:
        query: Search query
        data_dir: Base data directory
        model_name: Embedding model name
        limit: Maximum results
        source_type: Filter by source type

    Returns:
        List of matching results
    """
    embedder = Embedder(model_name=model_name, data_dir=data_dir)
    return embedder.search(query, limit=limit, source_type=source_type)


def get_index_stats(data_dir: Path | None = None) -> dict | None:
    """Get statistics about the current index."""
    if not DEPS_AVAILABLE:
        return None

    data_dir = data_dir or DEFAULT_DATA_DIR
    db_path = data_dir / "lancedb"

    if not db_path.exists():
        return None

    try:
        db = lancedb.connect(str(db_path))
        table = db.open_table("solana_index")

        # Count by source type
        all_rows = table.to_pandas()
        source_counts = all_rows["source_type"].value_counts().to_dict()

        return {
            "total_chunks": len(all_rows),
            "by_source_type": source_counts,
            "db_path": str(db_path),
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test search
    import sys

    if len(sys.argv) < 2:
        print("Usage: embedder.py <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"Searching for: {query}")

    results = search(query, limit=5)
    for i, result in enumerate(results):
        print(f"\n{i + 1}. {result['source_name']} ({result['source_type']})")
        print(f"   File: {result['source_file']}:{result['line_number']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Content: {result['content'][:200]}...")
