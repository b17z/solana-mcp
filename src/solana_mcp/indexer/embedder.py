"""Build vector embeddings and store in LanceDB.

Creates a searchable vector index from chunked content.
Supports incremental indexing to avoid full rebuilds.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .chunker import Chunk
from .manifest import (
    FileChange,
    FileEntry,
    Manifest,
    compute_changes,
    compute_file_hash,
    get_file_mtime_ns,
    load_manifest,
    needs_full_rebuild,
    save_manifest,
)

logger = logging.getLogger(__name__)

# Try to import embedding dependencies
try:
    import lancedb
    from sentence_transformers import SentenceTransformer

    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


# Default embedding model
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Default data directory
DEFAULT_DATA_DIR = Path.home() / ".solana-mcp"


@dataclass
class IndexStats:
    """Statistics from an indexing operation."""

    full_rebuild: bool = False
    rebuild_reason: str = ""
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def files_changed(self) -> int:
        return self.files_added + self.files_modified + self.files_deleted

    @property
    def is_incremental(self) -> bool:
        return not self.full_rebuild and self.files_changed > 0

    @property
    def is_noop(self) -> bool:
        return not self.full_rebuild and self.files_changed == 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.full_rebuild:
            return (
                f"Full rebuild ({self.rebuild_reason}): "
                f"{self.chunks_added} chunks indexed"
            )
        if self.is_noop:
            return "No changes detected"
        parts = []
        if self.files_added:
            parts.append(f"{self.files_added} added")
        if self.files_modified:
            parts.append(f"{self.files_modified} modified")
        if self.files_deleted:
            parts.append(f"{self.files_deleted} deleted")
        return (
            f"Incremental update: {', '.join(parts)} "
            f"(+{self.chunks_added}/-{self.chunks_deleted} chunks)"
        )


@dataclass
class DryRunResult:
    """Result of a dry-run analysis."""

    would_rebuild: bool = False
    rebuild_reason: str = ""
    files_to_add: list[str] = field(default_factory=list)
    files_to_modify: list[str] = field(default_factory=list)
    files_to_delete: list[str] = field(default_factory=list)
    estimated_chunks_add: int = 0
    estimated_chunks_delete: int = 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        if self.would_rebuild:
            return f"Would perform full rebuild: {self.rebuild_reason}"

        if not self.files_to_add and not self.files_to_modify and not self.files_to_delete:
            return "No changes detected"

        parts = []
        if self.files_to_add:
            parts.append(f"Add {len(self.files_to_add)} files")
        if self.files_to_modify:
            parts.append(f"Modify {len(self.files_to_modify)} files")
        if self.files_to_delete:
            parts.append(f"Delete {len(self.files_to_delete)} files")

        return (
            f"Would update: {', '.join(parts)} "
            f"(~{self.estimated_chunks_add} add, ~{self.estimated_chunks_delete} delete)"
        )


class IncrementalEmbedder:
    """
    Embedder with incremental indexing support.

    Tracks file state via manifest and only re-embeds changed files.
    """

    def __init__(
        self,
        data_dir: Path,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
    ):
        if not DEPS_AVAILABLE:
            raise ImportError(
                "Embedding dependencies not installed. "
                "Run: pip install lancedb sentence-transformers"
            )

        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.db_path = self.data_dir / "lancedb"
        self.manifest_path = self.data_dir / "manifest.json"
        self.table_name = "solana_index"

        self._model: SentenceTransformer | None = None
        self._db: Any = None

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def db(self) -> Any:
        if self._db is None:
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    def get_current_config(self) -> dict[str, Any]:
        """Get current configuration for manifest comparison."""
        return {
            "embedding_model": self.model_name,
            "chunk_config": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
        }

    def dry_run(
        self,
        current_files: dict[str, Path],
        file_types: dict[str, str],
    ) -> DryRunResult:
        """
        Analyze what would change without actually indexing.

        Args:
            current_files: Dict mapping relative paths to absolute Paths
            file_types: Dict mapping relative paths to file types

        Returns:
            DryRunResult describing pending changes
        """
        manifest = load_manifest(self.manifest_path)
        config = self.get_current_config()

        # Check if full rebuild needed
        rebuild_needed, reason = needs_full_rebuild(manifest, config)
        if rebuild_needed:
            return DryRunResult(would_rebuild=True, rebuild_reason=reason)

        # Compute changes
        changes = compute_changes(manifest, current_files)

        result = DryRunResult()
        for change in changes:
            if change.change_type == "add":
                result.files_to_add.append(change.path)
                result.estimated_chunks_add += 10  # Rough estimate
            elif change.change_type == "modify":
                result.files_to_modify.append(change.path)
                result.estimated_chunks_add += 10
                result.estimated_chunks_delete += len(change.old_chunk_ids)
            elif change.change_type == "delete":
                result.files_to_delete.append(change.path)
                result.estimated_chunks_delete += len(change.old_chunk_ids)

        return result

    def index(
        self,
        current_files: dict[str, Path],
        file_types: dict[str, str],
        chunk_fn: Callable[[Path, str, Path], list[Chunk]] | None = None,
        force_full: bool = False,
        progress_callback: Callable[[str], None] | None = None,
    ) -> IndexStats:
        """
        Index files incrementally.

        Args:
            current_files: Dict mapping relative paths to absolute Paths
            file_types: Dict mapping relative paths to file types
            chunk_fn: Function to chunk a single file
            force_full: Force full rebuild
            progress_callback: Optional progress callback

        Returns:
            IndexStats with operation details
        """

        def log(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                logger.info(msg)

        manifest = load_manifest(self.manifest_path)
        config = self.get_current_config()

        # Check if full rebuild needed
        if force_full:
            return self._full_rebuild(
                current_files, file_types, chunk_fn, "Forced rebuild", log
            )

        rebuild_needed, reason = needs_full_rebuild(manifest, config)
        if rebuild_needed:
            return self._full_rebuild(
                current_files, file_types, chunk_fn, reason, log
            )

        # Incremental update
        changes = compute_changes(manifest, current_files)
        if not changes:
            log("No changes detected")
            return IndexStats()

        return self._incremental_update(
            manifest, current_files, file_types, changes, chunk_fn, log
        )

    def _full_rebuild(
        self,
        current_files: dict[str, Path],
        file_types: dict[str, str],
        chunk_fn: Callable[[Path, str, Path], list[Chunk]] | None,
        reason: str,
        log: Callable[[str], None],
    ) -> IndexStats:
        """Perform a full index rebuild."""
        log(f"Full rebuild: {reason}")
        stats = IndexStats(full_rebuild=True, rebuild_reason=reason)

        # Collect all chunks
        all_chunks: list[Chunk] = []

        if chunk_fn:
            for rel_path, abs_path in current_files.items():
                file_type = file_types.get(rel_path, "docs")
                try:
                    chunks = chunk_fn(abs_path, file_type, self.data_dir)
                    all_chunks.extend(chunks)
                except Exception as e:
                    stats.errors.append(f"{rel_path}: {e}")

        if not all_chunks:
            log("No chunks to index")
            return stats

        log(f"Generating embeddings for {len(all_chunks)} chunks...")

        # Generate embeddings
        embeddings = self._embed_chunks(all_chunks, log)

        # Build records
        records = self._build_records(all_chunks, embeddings)

        # Drop and recreate table
        log("Writing to LanceDB...")
        try:
            self.db.drop_table(self.table_name)
        except Exception:
            pass  # Table may not exist

        self.db.create_table(self.table_name, records)
        stats.chunks_added = len(records)

        # Build new manifest
        manifest = Manifest(
            embedding_model=self.model_name,
            chunk_config=self.get_current_config()["chunk_config"],
        )

        # Track files
        for rel_path, abs_path in current_files.items():
            file_hash = compute_file_hash(abs_path)
            mtime = get_file_mtime_ns(abs_path)
            chunk_ids = [
                c.chunk_id for c in all_chunks if c.source_file == rel_path
            ]
            manifest.files[rel_path] = FileEntry(
                sha256=file_hash,
                mtime_ns=mtime,
                chunk_ids=chunk_ids,
            )

        save_manifest(manifest, self.manifest_path)
        log(f"Indexed {stats.chunks_added} chunks")

        return stats

    def _incremental_update(
        self,
        manifest: Manifest,
        current_files: dict[str, Path],
        file_types: dict[str, str],
        changes: list[FileChange],
        chunk_fn: Callable[[Path, str, Path], list[Chunk]] | None,
        log: Callable[[str], None],
    ) -> IndexStats:
        """Apply incremental updates."""
        stats = IndexStats()

        # Collect chunks to add and IDs to delete
        chunks_to_add: list[Chunk] = []
        ids_to_delete: list[str] = []

        for change in changes:
            if change.change_type == "add":
                stats.files_added += 1
                if chunk_fn and change.path in current_files:
                    abs_path = current_files[change.path]
                    file_type = file_types.get(change.path, "docs")
                    try:
                        chunks = chunk_fn(abs_path, file_type, self.data_dir)
                        chunks_to_add.extend(chunks)
                    except Exception as e:
                        stats.errors.append(f"{change.path}: {e}")

            elif change.change_type == "modify":
                stats.files_modified += 1
                ids_to_delete.extend(change.old_chunk_ids)
                if chunk_fn and change.path in current_files:
                    abs_path = current_files[change.path]
                    file_type = file_types.get(change.path, "docs")
                    try:
                        chunks = chunk_fn(abs_path, file_type, self.data_dir)
                        chunks_to_add.extend(chunks)
                    except Exception as e:
                        stats.errors.append(f"{change.path}: {e}")

            elif change.change_type == "delete":
                stats.files_deleted += 1
                ids_to_delete.extend(change.old_chunk_ids)

        # Apply deletions
        if ids_to_delete:
            log(f"Deleting {len(ids_to_delete)} old chunks...")
            self._delete_chunks(ids_to_delete)
            stats.chunks_deleted = len(ids_to_delete)

        # Apply additions
        if chunks_to_add:
            log(f"Adding {len(chunks_to_add)} new chunks...")
            embeddings = self._embed_chunks(chunks_to_add, log)
            records = self._build_records(chunks_to_add, embeddings)
            self._add_chunks(records)
            stats.chunks_added = len(records)

        # Update manifest
        for change in changes:
            if change.change_type == "delete":
                del manifest.files[change.path]
            elif change.change_type in ("add", "modify"):
                if change.path in current_files:
                    abs_path = current_files[change.path]
                    file_hash = compute_file_hash(abs_path)
                    mtime = get_file_mtime_ns(abs_path)
                    chunk_ids = [
                        c.chunk_id
                        for c in chunks_to_add
                        if c.source_file == change.path
                    ]
                    manifest.files[change.path] = FileEntry(
                        sha256=file_hash,
                        mtime_ns=mtime,
                        chunk_ids=chunk_ids,
                    )

        save_manifest(manifest, self.manifest_path)
        log(stats.summary())

        return stats

    def _embed_chunks(
        self,
        chunks: list[Chunk],
        log: Callable[[str], None],
    ) -> list[list[float]]:
        """Generate embeddings for chunks."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c.content for c in batch]
            embeddings = self.model.encode(texts).tolist()
            all_embeddings.extend(embeddings)

            if (i + self.batch_size) % 100 == 0 or i + self.batch_size >= len(chunks):
                log(f"  Embedded {min(i + self.batch_size, len(chunks))}/{len(chunks)}")

        return all_embeddings

    def _build_records(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> list[dict[str, Any]]:
        """Build LanceDB records from chunks and embeddings."""
        records = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            records.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "source_type": chunk.source_type,
                "source_file": chunk.source_file,
                "source_name": chunk.source_name,
                "line_number": chunk.line_number or 0,
                "metadata": json.dumps(chunk.metadata),
                "vector": embedding,
            })
        return records

    def _delete_chunks(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID from the index."""
        try:
            table = self.db.open_table(self.table_name)
            # LanceDB delete with filter
            for chunk_id in chunk_ids:
                # Sanitized in manifest.py, but be extra safe
                safe_id = chunk_id.replace("'", "''")
                table.delete(f"chunk_id = '{safe_id}'")
        except Exception as e:
            logger.warning("Failed to delete chunks: %s", e)

    def _add_chunks(self, records: list[dict[str, Any]]) -> None:
        """Add new chunks to the index."""
        try:
            table = self.db.open_table(self.table_name)
            table.add(records)
        except Exception as e:
            logger.error("Failed to add chunks: %s", e)
            raise


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
        for chunk, embedding in zip(chunks, all_embeddings, strict=True):
            record = {
                "chunk_id": chunk.chunk_id,
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
