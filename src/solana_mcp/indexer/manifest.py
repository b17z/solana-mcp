"""Manifest for tracking indexed files and enabling incremental updates.

The manifest tracks:
- File paths with their SHA256 hashes and mtimes
- Chunk IDs generated for each file
- Embedding model and chunking configuration
- Repository versions (git commit hashes)

This enables incremental indexing by detecting changed files.
"""

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..logging import get_logger

logger = get_logger("manifest")

# Maximum manifest file size (10MB)
MAX_MANIFEST_SIZE = 10 * 1024 * 1024

# Maximum chunks per file (sanity check)
MAX_CHUNKS_PER_FILE = 10000

# Version for manifest format migrations
MANIFEST_VERSION = "1.0.0"


class ManifestError(Exception):
    """Base exception for manifest errors."""

    pass


class ManifestValidationError(ManifestError):
    """Raised when manifest validation fails."""

    pass


class ManifestCorruptedError(ManifestError):
    """Raised when manifest file is corrupted."""

    pass


@dataclass
class FileEntry:
    """Metadata for a tracked file."""

    sha256: str
    mtime_ns: int
    chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "mtime_ns": self.mtime_ns,
            "chunk_ids": self.chunk_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileEntry":
        return cls(
            sha256=data["sha256"],
            mtime_ns=data["mtime_ns"],
            chunk_ids=data.get("chunk_ids", []),
        )


@dataclass
class Manifest:
    """
    Tracks indexed files for incremental updates.

    The manifest is persisted to JSON and loaded on startup to detect
    which files have changed and need re-indexing.
    """

    version: str = MANIFEST_VERSION
    updated_at: str = ""
    embedding_model: str = ""
    chunk_config: dict[str, int] = field(default_factory=dict)
    files: dict[str, FileEntry] = field(default_factory=dict)
    repo_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "embedding_model": self.embedding_model,
            "chunk_config": self.chunk_config,
            "files": {path: entry.to_dict() for path, entry in self.files.items()},
            "repo_versions": self.repo_versions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        """Parse manifest from dictionary with validation."""
        # Validate required fields
        if "version" not in data:
            raise ManifestValidationError("Missing required field: version")
        if "files" not in data:
            raise ManifestValidationError("Missing required field: files")
        if not isinstance(data.get("files"), dict):
            raise ManifestValidationError("'files' must be a dictionary")

        # Validate version format
        version = data["version"]
        if not re.match(r"^\d+\.\d+\.\d+$", version):
            raise ManifestValidationError(f"Invalid version format: {version}")

        files = {}
        for path, entry_data in data["files"].items():
            # Security: validate path doesn't contain traversal
            _validate_path(path)

            # Validate SHA256 format
            sha256 = entry_data.get("sha256", "")
            if not re.match(r"^[a-f0-9]{64}$", sha256):
                raise ManifestValidationError(f"Invalid SHA256 for {path}: {sha256}")

            # Validate chunk count
            chunk_ids = entry_data.get("chunk_ids", [])
            if len(chunk_ids) > MAX_CHUNKS_PER_FILE:
                raise ManifestValidationError(
                    f"Too many chunks for {path}: {len(chunk_ids)} > {MAX_CHUNKS_PER_FILE}"
                )

            files[path] = FileEntry.from_dict(entry_data)

        return cls(
            version=data["version"],
            updated_at=data.get("updated_at", ""),
            embedding_model=data.get("embedding_model", ""),
            chunk_config=data.get("chunk_config", {}),
            files=files,
            repo_versions=data.get("repo_versions", {}),
        )


def _validate_path(path: str) -> None:
    """Validate a file path for security.

    Raises ManifestValidationError if path is invalid or potentially malicious.
    """
    # Check for path traversal
    if ".." in path:
        raise ManifestValidationError(f"Path traversal detected in: {path}")

    # Check for absolute paths
    if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
        raise ManifestValidationError(f"Absolute paths not allowed: {path}")

    # Check for null bytes
    if "\x00" in path:
        raise ManifestValidationError(f"Null bytes in path: {path}")

    # Check for special characters that could cause issues
    if any(c in path for c in ["\n", "\r", "\t"]):
        raise ManifestValidationError(f"Invalid characters in path: {path}")


def _sanitize_chunk_id(chunk_id: str) -> str:
    """Sanitize a chunk ID for safe use in queries.

    Returns sanitized ID or raises ManifestValidationError if invalid.
    """
    # Only allow alphanumeric, underscore, and hyphen
    if not re.match(r"^[a-zA-Z0-9_-]+$", chunk_id):
        raise ManifestValidationError(f"Invalid chunk ID format: {chunk_id}")

    # Reasonable length limit
    if len(chunk_id) > 200:
        raise ManifestValidationError(f"Chunk ID too long: {len(chunk_id)}")

    return chunk_id


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file_mtime_ns(file_path: Path) -> int:
    """Get file modification time in nanoseconds."""
    return file_path.stat().st_mtime_ns


def load_manifest(manifest_path: Path) -> Manifest | None:
    """
    Load manifest from disk.

    Returns None if manifest doesn't exist.
    Raises ManifestCorruptedError if manifest is corrupted (and creates backup).
    """
    if not manifest_path.exists():
        return None

    # Check file size
    file_size = manifest_path.stat().st_size
    if file_size > MAX_MANIFEST_SIZE:
        raise ManifestValidationError(
            f"Manifest file too large: {file_size} > {MAX_MANIFEST_SIZE}"
        )

    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        return Manifest.from_dict(data)
    except json.JSONDecodeError as e:
        # Create backup of corrupted file
        backup_path = manifest_path.with_suffix(".json.corrupted")
        logger.warning(
            "Manifest corrupted, creating backup at %s: %s", backup_path, e
        )
        shutil.copy(manifest_path, backup_path)
        raise ManifestCorruptedError(f"Manifest JSON corrupted: {e}") from e
    except ManifestValidationError:
        # Re-raise validation errors
        raise
    except Exception as e:
        raise ManifestCorruptedError(f"Failed to load manifest: {e}") from e


def save_manifest(manifest: Manifest, manifest_path: Path) -> None:
    """
    Save manifest to disk atomically.

    Uses write-to-temp-then-rename pattern to prevent corruption
    from interrupted writes.
    """
    manifest.updated_at = datetime.now(UTC).isoformat()

    # Ensure parent directory exists
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first
    temp_path = manifest_path.with_suffix(".json.tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        temp_path.replace(manifest_path)
        logger.debug("Saved manifest to %s", manifest_path)

    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise


def generate_chunk_id(
    project: str,
    source_type: str,
    source_file: str,
    chunk_index: int,
    content: str,
) -> str:
    """
    Generate a unique, deterministic chunk ID.

    Format: {project}_{source_type}_{path_hash}_{index:04d}_{content_hash}

    The content hash ensures that if content changes, the ID changes,
    enabling proper delta updates.

    Args:
        project: Project identifier (e.g., "sol")
        source_type: Type of source (e.g., "rust", "simd", "docs")
        source_file: Relative path to source file
        chunk_index: Index of this chunk within the file
        content: Chunk content for hashing

    Returns:
        Unique chunk ID string
    """
    # Validate inputs
    _validate_path(source_file)

    path_hash = hashlib.sha256(source_file.encode()).hexdigest()[:8]
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]

    chunk_id = f"{project}_{source_type}_{path_hash}_{chunk_index:04d}_{content_hash}"

    # Validate the generated ID
    _sanitize_chunk_id(chunk_id)

    return chunk_id


@dataclass
class FileChange:
    """Represents a detected file change."""

    path: str
    change_type: str  # 'add', 'modify', 'delete'
    old_chunk_ids: list[str] = field(default_factory=list)


def compute_changes(
    manifest: Manifest | None,
    current_files: dict[str, Path],
    check_hashes: bool = True,
) -> list[FileChange]:
    """
    Compute file changes between manifest and current state.

    Uses a two-phase check:
    1. Fast path: Compare mtime - if unchanged, skip file
    2. Slow path: If mtime changed, compare hash to detect actual changes

    Args:
        manifest: Previous manifest (None for fresh index)
        current_files: Dict mapping relative paths to absolute Paths
        check_hashes: If False, skip hash checking (assume mtime change = content change)

    Returns:
        List of FileChange objects describing what changed
    """
    changes = []
    manifest_files = manifest.files if manifest else {}

    # Find new and modified files
    for rel_path, abs_path in current_files.items():
        _validate_path(rel_path)

        if rel_path not in manifest_files:
            # New file
            changes.append(FileChange(path=rel_path, change_type="add"))
            continue

        entry = manifest_files[rel_path]
        current_mtime = get_file_mtime_ns(abs_path)

        if current_mtime == entry.mtime_ns:
            # Fast path: mtime unchanged, assume no change
            continue

        if check_hashes:
            # Slow path: mtime changed, check hash
            current_hash = compute_file_hash(abs_path)
            if current_hash == entry.sha256:
                # Content unchanged, just update mtime (no re-embedding needed)
                # We'll handle this separately to avoid unnecessary work
                continue

        # File modified
        changes.append(
            FileChange(
                path=rel_path,
                change_type="modify",
                old_chunk_ids=entry.chunk_ids.copy(),
            )
        )

    # Find deleted files
    for rel_path, entry in manifest_files.items():
        if rel_path not in current_files:
            changes.append(
                FileChange(
                    path=rel_path,
                    change_type="delete",
                    old_chunk_ids=entry.chunk_ids.copy(),
                )
            )

    return changes


def needs_full_rebuild(manifest: Manifest | None, config: dict[str, Any]) -> tuple[bool, str]:
    """
    Check if a full rebuild is needed due to config changes.

    Args:
        manifest: Previous manifest
        config: Current configuration dict with 'embedding_model' and 'chunk_config'

    Returns:
        Tuple of (needs_rebuild, reason)
    """
    if manifest is None:
        return True, "No manifest exists"

    # Check embedding model change
    current_model = config.get("embedding_model", "")
    if manifest.embedding_model and manifest.embedding_model != current_model:
        return True, f"Embedding model changed: {manifest.embedding_model} -> {current_model}"

    # Check chunk config change
    current_chunk_config = config.get("chunk_config", {})
    if manifest.chunk_config and manifest.chunk_config != current_chunk_config:
        return True, f"Chunk config changed: {manifest.chunk_config} -> {current_chunk_config}"

    return False, ""
