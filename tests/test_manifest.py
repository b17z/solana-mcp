"""Tests for manifest functionality."""


import pytest

from solana_mcp.indexer.manifest import (
    FileEntry,
    Manifest,
    ManifestCorruptedError,
    ManifestValidationError,
    compute_file_hash,
    generate_chunk_id,
    get_file_mtime_ns,
    load_manifest,
    save_manifest,
)


class TestFileEntry:
    """Tests for FileEntry dataclass."""

    def test_to_dict(self):
        """FileEntry should serialize to dict."""
        entry = FileEntry(
            sha256="a" * 64,
            mtime_ns=1234567890,
            chunk_ids=["chunk1", "chunk2"],
        )
        d = entry.to_dict()
        assert d["sha256"] == "a" * 64
        assert d["mtime_ns"] == 1234567890
        assert d["chunk_ids"] == ["chunk1", "chunk2"]

    def test_from_dict(self):
        """FileEntry should deserialize from dict."""
        d = {
            "sha256": "b" * 64,
            "mtime_ns": 9876543210,
            "chunk_ids": ["c1", "c2", "c3"],
        }
        entry = FileEntry.from_dict(d)
        assert entry.sha256 == "b" * 64
        assert entry.mtime_ns == 9876543210
        assert entry.chunk_ids == ["c1", "c2", "c3"]

    def test_from_dict_missing_chunk_ids(self):
        """FileEntry should handle missing chunk_ids."""
        d = {"sha256": "c" * 64, "mtime_ns": 1111}
        entry = FileEntry.from_dict(d)
        assert entry.chunk_ids == []


class TestManifest:
    """Tests for Manifest dataclass."""

    def test_to_dict(self, sample_manifest):
        """Manifest should serialize to dict."""
        manifest = Manifest.from_dict(sample_manifest)
        d = manifest.to_dict()
        assert d["version"] == "1.0.0"
        assert "files" in d
        assert "runtime/stake.rs" in d["files"]

    def test_from_dict_valid(self, sample_manifest):
        """Manifest should parse valid dict."""
        manifest = Manifest.from_dict(sample_manifest)
        assert manifest.version == "1.0.0"
        assert manifest.embedding_model == "all-MiniLM-L6-v2"
        assert len(manifest.files) == 1

    def test_from_dict_missing_version(self):
        """Manifest should reject missing version."""
        with pytest.raises(ManifestValidationError, match="version"):
            Manifest.from_dict({"files": {}})

    def test_from_dict_missing_files(self):
        """Manifest should reject missing files."""
        with pytest.raises(ManifestValidationError, match="files"):
            Manifest.from_dict({"version": "1.0.0"})

    def test_from_dict_invalid_version_format(self):
        """Manifest should reject invalid version format."""
        with pytest.raises(ManifestValidationError, match="Invalid version"):
            Manifest.from_dict({"version": "invalid", "files": {}})

    def test_from_dict_invalid_sha256(self):
        """Manifest should reject invalid SHA256."""
        with pytest.raises(ManifestValidationError, match="Invalid SHA256"):
            Manifest.from_dict({
                "version": "1.0.0",
                "files": {
                    "test.rs": {"sha256": "short", "mtime_ns": 0}
                },
            })


class TestManifestSaveLoad:
    """Tests for manifest save/load operations."""

    def test_save_load_roundtrip(self, temp_data_dir, sample_manifest):
        """Manifest should survive save/load roundtrip."""
        manifest = Manifest.from_dict(sample_manifest)
        manifest_path = temp_data_dir / "manifest.json"

        save_manifest(manifest, manifest_path)
        loaded = load_manifest(manifest_path)

        assert loaded is not None
        assert loaded.version == manifest.version
        assert loaded.embedding_model == manifest.embedding_model
        assert len(loaded.files) == len(manifest.files)

    def test_load_nonexistent(self, temp_data_dir):
        """Loading nonexistent manifest should return None."""
        result = load_manifest(temp_data_dir / "nonexistent.json")
        assert result is None

    def test_load_corrupted_json(self, temp_data_dir):
        """Loading corrupted JSON should raise error and create backup."""
        manifest_path = temp_data_dir / "manifest.json"
        manifest_path.write_text("{invalid json")

        with pytest.raises(ManifestCorruptedError):
            load_manifest(manifest_path)

        # Check backup was created
        backup_path = manifest_path.with_suffix(".json.corrupted")
        assert backup_path.exists()

    def test_atomic_write(self, temp_data_dir, sample_manifest):
        """Save should use atomic write (no partial writes)."""
        manifest = Manifest.from_dict(sample_manifest)
        manifest_path = temp_data_dir / "manifest.json"

        save_manifest(manifest, manifest_path)

        # Temp file should not exist
        temp_path = manifest_path.with_suffix(".json.tmp")
        assert not temp_path.exists()

        # Main file should exist and be valid
        assert manifest_path.exists()
        loaded = load_manifest(manifest_path)
        assert loaded is not None


class TestFileHashing:
    """Tests for file hash utilities."""

    def test_compute_file_hash(self, temp_data_dir):
        """Should compute consistent SHA256 hash."""
        test_file = temp_data_dir / "test.txt"
        test_file.write_text("Hello, World!")

        hash1 = compute_file_hash(test_file)
        hash2 = compute_file_hash(test_file)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
        assert all(c in "0123456789abcdef" for c in hash1)

    def test_get_file_mtime_ns(self, temp_data_dir):
        """Should get file mtime in nanoseconds."""
        test_file = temp_data_dir / "test.txt"
        test_file.write_text("test")

        mtime = get_file_mtime_ns(test_file)

        assert isinstance(mtime, int)
        assert mtime > 0


class TestChunkIdGeneration:
    """Tests for chunk ID generation."""

    def test_generates_deterministic_id(self):
        """Same inputs should produce same ID."""
        id1 = generate_chunk_id("sol", "rust", "test.rs", 0, "content")
        id2 = generate_chunk_id("sol", "rust", "test.rs", 0, "content")
        assert id1 == id2

    def test_different_index_different_id(self):
        """Different chunk indices should produce different IDs."""
        id1 = generate_chunk_id("sol", "rust", "test.rs", 0, "content")
        id2 = generate_chunk_id("sol", "rust", "test.rs", 1, "content")
        assert id1 != id2

    def test_different_file_different_id(self):
        """Different files should produce different IDs."""
        id1 = generate_chunk_id("sol", "rust", "file1.rs", 0, "content")
        id2 = generate_chunk_id("sol", "rust", "file2.rs", 0, "content")
        assert id1 != id2

    def test_different_content_different_id(self):
        """Different content should produce different IDs."""
        id1 = generate_chunk_id("sol", "rust", "test.rs", 0, "content1")
        id2 = generate_chunk_id("sol", "rust", "test.rs", 0, "content2")
        assert id1 != id2

    def test_id_format(self):
        """ID should follow expected format."""
        chunk_id = generate_chunk_id("sol", "rust", "test.rs", 5, "content")
        parts = chunk_id.split("_")
        assert parts[0] == "sol"
        assert parts[1] == "rust"
        assert len(parts[2]) == 8  # path hash
        assert parts[3] == "0005"  # zero-padded index
        assert len(parts[4]) == 8  # content hash
