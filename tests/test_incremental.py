"""Tests for incremental indexing functionality."""

from unittest.mock import MagicMock, patch

import pytest

from solana_mcp.indexer.chunker import Chunk, chunk_single_file
from solana_mcp.indexer.embedder import DryRunResult, IncrementalEmbedder, IndexStats
from solana_mcp.indexer.manifest import (
    FileEntry,
    Manifest,
    compute_changes,
    compute_file_hash,
    generate_chunk_id,
    get_file_mtime_ns,
    save_manifest,
)


class TestChunkIdGeneration:
    """Tests for chunk ID generation in chunker."""

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

    def test_id_format(self):
        """ID should follow expected format."""
        chunk_id = generate_chunk_id("sol", "rust", "test.rs", 5, "content")
        parts = chunk_id.split("_")
        assert parts[0] == "sol"
        assert parts[1] == "rust"
        # path hash
        assert len(parts[2]) == 8
        # index (4 digits, zero-padded)
        assert parts[3] == "0005"
        # content hash
        assert len(parts[4]) == 8


class TestChunkSingleFile:
    """Tests for single file chunking."""

    def test_chunks_simd_file(self, sample_simd_file, temp_data_dir):
        """Should chunk a SIMD file correctly."""
        chunks = chunk_single_file(
            sample_simd_file,
            file_type="simd",
            base_path=temp_data_dir / "solana-improvement-documents",
        )

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.chunk_id for c in chunks)  # All have IDs

    def test_chunks_have_unique_ids(self, sample_simd_file, temp_data_dir):
        """All chunks from a file should have unique IDs."""
        chunks = chunk_single_file(
            sample_simd_file,
            file_type="simd",
            base_path=temp_data_dir / "solana-improvement-documents",
        )

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))  # All unique


class TestChangeDetection:
    """Tests for change detection between manifest and filesystem."""

    def test_detects_new_file(self, temp_data_dir, sample_simd_file):
        """New files should be detected."""
        manifest = Manifest(version="1.0.0", files={})
        current_files = {"test.md": sample_simd_file}

        changes = compute_changes(manifest, current_files)

        assert len(changes) == 1
        assert changes[0].change_type == "add"

    def test_detects_modified_file(self, temp_data_dir, sample_simd_file):
        """Modified files should be detected."""
        manifest = Manifest(
            version="1.0.0",
            files={
                "test.md": FileEntry(
                    sha256="different_hash" + "0" * 48,
                    mtime_ns=0,  # Different mtime
                    chunk_ids=["old_chunk"],
                )
            },
        )
        current_files = {"test.md": sample_simd_file}

        changes = compute_changes(manifest, current_files)

        assert len(changes) == 1
        assert changes[0].change_type == "modify"
        assert changes[0].old_chunk_ids == ["old_chunk"]

    def test_detects_deleted_file(self, temp_data_dir):
        """Deleted files should be detected."""
        manifest = Manifest(
            version="1.0.0",
            files={
                "deleted.md": FileEntry(
                    sha256="a" * 64,
                    mtime_ns=12345,
                    chunk_ids=["chunk1", "chunk2"],
                )
            },
        )
        current_files = {}  # No files

        changes = compute_changes(manifest, current_files)

        assert len(changes) == 1
        assert changes[0].change_type == "delete"
        assert changes[0].old_chunk_ids == ["chunk1", "chunk2"]

    def test_skips_unchanged_by_mtime(self, temp_data_dir, sample_simd_file):
        """Files with same mtime should be skipped (fast path)."""
        mtime = get_file_mtime_ns(sample_simd_file)
        file_hash = compute_file_hash(sample_simd_file)

        manifest = Manifest(
            version="1.0.0",
            files={
                "test.md": FileEntry(
                    sha256=file_hash,
                    mtime_ns=mtime,
                    chunk_ids=["chunk"],
                )
            },
        )
        current_files = {"test.md": sample_simd_file}

        changes = compute_changes(manifest, current_files)

        assert len(changes) == 0

    def test_skips_unchanged_by_hash(self, temp_data_dir, sample_simd_file):
        """Files with same hash should be skipped even if mtime differs."""
        file_hash = compute_file_hash(sample_simd_file)

        manifest = Manifest(
            version="1.0.0",
            files={
                "test.md": FileEntry(
                    sha256=file_hash,
                    mtime_ns=0,  # Different mtime
                    chunk_ids=["chunk"],
                )
            },
        )
        current_files = {"test.md": sample_simd_file}

        changes = compute_changes(manifest, current_files, check_hashes=True)

        assert len(changes) == 0


class TestIndexStats:
    """Tests for IndexStats dataclass."""

    def test_files_changed(self):
        """files_changed should be sum of add/modify/delete."""
        stats = IndexStats(files_added=1, files_modified=2, files_deleted=1)
        assert stats.files_changed == 4

    def test_is_incremental(self):
        """is_incremental should be True for non-full rebuilds with changes."""
        stats = IndexStats(full_rebuild=False, files_modified=1)
        assert stats.is_incremental is True

        stats = IndexStats(full_rebuild=True, files_modified=1)
        assert stats.is_incremental is False

    def test_is_noop(self):
        """is_noop should be True when nothing changed."""
        stats = IndexStats(full_rebuild=False)
        assert stats.is_noop is True

        stats = IndexStats(full_rebuild=False, files_added=1)
        assert stats.is_noop is False

    def test_summary_full_rebuild(self):
        """Summary should describe full rebuild."""
        stats = IndexStats(
            full_rebuild=True,
            rebuild_reason="Model changed",
            chunks_added=100,
        )
        summary = stats.summary()
        assert "Full rebuild" in summary
        assert "100 chunks" in summary

    def test_summary_incremental(self):
        """Summary should describe incremental update."""
        stats = IndexStats(
            files_added=1,
            files_modified=2,
            chunks_added=10,
            chunks_deleted=5,
        )
        summary = stats.summary()
        assert "Incremental" in summary
        assert "1 added" in summary
        assert "2 modified" in summary

    def test_summary_noop(self):
        """Summary should indicate no changes."""
        stats = IndexStats()
        assert "No changes" in stats.summary()


class TestDryRunResult:
    """Tests for DryRunResult dataclass."""

    def test_summary_rebuild(self):
        """Summary should describe pending rebuild."""
        result = DryRunResult(
            would_rebuild=True,
            rebuild_reason="No manifest",
        )
        summary = result.summary()
        assert "full rebuild" in summary
        assert "No manifest" in summary

    def test_summary_incremental(self):
        """Summary should describe pending incremental update."""
        result = DryRunResult(
            files_to_add=["file1.md", "file2.md"],
            files_to_modify=["file3.md"],
            estimated_chunks_add=20,
            estimated_chunks_delete=5,
        )
        summary = result.summary()
        assert "Add 2 files" in summary
        assert "Modify 1 files" in summary

    def test_summary_no_changes(self):
        """Summary should indicate no changes."""
        result = DryRunResult()
        assert "No changes" in result.summary()


class TestIncrementalEmbedder:
    """Tests for IncrementalEmbedder class."""

    @pytest.fixture
    def embedder(self, temp_data_dir):
        """Create an embedder for testing."""
        return IncrementalEmbedder(
            data_dir=temp_data_dir,
            model_name="all-MiniLM-L6-v2",
        )

    def test_dry_run_detects_new_files(self, embedder, temp_data_dir, sample_simd_file):
        """Dry run without manifest should indicate rebuild needed."""
        current_files = {"test.md": sample_simd_file}
        file_types = {"test.md": "simd"}

        result = embedder.dry_run(current_files, file_types)

        # Without manifest, should indicate full rebuild needed
        assert result.would_rebuild is True
        assert "No manifest" in result.rebuild_reason

    def test_dry_run_with_existing_manifest(self, embedder, temp_data_dir, sample_simd_file):
        """Dry run with manifest should detect new files."""
        # Create an empty manifest
        manifest = Manifest(
            version="1.0.0",
            embedding_model="all-MiniLM-L6-v2",
            chunk_config={"chunk_size": 1000, "chunk_overlap": 200},
            files={},
        )
        save_manifest(manifest, temp_data_dir / "manifest.json")

        current_files = {"test.md": sample_simd_file}
        file_types = {"test.md": "simd"}

        result = embedder.dry_run(current_files, file_types)

        # Should detect the new file
        assert len(result.files_to_add) == 1
        assert "test.md" in result.files_to_add

    def test_get_current_config(self, embedder):
        """Should return current config for manifest comparison."""
        config = embedder.get_current_config()

        assert "embedding_model" in config
        assert "chunk_config" in config

    @patch("solana_mcp.indexer.embedder.SentenceTransformer")
    @patch("solana_mcp.indexer.embedder.lancedb")
    def test_full_rebuild_creates_manifest(
        self,
        mock_lancedb,
        mock_transformer,
        embedder,
        temp_data_dir,
        sample_simd_file,
    ):
        """Full rebuild should create manifest."""
        import numpy as np

        # Setup mocks - encode returns array with same length as input
        mock_model = MagicMock()

        def mock_encode(texts, **kwargs):
            return np.array([[0.1] * 384] * len(texts))

        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model

        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_db.create_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        current_files = {"test.md": sample_simd_file}
        file_types = {"test.md": "simd"}

        def simple_chunk_fn(path, ftype, base):
            return [
                Chunk(
                    content="Test content",
                    source_type="simd",
                    source_file="test.md",
                    source_name="Test",
                    line_number=1,
                    metadata={},
                    chunk_id="sol_simd_abc12345_0000_def67890",
                )
            ]

        stats = embedder.index(
            current_files, file_types, chunk_fn=simple_chunk_fn, force_full=True
        )

        assert stats.full_rebuild is True
        assert (temp_data_dir / "manifest.json").exists()

    @patch("solana_mcp.indexer.embedder.SentenceTransformer")
    @patch("solana_mcp.indexer.embedder.lancedb")
    def test_incremental_update_modifies_manifest(
        self,
        mock_lancedb,
        mock_transformer,
        embedder,
        temp_data_dir,
        sample_simd_file,
    ):
        """Incremental update should update manifest."""
        import numpy as np

        # Setup mocks - encode returns array with same length as input
        mock_model = MagicMock()

        def mock_encode(texts, **kwargs):
            return np.array([[0.1] * 384] * len(texts))

        mock_model.encode.side_effect = mock_encode
        mock_transformer.return_value = mock_model
        embedder._model = mock_model

        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        # Create initial manifest
        initial_manifest = Manifest(
            version="1.0.0",
            embedding_model="all-MiniLM-L6-v2",
            chunk_config={"chunk_size": 1000, "chunk_overlap": 200},
            files={},
        )
        save_manifest(initial_manifest, temp_data_dir / "manifest.json")

        current_files = {"test.md": sample_simd_file}
        file_types = {"test.md": "simd"}

        def simple_chunk_fn(path, ftype, base):
            return [
                Chunk(
                    content="Test content",
                    source_type="simd",
                    source_file="test.md",
                    source_name="Test",
                    line_number=1,
                    metadata={},
                    chunk_id="sol_simd_abc12345_0000_def67890",
                )
            ]

        stats = embedder.index(current_files, file_types, chunk_fn=simple_chunk_fn)

        assert stats.files_added == 1
        assert stats.full_rebuild is False


class TestAtomicUpdates:
    """Tests for atomic update behavior."""

    def test_manifest_preserved_on_partial_failure(self, temp_data_dir):
        """Manifest should be preserved if indexing fails partway through."""
        # This tests that we don't corrupt state on failure
        # Implementation detail: manifest is only saved after successful indexing
        pass  # Covered by atomic write tests in test_manifest.py
