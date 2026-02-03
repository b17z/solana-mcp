"""Security tests for solana-mcp.

Tests for path validation, input sanitization, and other security measures.
"""

import pytest

from solana_mcp.indexer.manifest import (
    ManifestValidationError,
    _sanitize_chunk_id,
    _validate_path,
    generate_chunk_id,
)


class TestPathValidation:
    """Tests for path validation security."""

    def test_rejects_path_traversal(self):
        """Should reject paths with '..'."""
        with pytest.raises(ManifestValidationError, match="traversal"):
            _validate_path("../../../etc/passwd")

    def test_rejects_path_traversal_middle(self):
        """Should reject '..' anywhere in path."""
        with pytest.raises(ManifestValidationError, match="traversal"):
            _validate_path("foo/../bar/baz")

    def test_rejects_absolute_unix_paths(self):
        """Should reject absolute Unix paths."""
        with pytest.raises(ManifestValidationError, match="Absolute"):
            _validate_path("/etc/passwd")

    def test_rejects_absolute_windows_paths(self):
        """Should reject absolute Windows paths."""
        with pytest.raises(ManifestValidationError, match="Absolute"):
            _validate_path("C:\\Windows\\System32")

    def test_rejects_null_bytes(self):
        """Should reject paths with null bytes."""
        with pytest.raises(ManifestValidationError, match="Null"):
            _validate_path("file\x00.txt")

    def test_rejects_newlines(self):
        """Should reject paths with newlines."""
        with pytest.raises(ManifestValidationError, match="Invalid"):
            _validate_path("file\nname.txt")

    def test_rejects_carriage_returns(self):
        """Should reject paths with carriage returns."""
        with pytest.raises(ManifestValidationError, match="Invalid"):
            _validate_path("file\rname.txt")

    def test_rejects_tabs(self):
        """Should reject paths with tabs."""
        with pytest.raises(ManifestValidationError, match="Invalid"):
            _validate_path("file\tname.txt")

    def test_accepts_valid_paths(self):
        """Should accept valid relative paths."""
        _validate_path("runtime/stake.rs")
        _validate_path("src/lib.rs")
        _validate_path("proposals/0326-alpenglow.md")
        _validate_path("a/b/c/d/e/file.rs")

    def test_accepts_dots_in_names(self):
        """Should accept dots in file names."""
        _validate_path("file.test.rs")
        _validate_path(".hidden")


class TestChunkIdSanitization:
    """Tests for chunk ID sanitization."""

    def test_rejects_special_characters(self):
        """Should reject chunk IDs with special characters."""
        with pytest.raises(ManifestValidationError, match="Invalid"):
            _sanitize_chunk_id("chunk;DROP TABLE")

    def test_rejects_spaces(self):
        """Should reject chunk IDs with spaces."""
        with pytest.raises(ManifestValidationError, match="Invalid"):
            _sanitize_chunk_id("chunk with spaces")

    def test_rejects_quotes(self):
        """Should reject chunk IDs with quotes."""
        with pytest.raises(ManifestValidationError, match="Invalid"):
            _sanitize_chunk_id("chunk'quote")

    def test_rejects_too_long(self):
        """Should reject chunk IDs that are too long."""
        with pytest.raises(ManifestValidationError, match="too long"):
            _sanitize_chunk_id("a" * 201)

    def test_accepts_valid_ids(self):
        """Should accept valid chunk IDs."""
        _sanitize_chunk_id("sol_rust_abc12345_0001_def67890")
        _sanitize_chunk_id("sol_simd_12345678_0099_abcdef12")
        _sanitize_chunk_id("test-with-hyphens")
        _sanitize_chunk_id("test_with_underscores")

    def test_accepts_alphanumeric(self):
        """Should accept alphanumeric characters."""
        _sanitize_chunk_id("ABC123xyz456")


class TestChunkIdGenerationSecurity:
    """Tests for chunk ID generation security."""

    def test_rejects_malicious_source_file(self):
        """Should reject malicious source file paths."""
        with pytest.raises(ManifestValidationError):
            generate_chunk_id("sol", "rust", "../../../etc/passwd", 0, "content")

    def test_generates_safe_ids(self):
        """Generated IDs should be safe for queries."""
        chunk_id = generate_chunk_id("sol", "rust", "runtime/stake.rs", 0, "content")
        # Should be sanitizable
        _sanitize_chunk_id(chunk_id)

    def test_handles_unicode_content(self):
        """Should handle unicode content without issues."""
        chunk_id = generate_chunk_id("sol", "rust", "test.rs", 0, "content with emoji")
        assert chunk_id
        _sanitize_chunk_id(chunk_id)


class TestInputValidation:
    """Tests for general input validation."""

    def test_rejects_oversized_manifest(self, temp_data_dir):
        """Should reject oversized manifest files."""
        from solana_mcp.indexer.manifest import (
            MAX_MANIFEST_SIZE,
            ManifestValidationError,
            load_manifest,
        )

        manifest_path = temp_data_dir / "manifest.json"
        # Create a file larger than MAX_MANIFEST_SIZE
        manifest_path.write_text("x" * (MAX_MANIFEST_SIZE + 1))

        with pytest.raises(ManifestValidationError, match="too large"):
            load_manifest(manifest_path)

    def test_handles_malformed_yaml_config(self, temp_data_dir):
        """Should handle malformed YAML config gracefully."""
        from solana_mcp.config import load_config

        config_path = temp_data_dir / "config.yaml"
        config_path.write_text("invalid: yaml: content: [")

        # Should fall back to defaults, not crash
        config = load_config(config_path=config_path)
        assert config.embedding.model == "all-MiniLM-L6-v2"

    def test_validates_model_name_format(self):
        """Should warn about unknown model names."""
        from solana_mcp.config import EmbeddingConfig

        config = EmbeddingConfig(model="MALICIOUS; DROP TABLE;")
        # Should warn but not crash
        config.validate()

    def test_rejects_too_many_chunks_per_file(self):
        """Should reject manifest with too many chunks per file."""
        from solana_mcp.indexer.manifest import (
            MAX_CHUNKS_PER_FILE,
            Manifest,
            ManifestValidationError,
        )

        data = {
            "version": "1.0.0",
            "files": {
                "test.rs": {
                    "sha256": "a" * 64,
                    "mtime_ns": 0,
                    "chunk_ids": ["chunk"] * (MAX_CHUNKS_PER_FILE + 1),
                }
            },
        }

        with pytest.raises(ManifestValidationError, match="Too many chunks"):
            Manifest.from_dict(data)


class TestResourceLimits:
    """Tests for resource limit enforcement."""

    def test_limits_config_file_size(self, temp_data_dir):
        """Should reject oversized config files."""
        from solana_mcp.config import ConfigError, _load_config_file

        config_path = temp_data_dir / "config.yaml"
        config_path.write_text("x" * (1024 * 1024 + 1))  # > 1MB

        with pytest.raises(ConfigError, match="too large"):
            _load_config_file(config_path)
