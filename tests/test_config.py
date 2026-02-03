"""Tests for configuration functionality."""


import pytest

from solana_mcp.config import (
    ChunkingConfig,
    Config,
    ConfigError,
    EmbeddingConfig,
    get_model_info,
    load_config,
    save_config,
)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = EmbeddingConfig()
        assert config.model == "all-MiniLM-L6-v2"
        assert config.batch_size == 32

    def test_model_info(self):
        """Should return model info from registry."""
        config = EmbeddingConfig(model="all-MiniLM-L6-v2")
        assert config.model_info["dimensions"] == 384
        assert config.model_info["type"] == "local"

    def test_dimensions(self):
        """Should return correct dimensions for model."""
        config = EmbeddingConfig(model="all-mpnet-base-v2")
        assert config.dimensions == 768

    def test_requires_api_key(self):
        """Should correctly identify API models."""
        local_config = EmbeddingConfig(model="all-MiniLM-L6-v2")
        assert not local_config.requires_api_key

        api_config = EmbeddingConfig(model="voyage:voyage-code-3")
        assert api_config.requires_api_key

    def test_validate_unknown_model(self):
        """Should warn but not fail for unknown model."""
        config = EmbeddingConfig(model="unknown-model")
        config.validate()  # Should not raise

    def test_validate_invalid_batch_size(self):
        """Should reject invalid batch size."""
        config = EmbeddingConfig(batch_size=0)
        with pytest.raises(ConfigError, match="batch_size"):
            config.validate()


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = ChunkingConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200

    def test_to_dict(self):
        """Should serialize to dict."""
        config = ChunkingConfig(chunk_size=500, chunk_overlap=100)
        d = config.to_dict()
        assert d["chunk_size"] == 500
        assert d["chunk_overlap"] == 100

    def test_validate_chunk_size_too_small(self):
        """Should reject chunk size < 100."""
        config = ChunkingConfig(chunk_size=50)
        with pytest.raises(ConfigError, match="chunk_size"):
            config.validate()

    def test_validate_chunk_size_too_large(self):
        """Should reject chunk size > 10000."""
        config = ChunkingConfig(chunk_size=20000)
        with pytest.raises(ConfigError, match="chunk_size"):
            config.validate()

    def test_validate_overlap_negative(self):
        """Should reject negative overlap."""
        config = ChunkingConfig(chunk_overlap=-10)
        with pytest.raises(ConfigError, match="chunk_overlap"):
            config.validate()

    def test_validate_overlap_too_large(self):
        """Should reject overlap >= chunk_size."""
        config = ChunkingConfig(chunk_size=500, chunk_overlap=500)
        with pytest.raises(ConfigError, match="chunk_overlap"):
            config.validate()


class TestConfig:
    """Tests for main Config class."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = Config()
        assert config.embedding.model == "all-MiniLM-L6-v2"
        assert config.chunking.chunk_size == 1000

    def test_validate(self):
        """Should validate all nested configs."""
        config = Config()
        config.validate()  # Should not raise


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_defaults(self, temp_data_dir):
        """Should load defaults when no config file."""
        config = load_config(data_dir=temp_data_dir)
        assert config.embedding.model == "all-MiniLM-L6-v2"

    def test_load_from_file(self, temp_data_dir):
        """Should load from YAML file."""
        config_path = temp_data_dir / "config.yaml"
        config_path.write_text("""
embedding:
  model: all-mpnet-base-v2
  batch_size: 64
chunking:
  chunk_size: 800
  chunk_overlap: 150
""")
        config = load_config(config_path=config_path)
        assert config.embedding.model == "all-mpnet-base-v2"
        assert config.embedding.batch_size == 64
        assert config.chunking.chunk_size == 800

    def test_env_override_model(self, temp_data_dir, monkeypatch):
        """Environment variables should override config file."""
        monkeypatch.setenv("SOLANA_MCP_EMBEDDING_MODEL", "codesage/codesage-large")
        config = load_config(data_dir=temp_data_dir)
        assert config.embedding.model == "codesage/codesage-large"

    def test_env_override_batch_size(self, temp_data_dir, monkeypatch):
        """Environment variables should override batch size."""
        monkeypatch.setenv("SOLANA_MCP_BATCH_SIZE", "128")
        config = load_config(data_dir=temp_data_dir)
        assert config.embedding.batch_size == 128


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_and_load_roundtrip(self, temp_data_dir):
        """Config should survive save/load roundtrip."""
        config = Config(
            embedding=EmbeddingConfig(model="all-mpnet-base-v2", batch_size=48),
            chunking=ChunkingConfig(chunk_size=750, chunk_overlap=125),
        )
        config_path = temp_data_dir / "config.yaml"

        save_config(config, config_path)
        loaded = load_config(config_path=config_path)

        assert loaded.embedding.model == "all-mpnet-base-v2"
        assert loaded.embedding.batch_size == 48
        assert loaded.chunking.chunk_size == 750
        assert loaded.chunking.chunk_overlap == 125


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_specific_model(self):
        """Should return info for specific model."""
        info = get_model_info("all-MiniLM-L6-v2")
        assert "384" in info
        assert "local" in info

    def test_all_models(self):
        """Should list all models when no model specified."""
        info = get_model_info()
        assert "all-MiniLM-L6-v2" in info
        assert "all-mpnet-base-v2" in info
        assert "codesage/codesage-large" in info
        assert "voyage:voyage-code-3" in info

    def test_unknown_model(self):
        """Should indicate unknown model."""
        info = get_model_info("nonexistent-model")
        assert "Unknown" in info
