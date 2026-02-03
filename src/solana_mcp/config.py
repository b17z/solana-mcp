"""Configuration management for solana-mcp.

Supports loading configuration from:
1. Default values
2. Config file (~/.solana-mcp/config.yaml)
3. Environment variables (for secrets like API keys)

Configuration precedence: env vars > config file > defaults
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .logging import get_logger

logger = get_logger("config")

# Default values
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Supported embedding models with their properties
EMBEDDING_MODELS = {
    # Local models (sentence-transformers)
    "all-MiniLM-L6-v2": {
        "dimensions": 384,
        "max_tokens": 256,
        "type": "local",
        "description": "Fast, lightweight fallback model",
    },
    "all-mpnet-base-v2": {
        "dimensions": 768,
        "max_tokens": 384,
        "type": "local",
        "description": "Better quality, moderate speed",
    },
    "codesage/codesage-large": {
        "dimensions": 1024,
        "max_tokens": 1024,
        "type": "local",
        "description": "Code-specialized, recommended for Rust source",
    },
    # API models (require API keys)
    "voyage:voyage-code-3": {
        "dimensions": 1024,
        "max_tokens": 16000,
        "type": "api",
        "env_var": "VOYAGE_API_KEY",
        "description": "Best quality, requires API key ($0.06/1M tokens)",
    },
}


class ConfigError(Exception):
    """Configuration error."""

    pass


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE

    @property
    def model_info(self) -> dict[str, Any]:
        """Get model info from EMBEDDING_MODELS."""
        return EMBEDDING_MODELS.get(self.model, {})

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions for the model."""
        return self.model_info.get("dimensions", 384)

    @property
    def requires_api_key(self) -> bool:
        """Check if model requires an API key."""
        return self.model_info.get("type") == "api"

    @property
    def api_key_env_var(self) -> str | None:
        """Get the environment variable name for the API key."""
        return self.model_info.get("env_var")

    def validate(self) -> None:
        """Validate configuration.

        Raises ConfigError if validation fails.
        """
        if self.model not in EMBEDDING_MODELS:
            logger.warning(
                "Unknown embedding model: %s. Using default settings.", self.model
            )

        if self.requires_api_key:
            env_var = self.api_key_env_var
            if env_var and not os.environ.get(env_var):
                raise ConfigError(
                    f"Model {self.model} requires {env_var} environment variable"
                )

        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}")

        if self.batch_size > 256:
            logger.warning(
                "Large batch_size (%d) may cause memory issues", self.batch_size
            )


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    def validate(self) -> None:
        """Validate configuration.

        Raises ConfigError if validation fails.
        """
        if self.chunk_size < 100:
            raise ConfigError(f"chunk_size must be >= 100, got {self.chunk_size}")

        if self.chunk_size > 10000:
            raise ConfigError(f"chunk_size must be <= 10000, got {self.chunk_size}")

        if self.chunk_overlap < 0:
            raise ConfigError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")

        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )

    def to_dict(self) -> dict[str, int]:
        """Convert to dict for manifest storage."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


@dataclass
class Config:
    """Main configuration container."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    def validate(self) -> None:
        """Validate all configuration."""
        self.embedding.validate()
        self.chunking.validate()


def load_config(config_path: Path | None = None, data_dir: Path | None = None) -> Config:
    """
    Load configuration from file and environment.

    Args:
        config_path: Explicit path to config file (optional)
        data_dir: Data directory to look for config.yaml (optional)

    Returns:
        Validated Config object
    """
    config = Config()

    # Determine config file path
    if config_path is None and data_dir is not None:
        config_path = data_dir / "config.yaml"

    # Load from file if exists
    if config_path and config_path.exists():
        try:
            config = _load_config_file(config_path)
            logger.debug("Loaded config from %s", config_path)
        except Exception as e:
            logger.warning("Failed to load config from %s: %s", config_path, e)
            config = Config()

    # Override with environment variables
    config = _apply_env_overrides(config)

    # Validate
    config.validate()

    return config


def _load_config_file(config_path: Path) -> Config:
    """Load configuration from YAML file."""
    # Security: limit file size
    max_size = 1024 * 1024  # 1MB
    if config_path.stat().st_size > max_size:
        raise ConfigError(f"Config file too large: {config_path.stat().st_size} > {max_size}")

    with open(config_path, encoding="utf-8") as f:
        # Use safe_load to prevent code execution
        data = yaml.safe_load(f)

    if data is None:
        return Config()

    if not isinstance(data, dict):
        raise ConfigError("Config file must be a YAML mapping")

    # Validate keys
    allowed_keys = {"embedding", "chunking"}
    unknown_keys = set(data.keys()) - allowed_keys
    if unknown_keys:
        logger.warning("Unknown config keys ignored: %s", unknown_keys)

    # Parse embedding config
    embedding_data = data.get("embedding", {})
    if not isinstance(embedding_data, dict):
        raise ConfigError("'embedding' must be a mapping")

    embedding = EmbeddingConfig(
        model=str(embedding_data.get("model", DEFAULT_EMBEDDING_MODEL)),
        batch_size=int(embedding_data.get("batch_size", DEFAULT_BATCH_SIZE)),
    )

    # Parse chunking config
    chunking_data = data.get("chunking", {})
    if not isinstance(chunking_data, dict):
        raise ConfigError("'chunking' must be a mapping")

    chunking = ChunkingConfig(
        chunk_size=int(chunking_data.get("chunk_size", DEFAULT_CHUNK_SIZE)),
        chunk_overlap=int(chunking_data.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
    )

    return Config(embedding=embedding, chunking=chunking)


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides to config."""
    # SOLANA_MCP_EMBEDDING_MODEL overrides config file
    env_model = os.environ.get("SOLANA_MCP_EMBEDDING_MODEL")
    if env_model:
        config.embedding.model = env_model
        logger.debug("Using embedding model from env: %s", env_model)

    # SOLANA_MCP_BATCH_SIZE
    env_batch = os.environ.get("SOLANA_MCP_BATCH_SIZE")
    if env_batch:
        try:
            config.embedding.batch_size = int(env_batch)
        except ValueError:
            logger.warning("Invalid SOLANA_MCP_BATCH_SIZE: %s", env_batch)

    return config


def save_config(config: Config, config_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save
        config_path: Path to write config file
    """
    data = {
        "embedding": {
            "model": config.embedding.model,
            "batch_size": config.embedding.batch_size,
        },
        "chunking": {
            "chunk_size": config.chunking.chunk_size,
            "chunk_overlap": config.chunking.chunk_overlap,
        },
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved config to %s", config_path)


def get_model_info(model_name: str | None = None) -> str:
    """Get human-readable info about embedding models.

    Args:
        model_name: Specific model to get info for (None = all models)

    Returns:
        Formatted string with model information
    """
    if model_name:
        if model_name not in EMBEDDING_MODELS:
            return f"Unknown model: {model_name}"
        info = EMBEDDING_MODELS[model_name]
        return (
            f"{model_name}:\n"
            f"  Dimensions: {info['dimensions']}\n"
            f"  Max tokens: {info['max_tokens']}\n"
            f"  Type: {info['type']}\n"
            f"  Description: {info['description']}"
        )

    lines = ["Available embedding models:\n"]
    for name, info in EMBEDDING_MODELS.items():
        marker = " (default)" if name == DEFAULT_EMBEDDING_MODEL else ""
        api_note = " [requires API key]" if info["type"] == "api" else ""
        lines.append(
            f"  {name}{marker}{api_note}\n"
            f"    {info['dimensions']} dims, {info['max_tokens']} tokens\n"
            f"    {info['description']}\n"
        )

    return "\n".join(lines)
