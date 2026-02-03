"""Pytest configuration and fixtures for solana-mcp tests."""

from pathlib import Path

import pytest


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create an isolated data directory for testing."""
    data_dir = tmp_path / ".solana-mcp"
    data_dir.mkdir(parents=True)
    return data_dir


@pytest.fixture
def sample_rust_file(temp_data_dir: Path) -> Path:
    """Create a sample Rust file for testing."""
    rust_content = '''//! Sample Rust module for testing.

/// The number of lamports per SOL.
pub const LAMPORTS_PER_SOL: u64 = 1_000_000_000;

/// Maximum transaction size in bytes.
pub const MAX_TX_SIZE: usize = 1232;

/// Process a transaction.
///
/// # Arguments
/// * `tx` - The transaction to process
///
/// # Returns
/// Result indicating success or failure
pub fn process_transaction(tx: &Transaction) -> Result<(), Error> {
    // Validate transaction
    validate_tx(tx)?;

    // Execute
    execute_tx(tx)?;

    Ok(())
}

/// Stake account data structure.
#[derive(Debug, Clone)]
pub struct StakeAccount {
    /// The stake authority
    pub authority: Pubkey,
    /// Amount staked in lamports
    pub stake: u64,
    /// Activation epoch
    pub activation_epoch: u64,
}

impl StakeAccount {
    /// Create a new stake account.
    pub fn new(authority: Pubkey, stake: u64) -> Self {
        Self {
            authority,
            stake,
            activation_epoch: 0,
        }
    }
}
'''
    rust_dir = temp_data_dir / "agave" / "runtime"
    rust_dir.mkdir(parents=True)
    rust_file = rust_dir / "stake.rs"
    rust_file.write_text(rust_content)
    return rust_file


@pytest.fixture
def sample_simd_file(temp_data_dir: Path) -> Path:
    """Create a sample SIMD file for testing."""
    simd_content = '''---
simd: '0326'
title: Alpenglow Consensus Protocol
authors: Anatoly Yakovenko
type: Standard
status: Draft
---

# Summary

Alpenglow is a proposed upgrade to Solana's consensus mechanism.

# Motivation

The current TowerBFT consensus has limitations:
- High confirmation latency
- Complex fork choice rules

# Specification

## Overview

Alpenglow introduces:
- Single-slot finality
- Simplified fork choice

## Voset Algorithm

The voset (voting set) algorithm determines validators for each slot.

```rust
fn compute_voset(epoch: u64, slot: u64) -> Vec<Pubkey> {
    // Implementation details
}
```

# Security Considerations

- Requires 2/3 honest validators
- Resistant to long-range attacks
'''
    simd_dir = temp_data_dir / "solana-improvement-documents" / "proposals"
    simd_dir.mkdir(parents=True)
    simd_file = simd_dir / "0326-alpenglow.md"
    simd_file.write_text(simd_content)
    return simd_file


@pytest.fixture
def sample_manifest():
    """Create a sample manifest dictionary for testing."""
    return {
        "version": "1.0.0",
        "updated_at": "2026-02-03T12:00:00+00:00",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_config": {"chunk_size": 1000, "chunk_overlap": 200},
        "files": {
            "runtime/stake.rs": {
                "sha256": "a" * 64,
                "mtime_ns": 1707000000000000000,
                "chunk_ids": ["sol_rust_abc12345_0001_def67890"],
            }
        },
        "repo_versions": {
            "agave": "abc123def456",
        },
    }


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    import numpy as np

    def embed(texts, **kwargs):
        return np.array([[0.1] * 384] * len(texts))

    return embed
