# solana-mcp

RAG-powered MCP server for Solana runtime, SIMDs, and protocol documentation.

## Quick Reference

```bash
# Full build (download + compile + index)
solana-mcp build

# Individual steps
solana-mcp download    # Clone agave, SIMDs, alpenglow
solana-mcp compile     # Extract Rust functions/types to JSON
solana-mcp index       # Build vector embeddings in LanceDB

# Incremental updates (fast)
solana-mcp update      # Git pull + incremental re-index
solana-mcp index       # Incremental by default
solana-mcp index --full    # Force full rebuild
solana-mcp index --dry-run # Preview what would change

# Search
solana-mcp search "stake warmup"
solana-mcp status      # Check index status
solana-mcp models      # List available embedding models
```

## MCP Tools

| Tool | Purpose |
|------|---------|
| `sol_search` | Unified search across runtime + SIMDs |
| `sol_search_runtime` | Runtime code only (no SIMDs) |
| `sol_search_simd` | SIMDs only |
| `sol_grep_constant` | Fast constant lookup |
| `sol_analyze_function` | Get Rust implementation |
| `sol_expert_guidance` | Curated interpretations |
| `sol_get_current_version` | Current mainnet version |

## Project Structure

```
src/solana_mcp/
├── server.py           # MCP server (FastMCP)
├── cli.py              # CLI commands
├── config.py           # Configuration management
├── indexer/
│   ├── downloader.py   # Git clone repos
│   ├── compiler.py     # Extract Rust to JSON
│   ├── chunker.py      # Rust/markdown chunking + chunk IDs
│   ├── embedder.py     # Embeddings + LanceDB + IncrementalEmbedder
│   └── manifest.py     # File tracking for incremental updates
└── expert/
    └── guidance.py     # Curated interpretations
```

## Data Location

```
~/.solana-mcp/
├── config.yaml         # Configuration (optional)
├── manifest.json       # Index state tracking
├── agave/              # Cloned anza-xyz/agave
├── solana-improvement-documents/  # Cloned SIMDs
├── alpenglow/          # Cloned anza-xyz/alpenglow
├── compiled/           # JSON extracts
└── lancedb/            # Vector index
```

## Configuration

Create `~/.solana-mcp/config.yaml` to customize:

```yaml
embedding:
  model: "all-MiniLM-L6-v2"  # Or codesage/codesage-large
  batch_size: 32

chunking:
  chunk_size: 1000
  chunk_overlap: 200
```

Available models: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `codesage/codesage-large`, `voyage:voyage-code-3`

## Key Differences from Ethereum MCP

1. **No formal specs** - Rust code IS the spec
2. **No slashing** - Solana doesn't slash validators (indirect penalties only)
3. **Account model** - Flat accounts vs Ethereum's state trie
4. **Alpenglow** - New consensus replacing TowerBFT (Q1 2026)

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/
```

## Incremental Indexing

See [docs/INCREMENTAL_INDEXING.md](docs/INCREMENTAL_INDEXING.md) for detailed documentation.

Key concepts:
- **Manifest**: Tracks file hashes and chunk IDs
- **Chunk IDs**: Deterministic IDs based on content hash
- **Change detection**: Fast mtime check, then hash verification
- **Delta updates**: Only re-embed changed content

## Adding Expert Guidance

Edit `src/solana_mcp/expert/guidance.py`:

```python
GUIDANCE_DB["staking"] = GuidanceEntry(
    topic="Stake Delegation",
    summary="Solana uses delegated proof-of-stake with warmup/cooldown periods.",
    key_points=["No minimum stake", "~2 epoch warmup", "~2 epoch cooldown"],
    gotchas=["No slashing - only indirect penalties", "Rewards based on vote credits"],
    references=["programs/stake/src/stake_state.rs"],
)
```

## Source Repos

- **agave**: https://github.com/anza-xyz/agave (main runtime)
- **SIMDs**: https://github.com/solana-foundation/solana-improvement-documents
- **alpenglow**: https://github.com/anza-xyz/alpenglow (new consensus)
