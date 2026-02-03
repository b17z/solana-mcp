# solana-mcp

RAG-powered MCP server for Solana runtime, SIMDs, and validator client source code.

[![PyPI version](https://badge.fury.io/py/sol-mcp.svg)](https://badge.fury.io/py/sol-mcp)
[![CI](https://github.com/b17z/solana-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/b17z/solana-mcp/actions/workflows/ci.yml)

## What It Does

Indexes and searches across:
- **Agave** - Reference Rust validator (Anza)
- **Jito-Agave** - MEV-enabled fork (~70% of mainnet stake)
- **Jito Programs** - On-chain tip distribution and MEV programs
- **Firedancer** - Jump's C implementation (~22% with Frankendancer)
- **SIMDs** - Solana Improvement Documents
- **Alpenglow** - Future consensus protocol (not yet live)

## Installation

```bash
# From PyPI
pip install sol-mcp

# From source
pip install -e .

# With Voyage API embeddings (best quality)
pip install -e ".[voyage]"
```

## Quick Start

```bash
# Build the index (downloads repos + creates embeddings)
solana-mcp build

# Search
solana-mcp search "stake delegation"

# Check status
solana-mcp status
```

## Features

### Incremental Indexing

The v0.2.0 release introduces **incremental indexing** - only re-embeds changed files instead of rebuilding the entire index. This reduces update time from minutes to seconds.

```bash
# Update repos and incrementally re-index (fast!)
solana-mcp update

# Incremental index (default behavior)
solana-mcp index

# Preview what would change without indexing
solana-mcp index --dry-run

# Force full rebuild
solana-mcp index --full
```

**How it works:**
1. Tracks file hashes and modification times in a manifest
2. Detects which files changed since last index
3. Only re-embeds the changed content
4. Updates LanceDB incrementally (add/delete operations)

### Configurable Embedding Models

Choose from multiple embedding models based on your quality/speed tradeoff:

```bash
# List available models
solana-mcp models

# Use a specific model
solana-mcp index --model codesage/codesage-large
```

| Model | Dims | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| `all-MiniLM-L6-v2` | 384 | Fair | Fast | Default, good for quick searches |
| `all-mpnet-base-v2` | 768 | Good | Medium | Better quality |
| `codesage/codesage-large` | 1024 | Good | Medium | Code-specialized |
| `voyage:voyage-code-3` | 1024 | Excellent | API | Best quality, requires API key |

Configure in `~/.solana-mcp/config.yaml`:

```yaml
embedding:
  model: "codesage/codesage-large"
  batch_size: 32

chunking:
  chunk_size: 1000
  chunk_overlap: 200
```

### Expert Guidance

Curated knowledge beyond what's in the code:

```bash
# Via MCP tool
sol_expert_guidance("staking")
sol_expert_guidance("jito")
sol_expert_guidance("alpenglow")
```

Topics include: `staking`, `voting`, `slashing`, `towerbft`, `consensus`, `alpenglow`, `poh`, `accounts`, `svm`, `turbine`, `leader_schedule`, `epochs`, `mev`, `jito`, `bundles`, `tips`

## CLI Commands

```bash
# Full build pipeline
solana-mcp build                      # Download + compile + index
solana-mcp build --full               # Force full rebuild

# Individual steps
solana-mcp download                   # Clone agave, jito, firedancer, SIMDs, alpenglow
solana-mcp compile                    # Parse Rust/C code into JSON
solana-mcp index                      # Build vector embeddings
solana-mcp index --dry-run            # Preview changes
solana-mcp index --full               # Force full rebuild
solana-mcp index --model MODEL        # Use specific embedding model

# Update (git pull + incremental index)
solana-mcp update
solana-mcp update --full              # Update + force rebuild

# Search
solana-mcp search "stake delegation"
solana-mcp search "tower bft" --type rust
solana-mcp search "leader schedule" --limit 10

# Lookup
solana-mcp constant LAMPORTS_PER_SOL
solana-mcp function process_vote

# Info
solana-mcp status                     # Index status, manifest info
solana-mcp models                     # List embedding models
```

## MCP Tools

When running as an MCP server:

| Tool | Purpose |
|------|---------|
| `sol_search` | Semantic search across all indexed content |
| `sol_search_runtime` | Runtime code only (no SIMDs) |
| `sol_search_simd` | SIMDs specifically |
| `sol_grep_constant` | Fast constant lookup |
| `sol_analyze_function` | Get function source code |
| `sol_get_current_version` | Current mainnet version (v2.1) |
| `sol_list_versions` | Version history with features |
| `sol_get_consensus_status` | TowerBFT (current) vs Alpenglow (future) |
| `sol_list_feature_gates` | Feature gate activations |
| `sol_list_clients` | Validator client implementations |
| `sol_get_client` | Details on specific client |
| `sol_get_client_diversity` | Stake distribution across clients |
| `sol_expert_guidance` | Curated guidance on topics |

## Validator Clients

| Client | Language | Stake | Notes |
|--------|----------|-------|-------|
| Jito-Agave | Rust | ~70% | MEV-enabled fork |
| Frankendancer | C+Rust | ~22% | Firedancer networking + Agave runtime |
| Agave | Rust | ~8% | Reference implementation (Anza) |
| Firedancer | C | - | Full independent implementation (Jump) |

## Project Structure

```
src/solana_mcp/
├── server.py               # MCP server (FastMCP)
├── cli.py                  # CLI commands
├── config.py               # Configuration management
├── versions.py             # Version/client/consensus tracking
├── indexer/
│   ├── downloader.py       # Git clone with sparse checkout
│   ├── compiler.py         # Rust + C parsing (tree-sitter)
│   ├── chunker.py          # Code/markdown chunking + chunk IDs
│   ├── embedder.py         # Embeddings + LanceDB + incremental
│   └── manifest.py         # File tracking for incremental updates
└── expert/
    └── guidance.py         # Curated expert knowledge
```

## Data Location

```
~/.solana-mcp/
├── config.yaml             # Configuration (optional)
├── manifest.json           # Index state tracking
├── agave/                  # Reference client source
├── jito-solana/            # MEV fork source
├── jito-programs/          # On-chain MEV programs
├── firedancer/             # Jump's C implementation
├── solana-improvement-documents/
├── alpenglow/              # Future consensus
├── compiled/               # Extracted JSON
└── lancedb/                # Vector index
```

## Expert Guidance Topics

The `sol_expert_guidance` tool provides curated knowledge on:

**Staking & Consensus:**
- `staking` - Delegation, warmup/cooldown, rewards
- `voting` - Vote accounts, TowerBFT voting
- `slashing` - Current lack of slashing, future plans
- `towerbft` - Current consensus mechanism
- `consensus` - PoH + TowerBFT relationship
- `alpenglow` - Future consensus (~150ms finality)
- `poh` - Proof of History (ordering, not consensus)

**Runtime & Architecture:**
- `accounts` - Account model, rent, ownership
- `svm` - Solana Virtual Machine
- `turbine` - Block propagation
- `leader_schedule` - Slot assignment
- `epochs` - 432,000 slots, ~2-3 days

**MEV (Jito):**
- `mev` - MEV on Solana overview
- `jito` - Jito infrastructure and architecture
- `bundles` - Atomic transaction bundles
- `tips` - Tip distribution to validators/stakers

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=solana_mcp

# Lint
ruff check src/
```

## Running as MCP Server

```bash
# Start the server
sol-mcp

# Or with uvicorn for development
uvicorn solana_mcp.server:mcp --reload
```

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "solana": {
      "command": "sol-mcp"
    }
  }
}
```

## Documentation

- [Incremental Indexing](docs/INCREMENTAL_INDEXING.md) - How the incremental update system works
- [CLAUDE.md](CLAUDE.md) - Quick reference for Claude Code

## Differences from Official Solana MCP

The official [mcp.solana.com](https://mcp.solana.com) is documentation-focused.

This implementation:
- Indexes **source code** from multiple validator clients
- Parses **Rust and C** with tree-sitter
- Tracks **client diversity** and stake distribution
- Provides **incremental indexing** for fast updates

## License

MIT
