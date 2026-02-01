# solana-mcp

RAG-powered MCP server for Solana runtime, SIMDs, and validator client source code.

## What It Does

Indexes and searches across:
- **Agave** - Reference Rust validator (Anza)
- **Jito-Agave** - MEV-enabled fork (~70% of mainnet stake)
- **Jito Programs** - On-chain tip distribution and MEV programs
- **Firedancer** - Jump's C implementation (~22% with Frankendancer)
- **SIMDs** - Solana Improvement Documents
- **Alpenglow** - Future consensus protocol (not yet live)

## Quick Start

```bash
# Install
pip install -e .

# Full build (download repos + compile + index)
solana-mcp build

# Or step by step
solana-mcp download   # Clone repositories
solana-mcp compile    # Extract code to JSON
solana-mcp index      # Build vector embeddings
```

## CLI Commands

```bash
# Build pipeline
solana-mcp build      # Full pipeline
solana-mcp download   # Clone repos (agave, jito-solana, firedancer, SIMDs, alpenglow)
solana-mcp compile    # Parse Rust/C code into JSON
solana-mcp index      # Build LanceDB vector index

# Search
solana-mcp search "stake delegation"
solana-mcp search "tower bft" --type rust
solana-mcp search "leader schedule" --limit 10

# Lookup
solana-mcp constant LAMPORTS_PER_SOL
solana-mcp function process_vote

# Status
solana-mcp status     # Show what's downloaded/compiled/indexed
```

## MCP Tools

When running as an MCP server, these tools are available:

| Tool | Purpose |
|------|---------|
| `sol_search` | Semantic search across all indexed content |
| `sol_search_simd` | Search SIMDs specifically |
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

Solana validator clients indexed:

| Client | Language | Notes |
|--------|----------|-------|
| Jito-Agave | Rust | MEV-enabled fork |
| Frankendancer | C+Rust | Firedancer networking + Agave runtime |
| Agave | Rust | Reference implementation (Anza) |
| Firedancer | C | Full independent implementation (Jump) |

For current client diversity statistics, see [validators.app](https://www.validators.app/cluster-stats).

## Project Structure

```
src/solana_mcp/
├── server.py           # MCP server with all tools
├── cli.py              # CLI commands
├── versions.py         # Version/client/consensus tracking
├── indexer/
│   ├── downloader.py   # Git clone with sparse checkout
│   ├── compiler.py     # Rust + C parsing (tree-sitter)
│   ├── chunker.py      # Code/markdown chunking
│   └── embedder.py     # Embeddings + LanceDB
└── expert/
    └── guidance.py     # Curated expert knowledge
```

## Data Location

```
~/.solana-mcp/
├── agave/              # Reference client source
├── jito-solana/        # MEV fork source
├── jito-programs/      # On-chain MEV programs
├── firedancer/         # Jump's C implementation
├── solana-improvement-documents/
├── alpenglow/          # Future consensus
├── compiled/           # Extracted JSON
│   ├── agave/
│   ├── jito-solana/
│   ├── jito-programs/
│   └── firedancer/
└── lancedb/            # Vector index
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

## Solana vs Ethereum

| Aspect | Ethereum | Solana |
|--------|----------|--------|
| Spec format | Markdown + Python | Rust implementation |
| Slashing | Yes (3 penalties) | No (indirect penalties only) |
| Finality | ~13 min | ~12.8s → 150ms (Alpenglow) |
| Consensus | Casper FFG | TowerBFT → Alpenglow |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/
```

## Differences from Official Solana MCP

The official [mcp.solana.com](https://mcp.solana.com) is documentation-focused.

This implementation:
- Indexes actual **source code** from multiple clients
- Parses **Rust and C** with tree-sitter
- Tracks **client diversity** and stake distribution
- Includes **expert guidance** from protocol research
- Provides **constant/function lookup** with full source

## License

MIT
