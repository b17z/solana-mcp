# Incremental Indexing System

The incremental indexing system enables fast updates to the solana-mcp vector index by only re-embedding changed content. This significantly reduces indexing time from minutes to seconds for typical updates.

## Overview

Instead of rebuilding the entire index on every update, the system:

1. **Tracks** what files have been indexed using a manifest
2. **Detects** which files have changed since last index
3. **Computes** precise deltas (chunks to add/delete)
4. **Applies** incremental updates to LanceDB

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI Commands                           │
│  solana-mcp index [--full] [--dry-run] [--model ...]       │
│  solana-mcp update                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  IncrementalEmbedder                        │
│  - Load manifest                                            │
│  - Check if full rebuild needed                             │
│  - Compute file changes                                     │
│  - Apply delta updates                                      │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│    Manifest      │ │   Chunker    │ │    LanceDB       │
│  - File hashes   │ │  - Split     │ │  - Add chunks    │
│  - Chunk IDs     │ │  - Generate  │ │  - Delete chunks │
│  - Config        │ │    chunk IDs │ │  - Vector search │
└──────────────────┘ └──────────────┘ └──────────────────┘
```

## Key Components

### 1. Manifest (`indexer/manifest.py`)

The manifest tracks the state of indexed files:

```json
{
  "version": "1.0.0",
  "updated_at": "2026-02-03T14:30:00Z",
  "embedding_model": "all-MiniLM-L6-v2",
  "chunk_config": { "chunk_size": 1000, "chunk_overlap": 200 },
  "files": {
    "solana/runtime/src/bank.rs": {
      "sha256": "abc123...",
      "mtime_ns": 1707000000000000000,
      "chunk_ids": ["sol_rust_a1b2c3d4_0001", "sol_rust_a1b2c3d4_0002"]
    }
  },
  "repo_versions": {
    "agave": "abc123def456",
    "jito-solana": "789xyz...",
    "SIMDs": "def456..."
  }
}
```

**Key features:**
- **Atomic writes**: Uses write-to-temp-then-rename to prevent corruption
- **Validation**: Strict validation of paths, hashes, and chunk IDs
- **Security**: Rejects path traversal and injection attempts

### 2. Chunk IDs (`indexer/chunker.py`)

Each chunk gets a unique, deterministic ID:

```
sol_rust_a1b2c3d4_0001_ef567890
│   │    │         │    │
│   │    │         │    └── Content hash (8 chars)
│   │    │         └── Chunk index (4 digits, zero-padded)
│   │    └── Path hash (8 chars)
│   └── Source type (rust, simd, docs)
└── Project prefix
```

**Why content hash?** If the content changes but the position stays the same, the chunk ID changes. This enables precise tracking of what actually changed.

### 3. Configuration (`config.py`)

Embedding model and chunking parameters are configurable:

```yaml
# ~/.solana-mcp/config.yaml
embedding:
  model: "all-MiniLM-L6-v2"    # Or codesage/codesage-large for better quality
  batch_size: 32

chunking:
  chunk_size: 1000
  chunk_overlap: 200
```

**Supported models:**

| Model | Dims | Context | Quality | Notes |
|-------|------|---------|---------|-------|
| all-MiniLM-L6-v2 | 384 | 256 | Fair | Fast fallback (default) |
| all-mpnet-base-v2 | 768 | 384 | Good | Balanced |
| codesage/codesage-large | 1024 | 1024 | Good | Code-specialized |
| voyage:voyage-code-3 | 1024 | 16k | Excellent | API, requires VOYAGE_API_KEY |

### 4. IncrementalEmbedder (`indexer/embedder.py`)

The main class that orchestrates incremental indexing:

```python
embedder = IncrementalEmbedder(data_dir=Path("~/.solana-mcp"))

# Check what would change
dry_run = embedder.dry_run(current_files, file_types)
print(dry_run.summary())

# Perform indexing
stats = embedder.index(current_files, file_types)
print(stats.summary())
```

## Change Detection Algorithm

```
1. Load manifest from disk
2. Check if full rebuild needed:
   - No manifest exists → REBUILD
   - Embedding model changed → REBUILD
   - Chunk config changed → REBUILD
3. For each source file:
   a. Not in manifest → ADD (new file)
   b. mtime unchanged → SKIP (fast path)
   c. mtime changed, hash unchanged → SKIP (touch only)
   d. hash changed → MODIFY (delete old chunks, add new)
4. Manifest files not in current → DELETE
5. Apply delta to LanceDB:
   - Delete removed chunks
   - Add new chunks
6. Save updated manifest
```

### Fast Path vs Slow Path

- **Fast path**: If mtime hasn't changed, skip the file entirely (no hash computation)
- **Slow path**: If mtime changed, compute hash to check if content actually changed

This optimization means that most files are skipped in O(1) time during typical updates.

## CLI Usage

### Basic Indexing

```bash
# Incremental index (default)
solana-mcp index

# Force full rebuild
solana-mcp index --full

# Preview what would change
solana-mcp index --dry-run

# Use specific embedding model
solana-mcp index --model codesage/codesage-large
```

### Update Command

The `update` command combines git pull with incremental indexing:

```bash
solana-mcp update
```

This will:
1. Run `git pull --ff-only` on agave, jito-solana, and SIMDs
2. Detect changed files
3. Incrementally update the index

### Status Command

See detailed index status:

```bash
solana-mcp status
```

Output includes:
- Configuration (embedding model, chunk size)
- Manifest info (tracked files, total chunks)
- Repository versions

### Models Command

List available embedding models:

```bash
solana-mcp models
```

## Security Considerations

The system includes several security measures:

### Path Validation
- Rejects path traversal (`../../../etc/passwd`)
- Rejects absolute paths (`/etc/passwd`)
- Rejects null bytes and special characters

### Chunk ID Sanitization
- Only allows alphanumeric, underscore, and hyphen
- Length limit (200 chars)
- Prevents SQL injection in LanceDB queries

### Resource Limits
- Manifest file size limit (10MB)
- Max chunks per file (10,000)

### Safe YAML Loading
- Uses `yaml.safe_load()` to prevent code execution
- Validates all config values

## File Structure

```
~/.solana-mcp/
├── config.yaml           # Configuration (optional)
├── manifest.json         # Index state tracking
├── agave/                # Cloned repo (Solana runtime)
├── jito-solana/          # Cloned repo (Jito fork)
├── SIMDs/                # Cloned repo (Solana Improvement Documents)
└── lancedb/              # Vector database
    └── solana_docs/      # Main table
```

## Indexed Content

The system indexes content from multiple sources:

| Source | Content | Source Type |
|--------|---------|-------------|
| agave | Runtime Rust code (stake, vote, consensus) | `rust` |
| jito-solana | Jito-specific modifications | `rust` |
| SIMDs | Solana Improvement Documents | `simd` |

## Performance Characteristics

### Full Rebuild
- Time: ~2-5 minutes (depending on hardware and model)
- Embeds all chunks from scratch
- Creates fresh LanceDB table

### Incremental Update
- Time: Seconds to ~1 minute
- Only embeds changed chunks
- Preserves existing vectors

### Typical Scenarios

| Scenario | Time |
|----------|------|
| First build | 2-5 min |
| No changes | <1 sec |
| 1 file changed | 5-15 sec |
| Git pull with 5 files | 15-45 sec |
| Model change | 2-5 min (full rebuild) |

## Troubleshooting

### "Manifest corrupted" Error

The manifest file was damaged. The system will:
1. Create a backup at `manifest.json.corrupted`
2. Perform a full rebuild

### "Embedding model changed" Full Rebuild

If you change the embedding model, a full rebuild is required because the vector dimensions may differ.

### Index Appears Out of Date

Try running with `--dry-run` to see what would change:

```bash
solana-mcp index --dry-run
```

If no changes detected but you expect some:
1. Check if files actually changed (not just touched)
2. Force rebuild: `solana-mcp index --full`

### LanceDB Table Not Found

If the table was deleted but manifest exists:

```bash
solana-mcp index --full
```

## Implementation Details

### LanceDB Operations

For incremental updates, we use:
- `table.add(records)` to add new chunks
- `table.delete(f'chunk_id = "{id}"')` to remove old chunks

This is faster than drop+recreate for small changes.

### Atomic Manifest Writes

Manifest is saved atomically to prevent corruption:

```python
# Write to temp file
temp_path = manifest_path.with_suffix(".json.tmp")
with open(temp_path, "w") as f:
    json.dump(data, f)
    f.flush()
    os.fsync(f.fileno())

# Atomic rename
temp_path.replace(manifest_path)
```

### Thread Safety

The current implementation is not thread-safe. Running multiple indexing operations simultaneously may corrupt the manifest.

## Future Improvements

- [ ] Parallel embedding with batching
- [ ] Resumable indexing (checkpoint partial progress)
- [ ] Additional client code indexing (Firedancer, Sig)
- [ ] Index compaction/optimization
- [ ] Version migration for manifest format changes
