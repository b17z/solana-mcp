"""CLI for solana-mcp.

Commands:
- build: Full pipeline (download + compile + index)
- download: Clone repositories
- compile: Extract Rust to JSON
- index: Build vector embeddings (incremental by default)
- update: Git pull + incremental index
- search: Search the index
- status: Check index status
- models: List available embedding models
"""

import sys
from pathlib import Path

import click

from .config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS, get_model_info, load_config
from .indexer.chunker import chunk_all_simds, chunk_content
from .indexer.compiler import (
    compile_c,
    compile_rust,
    load_compiled_constants,
    load_compiled_items,
    lookup_constant,
    lookup_function,
)
from .indexer.downloader import (
    DEFAULT_DATA_DIR,
    REPOS,
    download_repos,
    list_downloaded_repos,
)
from .indexer.embedder import (
    DEPS_AVAILABLE,
    IncrementalEmbedder,
    build_index,
    get_index_stats,
    search,
)
from .indexer.manifest import load_manifest


@click.group()
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_DATA_DIR,
    help="Data directory (default: ~/.solana-mcp)",
)
@click.pass_context
def main(ctx, data_dir: Path):
    """Solana MCP - RAG-powered search for Solana runtime and SIMDs."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir


@main.command()
@click.option("--full", is_flag=True, help="Force full rebuild")
@click.pass_context
def build(ctx, full: bool):
    """Full build pipeline: download, compile, and index."""
    data_dir = ctx.obj["data_dir"]

    click.echo("=" * 60)
    click.echo("SOLANA MCP BUILD")
    click.echo("=" * 60)

    # Step 1: Download
    click.echo("\n[1/3] Downloading repositories...")
    results = download_repos(data_dir, progress_callback=click.echo)

    failed = [name for name, success in results.items() if not success]
    if failed:
        click.echo(f"Warning: Failed to download: {', '.join(failed)}")

    # Step 2: Compile
    click.echo("\n[2/3] Compiling Rust source...")
    ctx.invoke(compile)

    # Step 3: Index
    click.echo("\n[3/3] Building vector index...")
    ctx.invoke(index, full=full)

    click.echo("\n" + "=" * 60)
    click.echo("BUILD COMPLETE")
    click.echo("=" * 60)


@main.command()
@click.option("--repo", multiple=True, help="Specific repos to download")
@click.pass_context
def download(ctx, repo):
    """Download Solana repositories."""
    data_dir = ctx.obj["data_dir"]

    repos = list(repo) if repo else None

    click.echo("Downloading repositories...")
    results = download_repos(data_dir, repos=repos, progress_callback=click.echo)

    click.echo("\nResults:")
    for name, success in results.items():
        status = click.style("✓", fg="green") if success else click.style("✗", fg="red")
        click.echo(f"  {status} {name}")


@main.command()
@click.pass_context
def compile(ctx):
    """Compile source code to JSON extracts."""
    data_dir = ctx.obj["data_dir"]
    compiled_dir = data_dir / "compiled"

    total_stats = {
        "files_processed": 0,
        "items_extracted": 0,
        "constants_extracted": 0,
        "functions": 0,
        "structs": 0,
        "enums": 0,
    }

    # Compile agave (Rust - reference implementation)
    agave_dir = data_dir / "agave"
    if agave_dir.exists():
        click.echo("Compiling agave (Rust)...")

        # Compile key directories
        for subdir in ["programs", "runtime", "svm", "poh", "turbine", "core", "gossip", "ledger"]:
            source = agave_dir / subdir
            if source.exists():
                output = compiled_dir / "agave" / subdir
                click.echo(f"  {subdir}...")
                stats = compile_rust(source, output)

                for key in total_stats:
                    total_stats[key] += stats.get(key, 0)
    else:
        click.echo("Warning: agave not found. Run 'download' first.")

    # Compile jito-solana (Rust - MEV fork, ~70% of stake)
    jito_dir = data_dir / "jito-solana"
    if jito_dir.exists():
        click.echo("Compiling jito-solana (Rust)...")

        # Compile Jito-specific directories + core
        jito_subdirs = [
            "core", "runtime", "poh", "turbine", "gossip", "validator",
            "bundle", "tip-distributor", "block-engine",  # Jito-specific
        ]
        for subdir in jito_subdirs:
            source = jito_dir / subdir
            if source.exists():
                output = compiled_dir / "jito-solana" / subdir
                click.echo(f"  {subdir}...")
                stats = compile_rust(source, output)

                for key in total_stats:
                    total_stats[key] += stats.get(key, 0)

    # Compile firedancer (C - Jump's independent implementation)
    firedancer_dir = data_dir / "firedancer"
    if firedancer_dir.exists():
        click.echo("Compiling firedancer (C)...")

        # Compile key C source directories
        fd_subdirs = [
            "src/app", "src/ballet", "src/disco", "src/flamenco",
            "src/tango", "src/waltz", "src/choreo",
        ]
        for subdir in fd_subdirs:
            source = firedancer_dir / subdir
            if source.exists():
                output = compiled_dir / "firedancer" / subdir.replace("src/", "")
                click.echo(f"  {subdir}...")
                stats = compile_c(source, output)

                for key in total_stats:
                    total_stats[key] += stats.get(key, 0)

    # Compile alpenglow (Rust - future consensus, not yet live)
    alpenglow_dir = data_dir / "alpenglow"
    if alpenglow_dir.exists():
        click.echo("Compiling alpenglow (Rust)...")
        output = compiled_dir / "alpenglow"
        stats = compile_rust(alpenglow_dir, output)

        for key in total_stats:
            total_stats[key] += stats.get(key, 0)

    # Compile jito-programs (Rust - on-chain MEV programs)
    jito_programs_dir = data_dir / "jito-programs"
    if jito_programs_dir.exists():
        click.echo("Compiling jito-programs (Rust)...")

        jito_prog_subdirs = ["mev-programs", "tip-distribution", "tip-payment"]
        for subdir in jito_prog_subdirs:
            source = jito_programs_dir / subdir
            if source.exists():
                output = compiled_dir / "jito-programs" / subdir
                click.echo(f"  {subdir}...")
                stats = compile_rust(source, output)

                for key in total_stats:
                    total_stats[key] += stats.get(key, 0)

    click.echo("\nCompilation complete:")
    click.echo(f"  Files: {total_stats['files_processed']}")
    click.echo(f"  Functions: {total_stats['functions']}")
    click.echo(f"  Structs: {total_stats['structs']}")
    click.echo(f"  Enums: {total_stats['enums']}")
    click.echo(f"  Constants: {total_stats['constants_extracted']}")


@main.command()
@click.option("--full", is_flag=True, help="Force full rebuild")
@click.option("--dry-run", is_flag=True, help="Show what would change without indexing")
@click.option("--model", "model_name", help="Embedding model to use")
@click.pass_context
def index(ctx, full: bool, dry_run: bool, model_name: str | None):
    """Build vector embeddings index (incremental by default)."""
    if not DEPS_AVAILABLE:
        click.echo("Error: Embedding dependencies not installed.")
        click.echo("Run: pip install lancedb sentence-transformers")
        sys.exit(1)

    data_dir = ctx.obj["data_dir"]
    compiled_dir = data_dir / "compiled"

    # Load config
    config = load_config(data_dir=data_dir)
    model = model_name or config.embedding.model

    # Validate model
    if model not in EMBEDDING_MODELS:
        click.echo(f"Warning: Unknown model '{model}', using default")
        model = DEFAULT_EMBEDDING_MODEL

    all_chunks = []

    # Load compiled Rust items
    for compiled_subdir in compiled_dir.glob("**"):
        if (compiled_subdir / "items.json").exists():
            click.echo(f"Loading items from {compiled_subdir.relative_to(compiled_dir)}...")
            items = load_compiled_items(compiled_subdir)
            constants = load_compiled_constants(compiled_subdir)

            # Extract repo name and path prefix from directory structure
            # e.g., compiled/agave/programs → repo_name="agave", path_prefix="programs"
            rel_path = compiled_subdir.relative_to(compiled_dir)
            rel_parts = rel_path.parts
            repo_name = rel_parts[0] if rel_parts else "agave"
            # Path prefix is everything after the repo name
            path_prefix = "/".join(rel_parts[1:]) if len(rel_parts) > 1 else ""

            chunks = chunk_content(
                items=items,
                constants=constants,
                repo_name=repo_name,
                path_prefix=path_prefix,
            )
            all_chunks.extend(chunks)
            click.echo(f"  {len(chunks)} chunks")

    # Load SIMDs
    simd_dir = data_dir / "solana-improvement-documents"
    if simd_dir.exists():
        click.echo("Chunking SIMDs...")
        simd_chunks = chunk_all_simds(simd_dir)
        all_chunks.extend(simd_chunks)
        click.echo(f"  {len(simd_chunks)} chunks")

    if not all_chunks:
        click.echo("No content to index. Run 'download' and 'compile' first.")
        return

    # Build file tracking maps
    current_files: dict[str, Path] = {}
    file_types: dict[str, str] = {}

    for chunk in all_chunks:
        if chunk.source_file not in current_files:
            # Try to find actual file path
            abs_path = data_dir / chunk.source_file
            if abs_path.exists():
                current_files[chunk.source_file] = abs_path
                file_types[chunk.source_file] = chunk.source_type

    if dry_run:
        click.echo("\n[DRY RUN] Analyzing changes...")
        embedder = IncrementalEmbedder(
            data_dir=data_dir,
            model_name=model,
        )
        result = embedder.dry_run(current_files, file_types)
        click.echo(result.summary())
        if result.files_to_add:
            click.echo(f"  Files to add: {', '.join(result.files_to_add[:5])}")
            if len(result.files_to_add) > 5:
                click.echo(f"    ... and {len(result.files_to_add) - 5} more")
        if result.files_to_modify:
            click.echo(f"  Files to modify: {', '.join(result.files_to_modify[:5])}")
        if result.files_to_delete:
            click.echo(f"  Files to delete: {', '.join(result.files_to_delete[:5])}")
        return

    click.echo(f"\nIndexing {len(all_chunks)} total chunks...")

    # Use legacy build_index for now (chunks are already prepared)
    stats = build_index(
        all_chunks,
        data_dir=data_dir,
        model_name=model,
        progress_callback=click.echo,
    )

    click.echo(f"\nIndex built: {stats['chunks_indexed']} chunks")
    click.echo(f"Database: {stats['db_path']}")


@main.command()
@click.option("--full", is_flag=True, help="Force full rebuild after update")
@click.pass_context
def update(ctx, full: bool):
    """Update repos and re-index incrementally."""
    data_dir = ctx.obj["data_dir"]

    click.echo("Updating repositories...")

    # Pull all repos
    repos = list_downloaded_repos(data_dir)
    updated = []

    for name, info in repos.items():
        if info["exists"]:
            repo_path = data_dir / name
            click.echo(f"  Pulling {name}...")
            try:
                import subprocess

                result = subprocess.run(
                    ["git", "pull", "--ff-only"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                if "Already up to date" not in result.stdout:
                    updated.append(name)
                    click.echo("    Updated")
                else:
                    click.echo("    Up to date")
            except Exception as e:
                click.echo(f"    Failed: {e}")

    if updated:
        click.echo(f"\nUpdated: {', '.join(updated)}")
        click.echo("Re-compiling and re-indexing...")
        ctx.invoke(compile)
        ctx.invoke(index, full=full)
    else:
        click.echo("\nNo updates found")


@main.command("search")
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.option("--type", "source_type", help="Filter by type (rust, simd, docs)")
@click.pass_context
def search_cmd(ctx, query: str, limit: int, source_type: str | None):
    """Search the index."""
    if not DEPS_AVAILABLE:
        click.echo("Error: Search dependencies not installed.")
        sys.exit(1)

    data_dir = ctx.obj["data_dir"]

    results = search(
        query,
        data_dir=data_dir,
        limit=limit,
        source_type=source_type,
    )

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nResults for: {query}\n")

    for i, result in enumerate(results):
        score = 1 - result["score"]  # Convert distance to similarity
        click.echo(
            f"{i + 1}. [{result['source_type']}] "
            f"{click.style(result['source_name'], bold=True)} "
            f"(score: {score:.2%})"
        )
        click.echo(f"   {result['source_file']}:{result['line_number']}")

        # Show snippet
        content = result["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        for line in content.split("\n")[:3]:
            click.echo(f"   {line}")
        click.echo()


@main.command()
@click.argument("name")
@click.pass_context
def constant(ctx, name: str):
    """Look up a constant by name."""
    data_dir = ctx.obj["data_dir"]
    compiled_dir = data_dir / "compiled"

    # Search all compiled directories
    for subdir in compiled_dir.glob("**"):
        if (subdir / "index.json").exists():
            result = lookup_constant(name, subdir)
            if result:
                click.echo(f"\n{click.style(result.name, bold=True)}")
                click.echo(f"  Value: {result.value}")
                if result.type_annotation:
                    click.echo(f"  Type: {result.type_annotation}")
                click.echo(f"  File: {result.file_path}:{result.line_number}")
                if result.doc_comment:
                    click.echo(f"  Doc: {result.doc_comment}")
                return

    click.echo(f"Constant '{name}' not found.")


@main.command()
@click.argument("name")
@click.pass_context
def function(ctx, name: str):
    """Look up a function by name."""
    data_dir = ctx.obj["data_dir"]
    compiled_dir = data_dir / "compiled"

    for subdir in compiled_dir.glob("**"):
        if (subdir / "index.json").exists():
            result = lookup_function(name, subdir)
            if result:
                click.echo(f"\n{click.style(result.signature, bold=True)}")
                click.echo(f"  File: {result.file_path}:{result.line_number}")
                if result.doc_comment:
                    click.echo(f"\n  /// {result.doc_comment}\n")
                click.echo(result.body)
                return

    click.echo(f"Function '{name}' not found.")


@main.command()
@click.pass_context
def status(ctx):
    """Check index status."""
    data_dir = ctx.obj["data_dir"]

    click.echo("SOLANA MCP STATUS")
    click.echo("=" * 40)

    # Check repos
    click.echo("\nRepositories:")
    repos = list_downloaded_repos(data_dir)
    for name, info in repos.items():
        repo_config = REPOS.get(name, {})
        is_client = repo_config.get("client", False)
        stake_pct = repo_config.get("stake_pct")

        if info["exists"]:
            status_icon = click.style("✓", fg="green")
            version = info["version"] or "unknown"
            if is_client and stake_pct:
                click.echo(f"  {status_icon} {name} ({version}) - CLIENT ~{stake_pct}% stake")
            elif is_client:
                click.echo(f"  {status_icon} {name} ({version}) - CLIENT")
            else:
                click.echo(f"  {status_icon} {name} ({version})")
        else:
            status_icon = click.style("✗", fg="red")
            if is_client and stake_pct:
                click.echo(f"  {status_icon} {name} (not downloaded) - CLIENT ~{stake_pct}% stake")
            else:
                click.echo(f"  {status_icon} {name} (not downloaded)")

    # Check compiled
    compiled_dir = data_dir / "compiled"
    click.echo("\nCompiled:")
    if compiled_dir.exists():
        for subdir in compiled_dir.glob("**"):
            items_file = subdir / "items.json"
            if items_file.exists():
                items = load_compiled_items(subdir)
                constants = load_compiled_constants(subdir)
                rel_path = subdir.relative_to(compiled_dir)
                click.echo(f"  {rel_path}: {len(items)} items, {len(constants)} constants")
    else:
        click.echo("  Not compiled yet")

    # Check manifest
    click.echo("\nManifest:")
    manifest = load_manifest(data_dir / "manifest.json")
    if manifest:
        click.echo(f"  Version: {manifest.version}")
        click.echo(f"  Updated: {manifest.updated_at}")
        click.echo(f"  Model: {manifest.embedding_model}")
        click.echo(f"  Files tracked: {len(manifest.files)}")
    else:
        click.echo("  No manifest (full build needed)")

    # Check index
    click.echo("\nIndex:")
    if DEPS_AVAILABLE:
        stats = get_index_stats(data_dir)
        if stats and "error" not in stats:
            click.echo(f"  Total chunks: {stats['total_chunks']}")
            for source_type, count in stats.get("by_source_type", {}).items():
                click.echo(f"    {source_type}: {count}")
        else:
            click.echo("  Not indexed yet")
    else:
        click.echo("  Dependencies not installed")


@main.command()
@click.argument("model", required=False)
def models(model: str | None):
    """List available embedding models."""
    click.echo(get_model_info(model))


if __name__ == "__main__":
    main()
