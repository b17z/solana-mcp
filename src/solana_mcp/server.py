"""MCP server for Solana runtime and SIMDs.

Exposes tools for searching and analyzing Solana protocol code.
"""


import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import ValidationError

from .expert.guidance import get_expert_guidance, list_guidance_topics
from .indexer.compiler import lookup_constant, lookup_function
from .indexer.downloader import DEFAULT_DATA_DIR, list_downloaded_repos
from .indexer.embedder import DEPS_AVAILABLE, get_index_stats, search
from .models import (
    ClientLookupInput,
    ConstantLookupInput,
    FunctionLookupInput,
    GuidanceInput,
    SearchInput,
)
from .versions import (
    get_client,
    get_client_diversity,
    get_consensus_status,
    get_current_version,
    list_clients,
    list_feature_gates,
    list_versions,
)

logger = logging.getLogger(__name__)

# Initialize server
server = Server("solana-mcp")

# Data directory
DATA_DIR = DEFAULT_DATA_DIR


def _safe_path(base: Path, user_path: str) -> Path:
    """
    Resolve a user-provided path safely within a base directory.
    Prevents path traversal attacks (e.g., ../../../etc/passwd).

    Args:
        base: The base directory that all paths must stay within
        user_path: User-provided relative path

    Returns:
        Resolved path within base directory

    Raises:
        ValueError: If path would escape base directory
    """
    # Resolve the full path
    resolved = (base / user_path).resolve()

    # Ensure it's still within the base directory
    try:
        resolved.relative_to(base.resolve())
    except ValueError:
        logger.warning("Path traversal attempt blocked: %s", user_path)
        raise ValueError(f"Invalid path: {user_path}")

    return resolved


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="sol_search",
            description=(
                "Search across Solana runtime code and SIMDs. "
                "Use for questions about how Solana works, finding functions, "
                "understanding stake/vote/consensus mechanics."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'stake warmup period')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="sol_search_runtime",
            description=(
                "Search only Rust runtime code (no SIMDs). "
                "Use when you specifically want implementation details."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="sol_search_simd",
            description=(
                "Search only SIMDs (Solana Improvement Documents). "
                "Use for protocol proposals, specifications, and governance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="sol_grep_constant",
            description=(
                "Look up a specific constant by name. "
                "Fast exact-match lookup for constants like LAMPORTS_PER_SOL, "
                "MAX_LOCKOUT_HISTORY, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Constant name (e.g., 'LAMPORTS_PER_SOL')",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="sol_analyze_function",
            description=(
                "Get the full Rust implementation of a function. "
                "Use to understand exactly how something works in the runtime."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Function name (e.g., 'process_stake_instruction')",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="sol_expert_guidance",
            description=(
                "Get curated expert guidance on Solana topics. "
                "Covers: staking, voting, slashing, alpenglow, poh, accounts, svm, turbine, towerbft. "
                "Includes gotchas and nuances not obvious from the code."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic (e.g., 'staking', 'towerbft', 'alpenglow')",
                    },
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="sol_list_guidance_topics",
            description="List all available expert guidance topics.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="sol_get_status",
            description="Get status of the Solana MCP index.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="sol_get_current_version",
            description=(
                "Get the current Solana mainnet version. "
                "Returns version number, release date, and key features."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="sol_list_versions",
            description=(
                "List all major Solana versions with their features and release dates. "
                "Includes historical versions and planned future versions."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="sol_get_consensus_status",
            description=(
                "Get current consensus mechanism status. "
                "Shows TowerBFT (current) vs Alpenglow (future) status, finality times, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="sol_list_feature_gates",
            description=(
                "List Solana feature gates (protocol feature flags). "
                "Shows which features are activated and when."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "activated_only": {
                        "type": "boolean",
                        "description": "Only show activated features",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="sol_list_clients",
            description=(
                "List Solana validator client implementations. "
                "Includes Agave, Jito-Agave, Firedancer, Frankendancer, Sig."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "production_only": {
                        "type": "boolean",
                        "description": "Only show production-ready clients",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="sol_get_client",
            description=(
                "Get details about a specific Solana client. "
                "E.g., 'firedancer', 'jito', 'agave', 'frankendancer', 'sig'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Client name",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="sol_get_client_diversity",
            description=(
                "Get Solana client diversity statistics. "
                "Shows stake distribution across clients and diversity concerns."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "sol_search":
        try:
            validated = SearchInput(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                source_type=None,
            )
        except ValidationError as e:
            return [TextContent(type="text", text=f"Validation error: {e}")]
        return await handle_search(
            validated.query,
            validated.limit,
            source_type=validated.source_type,
        )

    elif name == "sol_search_runtime":
        try:
            validated = SearchInput(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                source_type="rust",
            )
        except ValidationError as e:
            return [TextContent(type="text", text=f"Validation error: {e}")]
        return await handle_search(
            validated.query,
            validated.limit,
            source_type=validated.source_type,
        )

    elif name == "sol_search_simd":
        try:
            validated = SearchInput(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
                source_type="simd",
            )
        except ValidationError as e:
            return [TextContent(type="text", text=f"Validation error: {e}")]
        return await handle_search(
            validated.query,
            validated.limit,
            source_type=validated.source_type,
        )

    elif name == "sol_grep_constant":
        try:
            validated = ConstantLookupInput(name=arguments["name"])
        except ValidationError as e:
            return [TextContent(type="text", text=f"Validation error: {e}")]
        return await handle_grep_constant(validated.name)

    elif name == "sol_analyze_function":
        try:
            validated = FunctionLookupInput(name=arguments["name"])
        except ValidationError as e:
            return [TextContent(type="text", text=f"Validation error: {e}")]
        return await handle_analyze_function(validated.name)

    elif name == "sol_expert_guidance":
        try:
            validated = GuidanceInput(topic=arguments["topic"])
        except ValidationError as e:
            return [TextContent(type="text", text=f"Validation error: {e}")]
        return await handle_expert_guidance(validated.topic)

    elif name == "sol_list_guidance_topics":
        topics = list_guidance_topics()
        return [TextContent(
            type="text",
            text="Available guidance topics:\n" + "\n".join(f"  - {t}" for t in topics),
        )]

    elif name == "sol_get_status":
        return await handle_status()

    elif name == "sol_get_current_version":
        return await handle_current_version()

    elif name == "sol_list_versions":
        return await handle_list_versions()

    elif name == "sol_get_consensus_status":
        return await handle_consensus_status()

    elif name == "sol_list_feature_gates":
        return await handle_feature_gates(arguments.get("activated_only", False))

    elif name == "sol_list_clients":
        return await handle_list_clients(arguments.get("production_only", False))

    elif name == "sol_get_client":
        try:
            validated = ClientLookupInput(name=arguments["name"])
        except ValidationError as e:
            return [TextContent(type="text", text=f"Validation error: {e}")]
        return await handle_get_client(validated.name)

    elif name == "sol_get_client_diversity":
        return await handle_client_diversity()

    else:
        return [TextContent(
            type="text",
            text=f"Unknown tool: {name}",
        )]


async def handle_search(
    query: str,
    limit: int,
    source_type: str | None,
) -> list[TextContent]:
    """Handle search requests."""
    if not DEPS_AVAILABLE:
        return [TextContent(
            type="text",
            text="Search dependencies not installed. Run: pip install lancedb sentence-transformers",
        )]

    results = search(
        query,
        data_dir=DATA_DIR,
        limit=limit,
        source_type=source_type,
    )

    if not results:
        return [TextContent(
            type="text",
            text=f"No results found for: {query}",
        )]

    # Format results
    output = [f"Search results for: {query}\n"]

    for i, result in enumerate(results):
        score = 1 - result["score"]  # Convert distance to similarity
        output.append(
            f"\n{i + 1}. [{result['source_type']}] {result['source_name']} "
            f"(score: {score:.0%})"
        )
        output.append(f"   File: {result['source_file']}:{result['line_number']}")

        # Include content
        content = result["content"]
        if len(content) > 500:
            content = content[:500] + "\n... (truncated)"
        output.append(f"\n```\n{content}\n```")

    return [TextContent(type="text", text="\n".join(output))]


async def handle_grep_constant(name: str) -> list[TextContent]:
    """Handle constant lookup."""
    compiled_dir = DATA_DIR / "compiled"

    # Search all compiled directories
    for subdir in compiled_dir.glob("**"):
        if (subdir / "index.json").exists():
            result = lookup_constant(name, subdir)
            if result:
                output = [
                    f"# {result.name}",
                    f"Value: {result.value}",
                ]
                if result.type_annotation:
                    output.append(f"Type: {result.type_annotation}")
                output.append(f"File: {result.file_path}:{result.line_number}")
                if result.doc_comment:
                    output.append(f"\nDoc: {result.doc_comment}")

                return [TextContent(type="text", text="\n".join(output))]

    return [TextContent(
        type="text",
        text=f"Constant '{name}' not found in index.",
    )]


async def handle_analyze_function(name: str) -> list[TextContent]:
    """Handle function lookup."""
    compiled_dir = DATA_DIR / "compiled"

    for subdir in compiled_dir.glob("**"):
        if (subdir / "index.json").exists():
            result = lookup_function(name, subdir)
            if result:
                output = [
                    f"# {result.signature}",
                    f"File: {result.file_path}:{result.line_number}",
                ]
                if result.doc_comment:
                    output.append(f"\n/// {result.doc_comment}\n")
                output.append(f"\n```rust\n{result.body}\n```")

                return [TextContent(type="text", text="\n".join(output))]

    return [TextContent(
        type="text",
        text=f"Function '{name}' not found in index.",
    )]


async def handle_expert_guidance(topic: str) -> list[TextContent]:
    """Handle expert guidance lookup."""
    guidance = get_expert_guidance(topic)

    if not guidance:
        topics = list_guidance_topics()
        return [TextContent(
            type="text",
            text=f"No guidance found for '{topic}'.\n\nAvailable topics:\n"
            + "\n".join(f"  - {t}" for t in topics),
        )]

    output = [
        f"# {guidance.topic}",
        f"\n{guidance.summary}\n",
        "## Key Points",
    ]
    for point in guidance.key_points:
        output.append(f"- {point}")

    output.append("\n## Gotchas")
    for gotcha in guidance.gotchas:
        output.append(f"- {gotcha}")

    output.append("\n## References")
    for ref in guidance.references:
        output.append(f"- {ref}")

    return [TextContent(type="text", text="\n".join(output))]


async def handle_status() -> list[TextContent]:
    """Handle status check."""
    output = ["# Solana MCP Status\n"]

    # Repos
    output.append("## Repositories")
    repos = list_downloaded_repos(DATA_DIR)
    for name, info in repos.items():
        status = "✓" if info["exists"] else "✗"
        version = info.get("version", "not downloaded")
        output.append(f"  {status} {name}: {version}")

    # Index
    output.append("\n## Index")
    if DEPS_AVAILABLE:
        stats = get_index_stats(DATA_DIR)
        if stats and "error" not in stats:
            output.append(f"  Total chunks: {stats['total_chunks']}")
            for source_type, count in stats.get("by_source_type", {}).items():
                output.append(f"    {source_type}: {count}")
        else:
            output.append("  Not indexed")
    else:
        output.append("  Dependencies not installed")

    return [TextContent(type="text", text="\n".join(output))]


async def handle_current_version() -> list[TextContent]:
    """Handle current version request."""
    version = get_current_version()

    output = [
        f"# Solana {version.version}",
        f"Release: {version.release_date}",
        f"\n{version.description}",
        "\n## Key Features",
    ]
    for feature in version.key_features:
        output.append(f"- {feature}")

    if version.breaking_changes:
        output.append("\n## Breaking Changes")
        for change in version.breaking_changes:
            output.append(f"- {change}")

    return [TextContent(type="text", text="\n".join(output))]


async def handle_list_versions() -> list[TextContent]:
    """Handle list versions request."""
    versions = list_versions()

    output = ["# Solana Version History\n"]

    for v in versions:
        current = " (CURRENT)" if v.current else ""
        future = " (NOT YET LIVE)" if "expected" in v.release_date.lower() else ""
        output.append(f"## {v.version}{current}{future}")
        output.append(f"Release: {v.release_date}")
        output.append(f"{v.description}\n")

    return [TextContent(type="text", text="\n".join(output))]


async def handle_consensus_status() -> list[TextContent]:
    """Handle consensus status request."""
    status = get_consensus_status()

    output = [
        "# Solana Consensus Status",
        "",
        "## Current (Mainnet)",
        f"**{status['current']}**",
        f"{status['current_description']}",
        f"- Finality: {status['finality']}",
        f"- Optimistic confirmation: {status['optimistic_confirmation']}",
        "",
        "## Proof of History",
        f"Status: {status['poh_status']}",
        "",
        "## Future: Alpenglow",
        f"Status: {status['future_status']}",
        f"- Expected finality: {status['future_finality']}",
        "",
        "**Note:** TowerBFT is the CURRENT consensus. Alpenglow is NOT YET LIVE.",
    ]

    return [TextContent(type="text", text="\n".join(output))]


async def handle_feature_gates(activated_only: bool) -> list[TextContent]:
    """Handle feature gates request."""
    features = list_feature_gates(activated_only)

    output = ["# Solana Feature Gates\n"]

    for f in features:
        status = "✓ Activated" if f.activated_slot else "○ Pending"
        output.append(f"## {f.name}")
        output.append(f"Status: {status}")
        output.append(f"Introduced: {f.version_introduced}")
        if f.activated_slot:
            output.append(f"Activated: slot {f.activated_slot} ({f.activated_date})")
        output.append(f"Description: {f.description}")
        output.append(f"Feature ID: `{f.feature_id}`\n")

    return [TextContent(type="text", text="\n".join(output))]


async def handle_list_clients(production_only: bool) -> list[TextContent]:
    """Handle list clients request."""
    clients = list_clients(production_only)

    output = ["# Solana Validator Clients\n"]

    for c in clients:
        stake_str = f" (~{c.stake_percentage}% stake)" if c.stake_percentage else ""
        output.append(f"## {c.name}{stake_str}")
        output.append(f"**{c.organization}** | {c.language} | {c.mainnet_status}")
        output.append(f"\n{c.description}")
        output.append(f"\nRepo: {c.repo}\n")

    return [TextContent(type="text", text="\n".join(output))]


async def handle_get_client(name: str) -> list[TextContent]:
    """Handle get client request."""
    client = get_client(name)

    if not client:
        clients = list_clients()
        names = [c.name for c in clients]
        return [TextContent(
            type="text",
            text=f"Client '{name}' not found.\n\nAvailable clients: {', '.join(names)}",
        )]

    stake_str = f" (~{client.stake_percentage}% stake)" if client.stake_percentage else ""
    output = [
        f"# {client.name}{stake_str}",
        f"Organization: {client.organization}",
        f"Language: {client.language}",
        f"Status: {client.mainnet_status}",
        f"Repo: {client.repo}",
        f"\n{client.description}",
        "\n## Key Differentiators",
    ]
    for diff in client.key_differentiators:
        output.append(f"- {diff}")

    output.append("\n## Notes")
    for note in client.notes:
        output.append(f"- {note}")

    return [TextContent(type="text", text="\n".join(output))]


async def handle_client_diversity() -> list[TextContent]:
    """Handle client diversity request."""
    diversity = get_client_diversity()

    output = [
        "# Solana Client Diversity",
        f"\nTotal clients: {diversity['total_clients']}",
        f"Production clients: {diversity['production_clients']}",
        "\n## Stake Distribution (Oct 2025)",
    ]

    for client, info in diversity["client_breakdown"].items():
        output.append(f"- **{client}**: {info}")

    output.append("\n## Diversity Notes")
    for note in diversity["diversity_notes"]:
        output.append(f"- {note}")

    return [TextContent(type="text", text="\n".join(output))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run():
    """Entry point for the MCP server."""
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    run()
