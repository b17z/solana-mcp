"""Download Solana source repositories for indexing.

Clones:
- anza-xyz/agave (reference validator runtime)
- solana-foundation/solana-improvement-documents (SIMDs)
- anza-xyz/alpenglow (new consensus protocol, not yet live)
- firedancer-io/firedancer (Jump's C implementation)
- jito-foundation/jito-solana (MEV-enabled Agave fork, ~70% of stake)
- jito-foundation/jito-programs (on-chain tip distribution programs)
"""

import subprocess
from pathlib import Path
from typing import Callable

# Default data directory
DEFAULT_DATA_DIR = Path.home() / ".solana-mcp"

# Repositories to clone
REPOS = {
    # Reference implementation (Anza)
    "agave": {
        "url": "https://github.com/anza-xyz/agave.git",
        "branch": "master",
        "client": True,
        "stake_pct": 8.0,  # ~8% stake on vanilla Agave
        "sparse_paths": [
            "programs/stake",
            "programs/vote",
            "programs/bpf_loader",
            "programs/system",
            "runtime",
            "svm",
            "poh",
            "turbine",
            "gossip",
            "ledger",
            "validator",
            "sdk",
            "core",  # TowerBFT consensus, replay_stage, fork choice
            "docs",
        ],
    },
    # Jito-Agave: MEV-enabled fork, dominant client (~70% stake)
    "jito-solana": {
        "url": "https://github.com/jito-foundation/jito-solana.git",
        "branch": "master",
        "client": True,
        "stake_pct": 70.0,  # ~70% of mainnet stake
        "sparse_paths": [
            "core",  # Block engine integration
            "runtime",
            "programs/stake",
            "programs/vote",
            "poh",
            "turbine",
            "gossip",
            "validator",
            "bundle",  # Jito-specific: bundle processing
            "tip-distributor",  # Jito-specific: tip distribution
            "block-engine",  # Jito-specific: MEV infrastructure
        ],
    },
    # Firedancer: Jump's C implementation (independent failure domain)
    "firedancer": {
        "url": "https://github.com/firedancer-io/firedancer.git",
        "branch": "main",
        "client": True,
        "stake_pct": 22.0,  # ~21% Frankendancer + <1% full Firedancer
        "sparse_paths": [
            "src/app",  # Validator application
            "src/ballet",  # Cryptography, hashing
            "src/disco",  # Distributed consensus
            "src/flamenco",  # Runtime, accounts
            "src/tango",  # Messaging infrastructure
            "src/waltz",  # Networking (QUIC, UDP)
            "src/choreo",  # Fork choice, TowerBFT
            "contrib",  # Build scripts, tests
        ],
    },
    # SIMDs (small repo, full clone)
    "solana-improvement-documents": {
        "url": "https://github.com/solana-foundation/solana-improvement-documents.git",
        "branch": "main",
        "client": False,
        "sparse_paths": None,  # Clone full repo (it's small)
    },
    # Alpenglow: Future consensus (not yet live)
    "alpenglow": {
        "url": "https://github.com/anza-xyz/alpenglow.git",
        "branch": "master",
        "client": False,
        "sparse_paths": None,  # Clone full repo
    },
    # ===================
    # MEV Infrastructure
    # ===================
    # Jito on-chain programs (tip distribution, etc.)
    "jito-programs": {
        "url": "https://github.com/jito-foundation/jito-programs.git",
        "branch": "master",
        "client": False,
        "sparse_paths": [
            "mev-programs",  # Core MEV programs
            "tip-distribution",  # Tip distribution program
            "tip-payment",  # Tip payment handling
        ],
    },
}


def run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a git command."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result


def clone_repo(
    name: str,
    url: str,
    dest: Path,
    branch: str = "main",
    sparse_paths: list[str] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> bool:
    """
    Clone a repository, optionally with sparse checkout.

    Args:
        name: Repository name for logging
        url: Git URL to clone
        dest: Destination path
        branch: Branch to checkout
        sparse_paths: If provided, only checkout these paths (sparse checkout)
        progress_callback: Optional callback for progress updates

    Returns:
        True if successful, False otherwise
    """
    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    if dest.exists():
        log(f"  {name}: Already exists, pulling latest...")
        result = run_git(["pull", "--ff-only"], cwd=dest)
        if result.returncode != 0:
            log(f"  {name}: Pull failed, trying fetch + reset...")
            run_git(["fetch", "origin"], cwd=dest)
            run_git(["reset", "--hard", f"origin/{branch}"], cwd=dest)
        return True

    log(f"  {name}: Cloning from {url}...")

    if sparse_paths:
        # Sparse checkout for large repos
        dest.mkdir(parents=True, exist_ok=True)

        # Initialize repo
        run_git(["init"], cwd=dest)
        run_git(["remote", "add", "origin", url], cwd=dest)

        # Configure sparse checkout
        run_git(["config", "core.sparseCheckout", "true"], cwd=dest)

        # Write sparse-checkout file
        sparse_file = dest / ".git" / "info" / "sparse-checkout"
        sparse_file.parent.mkdir(parents=True, exist_ok=True)
        sparse_file.write_text("\n".join(sparse_paths) + "\n")

        # Fetch and checkout
        log(f"  {name}: Fetching (sparse checkout: {len(sparse_paths)} paths)...")
        result = run_git(["fetch", "--depth=1", "origin", branch], cwd=dest)
        if result.returncode != 0:
            log(f"  {name}: Fetch failed: {result.stderr}")
            return False

        result = run_git(["checkout", branch], cwd=dest)
        if result.returncode != 0:
            log(f"  {name}: Checkout failed: {result.stderr}")
            return False
    else:
        # Full clone for small repos
        result = run_git(["clone", "--depth=1", "--branch", branch, url, str(dest)])
        if result.returncode != 0:
            log(f"  {name}: Clone failed: {result.stderr}")
            return False

    log(f"  {name}: Done")
    return True


def download_repos(
    data_dir: Path | None = None,
    repos: list[str] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, bool]:
    """
    Download all configured repositories.

    Args:
        data_dir: Base directory for downloads (default: ~/.solana-mcp)
        repos: List of repo names to download (default: all)
        progress_callback: Optional callback for progress updates

    Returns:
        Dict mapping repo name to success status
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    data_dir.mkdir(parents=True, exist_ok=True)

    if repos is None:
        repos = list(REPOS.keys())

    results = {}

    for name in repos:
        if name not in REPOS:
            if progress_callback:
                progress_callback(f"  {name}: Unknown repository, skipping")
            results[name] = False
            continue

        config = REPOS[name]
        dest = data_dir / name

        success = clone_repo(
            name=name,
            url=config["url"],
            dest=dest,
            branch=config["branch"],
            sparse_paths=config.get("sparse_paths"),
            progress_callback=progress_callback,
        )
        results[name] = success

    return results


def get_repo_path(name: str, data_dir: Path | None = None) -> Path | None:
    """Get the path to a downloaded repository."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    path = data_dir / name
    if path.exists():
        return path
    return None


def get_repo_version(name: str, data_dir: Path | None = None) -> str | None:
    """Get the current commit hash of a repository."""
    path = get_repo_path(name, data_dir)
    if path is None:
        return None

    result = run_git(["rev-parse", "HEAD"], cwd=path)
    if result.returncode == 0:
        return result.stdout.strip()[:12]
    return None


def list_downloaded_repos(data_dir: Path | None = None) -> dict[str, dict]:
    """List all downloaded repositories with their status."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    status = {}
    for name in REPOS:
        path = data_dir / name
        if path.exists():
            version = get_repo_version(name, data_dir)
            status[name] = {
                "path": str(path),
                "version": version,
                "exists": True,
            }
        else:
            status[name] = {
                "path": str(path),
                "version": None,
                "exists": False,
            }

    return status


if __name__ == "__main__":
    # Test download
    print("Downloading Solana repositories...")
    results = download_repos(progress_callback=print)
    print("\nResults:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
