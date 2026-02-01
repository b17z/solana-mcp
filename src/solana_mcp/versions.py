"""Solana version and upgrade tracking.

Tracks mainnet versions, cluster upgrades, feature activations, and clients.
Similar to Ethereum's fork tracking but adapted for Solana's model.

Key differences from Ethereum:
- Solana uses semantic versioning (v1.18.x, v2.0.x) not named forks
- Features activated via feature gates, not hard forks
- Upgrades are rolling, not simultaneous network-wide
- Multiple client implementations (like Ethereum's client diversity)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolanaVersion:
    """A Solana mainnet version."""

    version: str
    release_date: str
    description: str
    key_features: list[str]
    breaking_changes: list[str]
    current: bool = False


@dataclass(frozen=True)
class FeatureGate:
    """A Solana feature gate."""

    name: str
    feature_id: str  # The pubkey of the feature
    description: str
    activated_slot: int | None  # None if not yet activated
    activated_date: str | None
    version_introduced: str


@dataclass(frozen=True)
class SolanaClient:
    """A Solana validator client implementation."""

    name: str
    organization: str
    language: str
    repo: str
    description: str
    mainnet_status: str  # "production", "beta", "testnet", "development"
    stake_percentage: float | None  # Approximate % of stake running this client
    key_differentiators: list[str]
    notes: list[str]


# Solana validator clients
CLIENTS: list[SolanaClient] = [
    SolanaClient(
        name="Agave",
        organization="Anza (formerly Solana Labs)",
        language="Rust",
        repo="https://github.com/anza-xyz/agave",
        description="Original Solana client, now maintained by Anza. Reference implementation.",
        mainnet_status="production",
        stake_percentage=None,  # Combined with Jito-Agave below
        key_differentiators=[
            "Reference implementation - defines protocol behavior",
            "Most battle-tested codebase",
            "Widest tooling and documentation support",
        ],
        notes=[
            "Forked from solana-labs/solana when Anza spun out",
            "Basis for Jito-Agave (MEV-enabled fork)",
        ],
    ),
    SolanaClient(
        name="Jito-Agave",
        organization="Jito Labs",
        language="Rust",
        repo="https://github.com/jito-foundation/jito-solana",
        description="Agave fork with MEV infrastructure (block engine, bundles, tips).",
        mainnet_status="production",
        stake_percentage=70.0,  # As of Oct 2025
        key_differentiators=[
            "MEV block engine integration",
            "Bundle support for atomic transaction execution",
            "Tip distribution to stakers",
            "Dominant client on mainnet",
        ],
        notes=[
            "~70% of mainnet stake as of Oct 2025",
            "De facto standard for validators seeking MEV rewards",
            "Based on Agave, tracks upstream releases",
        ],
    ),
    SolanaClient(
        name="Firedancer",
        organization="Jump Crypto",
        language="C",
        repo="https://github.com/firedancer-io/firedancer",
        description="Ground-up rewrite in C for maximum performance. Independent implementation.",
        mainnet_status="production",
        stake_percentage=None,  # Full Firedancer still limited
        key_differentiators=[
            "Complete rewrite - independent failure domain",
            "Written in C for hardware-level optimization",
            "Targeting 1M TPS theoretical throughput",
            "Different codebase = different bugs (diversity)",
        ],
        notes=[
            "Went live on mainnet December 2024",
            "~3 years of development by Jump Crypto",
            "Full Firedancer still in limited deployment",
            "Most validators run Frankendancer (hybrid) instead",
        ],
    ),
    SolanaClient(
        name="Frankendancer",
        organization="Jump Crypto",
        language="C + Rust",
        repo="https://github.com/firedancer-io/firedancer",
        description="Hybrid: Firedancer networking + Agave runtime/consensus. Best of both.",
        mainnet_status="production",
        stake_percentage=21.0,  # As of Oct 2025
        key_differentiators=[
            "Firedancer's high-performance networking (600K+ TPS capable)",
            "Agave's battle-tested runtime and consensus",
            "Lower risk than full Firedancer",
            "Easier migration path for validators",
        ],
        notes=[
            "~21% of mainnet stake as of Oct 2025 (up from 8% in June)",
            "Launched on mainnet September 2024",
            "Recommended stepping stone to full Firedancer",
            "Combines Firedancer's fd_quic, fd_tpu with Agave runtime",
        ],
    ),
    SolanaClient(
        name="Sig",
        organization="Syndica",
        language="Zig",
        repo="https://github.com/Syndica/sig",
        description="Zig-based client focused on RPC performance and light clients.",
        mainnet_status="development",
        stake_percentage=0.0,
        key_differentiators=[
            "Written in Zig for safety and performance",
            "Focus on RPC node use case",
            "Potential for light client support",
            "Third independent implementation",
        ],
        notes=[
            "Not yet production-ready for validation",
            "Targeting RPC providers initially",
            "Adds to client diversity ecosystem",
        ],
    ),
]


# Major Solana versions
# Note: Solana versions are more continuous than Ethereum forks
VERSIONS: list[SolanaVersion] = [
    SolanaVersion(
        version="v1.14",
        release_date="2022-10",
        description="QUIC networking, stake-weighted QoS",
        key_features=[
            "QUIC protocol for transaction submission",
            "Stake-weighted quality of service",
            "Improved transaction processing",
        ],
        breaking_changes=[],
    ),
    SolanaVersion(
        version="v1.16",
        release_date="2023-06",
        description="Improved vote costs, versioned transactions",
        key_features=[
            "Reduced vote transaction costs",
            "Address Lookup Tables (ALT) improvements",
            "Versioned transactions default",
        ],
        breaking_changes=[],
    ),
    SolanaVersion(
        version="v1.17",
        release_date="2023-10",
        description="Turbine improvements, SIMD support",
        key_features=[
            "Turbine protocol improvements",
            "Better shred recovery",
            "Gossip optimizations",
        ],
        breaking_changes=[],
    ),
    SolanaVersion(
        version="v1.18",
        release_date="2024-04",
        description="Token extensions, improved finality",
        key_features=[
            "Token-2022 (Token Extensions) wider adoption",
            "Confidential transfers",
            "Transfer hooks",
            "Improved RPC performance",
        ],
        breaking_changes=[],
    ),
    SolanaVersion(
        version="v2.0",
        release_date="2024-08",
        description="Major refactor, Agave client",
        key_features=[
            "Anza's Agave client becomes primary",
            "SVM (Solana Virtual Machine) modularization",
            "Improved validator performance",
            "Better error messages",
        ],
        breaking_changes=[
            "Some deprecated APIs removed",
            "Validator config changes",
        ],
    ),
    SolanaVersion(
        version="v2.1",
        release_date="2024-12",
        description="Performance improvements, Alpenglow prep",
        key_features=[
            "Transaction scheduling improvements",
            "Reduced block propagation latency",
            "Foundation for Alpenglow consensus",
        ],
        breaking_changes=[],
        current=True,
    ),
    # Future
    SolanaVersion(
        version="v2.2 (Alpenglow)",
        release_date="2026-Q1 (expected)",
        description="New consensus protocol - NOT YET LIVE",
        key_features=[
            "Votor: replaces TowerBFT for voting",
            "Rotor: improved data dissemination",
            "~150ms finality (down from 12.8s)",
            "Based on Martin-Alvisi Fast BFT",
        ],
        breaking_changes=[
            "TowerBFT deprecated",
            "Consensus participation changes",
        ],
        current=False,
    ),
]


# Notable feature gates
FEATURE_GATES: list[FeatureGate] = [
    FeatureGate(
        name="require_static_program_ids_in_transaction",
        feature_id="8FUwMvCqV8HMFmKrqZ8JYzLLVwqVrtNh8LdFJCrGdTvr",
        description="Require program IDs to be static in transactions",
        activated_slot=None,
        activated_date=None,
        version_introduced="v1.14",
    ),
    FeatureGate(
        name="vote_state_update_credit_per_dequeue",
        feature_id="CveezY6FDLVBToHDcvJRmtMouqzsmj4UXYh5ths5G5Uv",
        description="Credit per dequeue in vote state updates",
        activated_slot=199_000_000,
        activated_date="2023-09",
        version_introduced="v1.16",
    ),
    FeatureGate(
        name="enable_partitioned_epoch_reward",
        feature_id="9bn2vTJUsUcnpiZWbu2woSKtTGW3ErZC9ERv88SDqQjK",
        description="Partitioned epoch rewards for faster distribution",
        activated_slot=240_000_000,
        activated_date="2024-04",
        version_introduced="v1.18",
    ),
    FeatureGate(
        name="enable_tower_sync_from_snapshots",
        feature_id="HxTQMtHPKrRjVBo76vBPjPKqFHKPnZdfDPnvkyMUppwo",
        description="Allow tower sync from snapshots",
        activated_slot=250_000_000,
        activated_date="2024-06",
        version_introduced="v2.0",
    ),
]


def get_current_version() -> SolanaVersion:
    """Get the current mainnet version."""
    for version in VERSIONS:
        if version.current:
            return version
    return VERSIONS[-2]  # Second to last (last is future)


def list_versions() -> list[SolanaVersion]:
    """List all versions."""
    return VERSIONS


def get_version(version_str: str) -> SolanaVersion | None:
    """Get a specific version by string."""
    for v in VERSIONS:
        if version_str in v.version:
            return v
    return None


def list_feature_gates(activated_only: bool = False) -> list[FeatureGate]:
    """List feature gates."""
    if activated_only:
        return [f for f in FEATURE_GATES if f.activated_slot is not None]
    return FEATURE_GATES


def get_consensus_status() -> dict:
    """Get current consensus mechanism status."""
    return {
        "current": "TowerBFT",
        "current_description": "PBFT variant optimized for Proof of History",
        "finality": "~12.8 seconds (32 confirmations)",
        "optimistic_confirmation": "~2.5 seconds",
        "future": "Alpenglow",
        "future_status": "SIMD-0326 approved, expected Q1 2026",
        "future_finality": "~150ms median",
        "poh_status": "Active (ordering layer, not consensus)",
    }


def list_clients(production_only: bool = False) -> list[SolanaClient]:
    """List Solana validator clients."""
    if production_only:
        return [c for c in CLIENTS if c.mainnet_status == "production"]
    return CLIENTS


def get_client(name: str) -> SolanaClient | None:
    """Get a specific client by name."""
    name_lower = name.lower()
    for client in CLIENTS:
        if name_lower in client.name.lower():
            return client
    return None


def get_client_diversity() -> dict:
    """Get current client diversity statistics."""
    production_clients = [c for c in CLIENTS if c.mainnet_status == "production"]

    return {
        "total_clients": len(CLIENTS),
        "production_clients": len(production_clients),
        "client_breakdown": {
            "Jito-Agave (Rust)": "~70% stake - MEV-enabled Agave fork",
            "Frankendancer (C+Rust)": "~21% stake - Firedancer networking + Agave runtime",
            "Agave (Rust)": "~8% stake - Reference implementation",
            "Firedancer (C)": "<1% stake - Full independent implementation",
            "Sig (Zig)": "0% - Development stage",
        },
        "diversity_notes": [
            "Jito-Agave dominance (~70%) is a centralization concern",
            "Frankendancer growth (8% → 21% in 4 months) improving diversity",
            "Full Firedancer provides true independent failure domain",
            "Ethereum has better diversity (~33% each for top clients)",
        ],
    }


if __name__ == "__main__":
    print("Solana Versions:")
    print("=" * 60)
    for v in VERSIONS:
        current = " (CURRENT)" if v.current else ""
        print(f"\n{v.version}{current} - {v.release_date}")
        print(f"  {v.description}")

    print("\n\nConsensus Status:")
    print("=" * 60)
    status = get_consensus_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n\nClient Diversity:")
    print("=" * 60)
    diversity = get_client_diversity()
    for name, info in diversity["client_breakdown"].items():
        print(f"  {name}: {info}")
