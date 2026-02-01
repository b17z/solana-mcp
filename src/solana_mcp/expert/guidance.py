"""Expert guidance system for curated Solana interpretations.

This module provides curated expert knowledge that goes beyond
what's explicitly stated in the runtime code. It captures nuances,
gotchas, and insights from Anza research and community discussions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GuidanceEntry:
    """A curated guidance entry."""

    topic: str
    summary: str
    key_points: list[str]
    gotchas: list[str]
    references: list[str]


# Guidance database - populated incrementally
GUIDANCE_DB: dict[str, GuidanceEntry] = {
    "staking": GuidanceEntry(
        topic="Stake Delegation and Rewards",
        summary="Solana uses delegated PoS with warmup/cooldown periods and NO slashing.",
        key_points=[
            "No minimum stake requirement - any amount can be delegated",
            "Warmup period: ~2 epochs for stake to become active",
            "Cooldown period: ~2 epochs for stake to become withdrawable after deactivation",
            "Rewards distributed per-epoch based on vote credits earned by validator",
            "Stakers delegate to validators, not run validators themselves (unlike Ethereum)",
            "Stake accounts are separate from vote accounts",
        ],
        gotchas=[
            "NO SLASHING - Solana does not slash validators for misbehavior",
            "Indirect penalties only: missed leader slots, lost delegations, delinquency",
            "Warmup is proportional - large stakes take longer to fully activate",
            "Stake must be fully cooled down before withdrawal (can't partial withdraw active stake)",
            "Rewards compound automatically if not withdrawn",
        ],
        references=[
            "programs/stake/src/stake_state.rs",
            "programs/stake/src/stake_instruction.rs",
            "runtime/src/bank.rs - epoch reward distribution",
        ],
    ),
    "voting": GuidanceEntry(
        topic="Vote Accounts and Tower Sync",
        summary="Validators vote on blocks using vote accounts; TowerBFT provides consensus.",
        key_points=[
            "Vote accounts track validator's votes and lockouts",
            "Each vote has a lockout that doubles with confirmation depth",
            "32 confirmation levels = 2^32 slots lockout (~centuries)",
            "Vote credits earned for correct votes, used for reward calculation",
            "Tower sync ensures validators don't vote on conflicting forks",
            "Vote transaction fees paid by validator from vote account",
        ],
        gotchas=[
            "Vote accounts need SOL balance for transaction fees",
            "Delinquent validators (missing votes) still in validator set but earn no rewards",
            "Vote latency matters - late votes may not count for credits",
            "Tower state persisted to disk - losing it can cause safety violations",
            "Commission changes take effect next epoch",
        ],
        references=[
            "programs/vote/src/vote_state.rs",
            "programs/vote/src/vote_instruction.rs",
        ],
    ),
    "slashing": GuidanceEntry(
        topic="Why Solana Doesn't Slash (Yet)",
        summary="Solana has NO protocol-level slashing; penalties are indirect market-based.",
        key_points=[
            "No double-sign slashing like Ethereum",
            "No surround vote slashing",
            "No attestation penalties for missed votes",
            "Delinquent validators just don't earn rewards",
            "Market handles bad actors: stakers withdraw and delegate elsewhere",
            "Leader schedule skips validators not producing blocks",
        ],
        gotchas=[
            "This may change - SIMDs could introduce slashing in future",
            "Economic security relies on delegation markets, not protocol punishment",
            "Validators CAN vote on conflicting forks without immediate penalty",
            "TowerBFT lockouts prevent this in practice, but no slashing if violated",
            "Alpenglow (Q1 2026) may introduce new safety mechanisms",
        ],
        references=[
            "N/A - slashing not implemented",
            "SIMD discussions on potential future slashing",
        ],
    ),
    "alpenglow": GuidanceEntry(
        topic="Alpenglow: New Consensus Protocol (Q1 2026)",
        summary="Alpenglow replaces TowerBFT with Votor/Rotor for 100x faster finality.",
        key_points=[
            "Finality: 12.8s (TowerBFT) → ~150ms median, sometimes 100ms",
            "Votor: replaces TowerBFT for voting and block finalization",
            "Rotor: replaces/refines Turbine for data dissemination",
            "Based on Martin-Alvisi Fast Byzantine Consensus (5f+1 bound)",
            "Adversary tolerance: 33% → 20% for fast path",
            "Proof of History (PoH) still used as cryptographic clock",
            "SIMD-0326 approved by validator governance",
        ],
        gotchas=[
            "Testnet expected by Breakpoint (Dec 2025), mainnet Q1 2026",
            "Migration from TowerBFT will require careful coordination",
            "Lower adversary tolerance (20% vs 33%) is tradeoff for speed",
            "PoH is NOT being removed - still provides ordering",
            "Research led by Prof. Wattenhofer (ETH Zurich) who found TowerBFT vulnerabilities",
        ],
        references=[
            "SIMD-0326: proposals/0326-alpenglow.md",
            "https://www.anza.xyz/blog/alpenglow-a-new-consensus-for-solana",
            "https://www.helius.dev/blog/alpenglow",
        ],
    ),
    "poh": GuidanceEntry(
        topic="Proof of History",
        summary="PoH is a cryptographic clock providing ordering, NOT consensus.",
        key_points=[
            "Sequential SHA-256 hashing creates verifiable passage of time",
            "Each hash depends on previous - can't be parallelized",
            "Provides global ordering without communication",
            "Leader produces PoH stream, validators verify",
            "NOT a consensus mechanism - TowerBFT/Alpenglow handle consensus",
            "Enables high throughput by reducing coordination overhead",
        ],
        gotchas=[
            "PoH is often misunderstood as Solana's consensus - it's not",
            "PoH generator is a single point of failure during leader slot",
            "Verification is parallelizable, generation is not",
            "Time assumptions matter - clock drift can cause issues",
            "Alpenglow keeps PoH, only replaces the consensus layer",
        ],
        references=[
            "poh/src/poh_recorder.rs",
            "poh/src/poh_service.rs",
        ],
    ),
    "accounts": GuidanceEntry(
        topic="Account Model",
        summary="Solana uses flat account model with ownership and rent.",
        key_points=[
            "Accounts are flat key-value pairs, not a state trie",
            "Each account has: lamports, data, owner, executable, rent_epoch",
            "Programs (smart contracts) own accounts they create",
            "Only the owner program can modify account data",
            "System program owns native SOL accounts",
            "Rent: accounts must maintain minimum balance or be garbage collected",
        ],
        gotchas=[
            "Account data size is fixed at creation (can't resize easily)",
            "Rent-exempt threshold: ~2 years of rent paid upfront",
            "Cross-program invocations (CPI) pass accounts explicitly",
            "PDAs (Program Derived Addresses) for deterministic account addresses",
            "Account lookup tables (ALTs) for transaction size optimization",
        ],
        references=[
            "runtime/src/accounts.rs",
            "sdk/src/account.rs",
        ],
    ),
    "svm": GuidanceEntry(
        topic="Solana Virtual Machine",
        summary="SVM executes BPF/SBF bytecode with parallel transaction processing.",
        key_points=[
            "Programs compile to BPF (Berkeley Packet Filter) bytecode",
            "SBF (Solana BPF) is Solana's extended BPF variant",
            "Parallel execution: transactions touching different accounts run concurrently",
            "Compute units: each instruction costs CUs, capped per transaction",
            "Native programs (vote, stake, system) run directly, not in VM",
            "Upgradeable programs use BPF Loader v3",
        ],
        gotchas=[
            "Account locking determines parallelism - overlapping accounts = sequential",
            "Compute budget can be increased with priority fees",
            "Stack size limited - deep recursion will fail",
            "Heap size limited to 32KB by default",
            "Cross-program invocations have depth limit (4)",
        ],
        references=[
            "svm/src/",
            "program-runtime/src/",
            "programs/bpf_loader/",
        ],
    ),
    "turbine": GuidanceEntry(
        topic="Turbine Block Propagation",
        summary="Turbine uses erasure coding and tree structure for efficient block propagation.",
        key_points=[
            "Blocks split into shreds (data fragments)",
            "Reed-Solomon erasure coding for redundancy",
            "Tree structure: leader → root → branches → leaves",
            "Each node forwards to small fanout, not broadcast",
            "Reduces leader bandwidth from O(n) to O(log n)",
            "Retransmit stage handles missing shreds",
        ],
        gotchas=[
            "Tree structure means propagation latency varies by position",
            "Stake-weighted tree assignment - higher stake = earlier in tree",
            "Network partitions can cause shred loss cascades",
            "Rotor (Alpenglow) refines but doesn't fully replace Turbine concepts",
            "Repair protocol handles missing shreds after propagation",
        ],
        references=[
            "turbine/src/",
            "ledger/src/shred.rs",
        ],
    ),
    "towerbft": GuidanceEntry(
        topic="TowerBFT: Current Consensus (Pre-Alpenglow)",
        summary="TowerBFT is Solana's CURRENT consensus mechanism - a PBFT variant optimized for PoH.",
        key_points=[
            "CURRENT CONSENSUS (Jan 2026) - Alpenglow not yet live on mainnet",
            "PBFT-like consensus optimized to leverage Proof of History",
            "Validators vote on blocks; votes have exponential lockouts",
            "Lockout doubles with each confirmation: 2, 4, 8, 16... slots",
            "32 confirmations = 2^32 slots lockout (effectively permanent)",
            "Fork choice: pick fork with most stake-weighted votes",
            "Finality: ~12.8 seconds (32 confirmations at 400ms slots)",
            "Optimistic confirmation: ~2.5 seconds for high-confidence",
        ],
        gotchas=[
            "TowerBFT is NOT being replaced until Alpenglow ships (expected Q1 2026)",
            "PoH provides ordering, TowerBFT provides consensus - they work together",
            "Lockouts prevent validators from easily switching forks (Byzantine resistance)",
            "Tower state must be persisted - losing it risks voting on conflicting forks",
            "No explicit slashing for safety violations, but lockouts make violations expensive",
            "Wattenhofer et al. found theoretical liveness vulnerabilities (epsilon stake attack)",
            "In practice, TowerBFT has been reliable since mainnet launch",
        ],
        references=[
            "programs/vote/src/vote_state.rs - VoteState, lockouts",
            "core/src/consensus.rs - Tower, fork choice",
            "core/src/replay_stage.rs - block replay and voting",
            "https://docs.solanalabs.com/consensus/tower-bft",
        ],
    ),
    "consensus": GuidanceEntry(
        topic="Solana Consensus Overview",
        summary="Solana consensus = PoH (ordering) + TowerBFT (agreement). Alpenglow coming Q1 2026.",
        key_points=[
            "TWO components work together: PoH for ordering, TowerBFT for consensus",
            "PoH is NOT consensus - it's a cryptographic clock for ordering",
            "TowerBFT is the ACTUAL consensus mechanism (PBFT variant)",
            "Leader rotation: stake-weighted schedule, ~4 slots per leader",
            "Blocks confirmed via vote transactions from validators",
            "Finality: ~12.8s (TowerBFT) → ~150ms (Alpenglow, not yet live)",
            "Fork choice: heaviest subtree by stake-weighted votes",
        ],
        gotchas=[
            "Common misconception: 'PoH is Solana's consensus' - FALSE",
            "PoH reduces communication overhead, TowerBFT provides BFT agreement",
            "Alpenglow (SIMD-0326) approved but NOT YET LIVE as of Jan 2026",
            "Expected timeline: testnet late 2025, mainnet Q1 2026",
            "Current mainnet still runs TowerBFT + PoH",
        ],
        references=[
            "See 'towerbft' for current consensus details",
            "See 'alpenglow' for future consensus details",
            "See 'poh' for Proof of History details",
        ],
    ),
    "leader_schedule": GuidanceEntry(
        topic="Leader Schedule and Block Production",
        summary="Validators take turns producing blocks based on stake-weighted schedule.",
        key_points=[
            "Leader schedule computed per epoch from stake distribution",
            "Each leader gets ~4 consecutive slots (1.6 seconds)",
            "Higher stake = more leader slots = more block rewards",
            "Leader produces blocks containing transactions",
            "Blocks signed by leader and include PoH hashes",
            "Missed leader slots: no block produced, network continues",
        ],
        gotchas=[
            "Leader schedule is deterministic - everyone computes the same schedule",
            "Schedule based on stake at epoch boundary (2 epochs ahead)",
            "Validators can be 'delinquent' if they miss too many slots",
            "No penalty for missed slots except lost rewards",
            "Jito and other MEV infrastructure can affect leader slot value",
        ],
        references=[
            "runtime/src/bank.rs - leader_schedule_epoch",
            "ledger/src/leader_schedule.rs",
        ],
    ),
    "epochs": GuidanceEntry(
        topic="Epochs and Timing",
        summary="Epochs are ~2-3 day periods for stake activation, rewards, and schedule changes.",
        key_points=[
            "Epoch = 432,000 slots = ~2-3 days (at 400ms/slot)",
            "Stake changes (activation/deactivation) take effect at epoch boundaries",
            "Rewards distributed at end of each epoch",
            "Leader schedule computed for epoch N+2 at start of epoch N",
            "Rent collection happens per-epoch",
            "Vote account commission changes effective next epoch",
        ],
        gotchas=[
            "Slot duration is 400ms TARGET but can vary",
            "Epoch duration varies with actual slot times",
            "Stake warmup/cooldown spans epoch boundaries",
            "'Current epoch' for stake purposes is when it becomes active",
            "First epoch after activation only gets partial rewards",
        ],
        references=[
            "sdk/src/clock.rs - Epoch, Slot, Clock",
            "runtime/src/bank.rs - epoch transitions",
        ],
    ),
    # ===================
    # MEV / Jito
    # ===================
    "mev": GuidanceEntry(
        topic="MEV on Solana (Jito)",
        summary="Jito provides MEV infrastructure for Solana - bundles, block engine, and tip distribution.",
        key_points=[
            "Jito-Agave: MEV-enabled fork of Agave, runs ~70% of mainnet stake",
            "Block Engine: off-chain service that receives bundles from searchers",
            "Bundles: atomic transaction groups that execute all-or-nothing",
            "Tips: searchers pay validators directly for bundle inclusion",
            "Tip distribution: tips split between validator and stakers",
            "MEV sources: DEX arbitrage, liquidations, NFT mints",
            "No mempool: Solana's no-mempool design changes MEV dynamics",
        ],
        gotchas=[
            "Solana MEV is different from Ethereum - no mempool means no traditional frontrunning",
            "Searchers connect to block engine, not directly to validators",
            "Bundles land at end of block (not beginning like Ethereum)",
            "Tip accounts: special accounts that receive and distribute tips",
            "Jito dominance (~70%) is a centralization concern",
            "Jito temporarily paused mempool feature in 2024 due to sandwich concerns",
            "Validators can run without Jito but lose MEV revenue",
        ],
        references=[
            "jito-solana/bundle/",
            "jito-solana/tip-distributor/",
            "https://jito-labs.gitbook.io/",
        ],
    ),
    "jito": GuidanceEntry(
        topic="Jito Infrastructure",
        summary="Jito Labs provides the dominant MEV infrastructure for Solana validators.",
        key_points=[
            "Jito-Agave: forked Agave client with block engine integration",
            "Block Engine: receives bundles, simulates, forwards to validators",
            "Bundles: 1-5 transactions that execute atomically",
            "Tips: SOL payments from searchers to validators for inclusion",
            "Tip Distribution Program: on-chain program for tip accounting",
            "MEV Dashboard: jito.wtf shows MEV statistics",
            "~70% of mainnet validators run Jito-Agave",
        ],
        gotchas=[
            "Jito is a company, not a protocol - centralization risk",
            "Block engine is off-chain and closed-source",
            "Validators trust Jito to forward bundles fairly",
            "Tip distribution happens on-chain (verifiable)",
            "Jito foundation (non-profit) governs JTO token",
            "JTO token launched Dec 2023 - governance + staking",
            "Frankendancer integration coming - Jito + Firedancer networking",
        ],
        references=[
            "jito-foundation/jito-solana",
            "jito-foundation/jito-programs",
            "https://www.jito.network/",
        ],
    ),
    "bundles": GuidanceEntry(
        topic="Jito Bundles",
        summary="Bundles are atomic transaction groups for MEV extraction on Solana.",
        key_points=[
            "Bundle = 1-5 transactions that execute all-or-nothing",
            "Atomic: all succeed or all fail (no partial execution)",
            "Ordered: transactions execute in specified order",
            "Tip required: bundles include tip transaction to validator",
            "Simulation: block engine simulates bundles before forwarding",
            "Landing: bundles placed at END of block (after normal txs)",
            "Priority: higher tip = higher priority in block engine queue",
        ],
        gotchas=[
            "Bundles land at BLOCK END, not beginning (unlike Ethereum)",
            "This means bundles can't frontrun normal transactions",
            "Bundles CAN backrun - execute after a target transaction",
            "Sandwich attacks harder but not impossible",
            "Bundle simulation can fail - no guarantee of landing",
            "Tip is paid even if bundle reverts (validator gets tip)",
            "Multiple bundles can conflict - block engine handles conflicts",
        ],
        references=[
            "jito-solana/bundle/",
            "https://jito-labs.gitbook.io/mev/",
        ],
    ),
    "tips": GuidanceEntry(
        topic="Jito Tip Distribution",
        summary="Tips are SOL payments from searchers to validators, distributed to stakers.",
        key_points=[
            "Tip accounts: 8 special accounts that receive tips",
            "Searchers send tip transaction as part of bundle",
            "Tips collected per-block, distributed per-epoch",
            "Distribution: validator commission + staker rewards",
            "Tip Distribution Program: on-chain, verifiable distribution",
            "NCN (Node Consensus Network): upcoming decentralized tip routing",
            "Average tips: varies widely, ~0.001-0.1 SOL per block",
        ],
        gotchas=[
            "Tips are NOT transaction fees - separate mechanism",
            "Tip accounts rotate - don't hardcode addresses",
            "Validator can set custom tip commission (separate from vote commission)",
            "Stakers receive tip rewards proportional to stake",
            "Tips visible on-chain but searcher identity often hidden",
            "Large tips often indicate high-value MEV (big arb, liquidation)",
        ],
        references=[
            "jito-solana/tip-distributor/",
            "jito-foundation/jito-programs",
        ],
    ),
}


def get_expert_guidance(topic: str) -> GuidanceEntry | None:
    """
    Get expert guidance for a topic.

    Searches for exact match first, then partial matches.
    """
    topic_lower = topic.lower()

    # Exact match
    if topic_lower in GUIDANCE_DB:
        return GUIDANCE_DB[topic_lower]

    # Partial match
    for key, entry in GUIDANCE_DB.items():
        if topic_lower in key or key in topic_lower:
            return entry
        if topic_lower in entry.summary.lower():
            return entry

    return None


def list_guidance_topics() -> list[str]:
    """List all available guidance topics."""
    return list(GUIDANCE_DB.keys())


def add_guidance(entry: GuidanceEntry) -> None:
    """Add a new guidance entry."""
    GUIDANCE_DB[entry.topic.lower()] = entry
