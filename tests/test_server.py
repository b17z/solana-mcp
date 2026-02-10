"""Tests for the MCP server."""

from solana_mcp.server import _source_to_github_url


class TestSourceToGithubUrl:
    """Tests for GitHub URL generation."""

    def test_agave_with_repo_param(self):
        """Agave uses master branch when repo is provided."""
        url = _source_to_github_url("programs/stake/src/lib.rs", 42, repo="agave")
        assert url == "https://github.com/anza-xyz/agave/blob/master/programs/stake/src/lib.rs#L42"

    def test_jito_solana_with_repo_param(self):
        """Jito-solana uses master branch."""
        url = _source_to_github_url("core/src/bundle.rs", 100, repo="jito-solana")
        assert url == "https://github.com/jito-foundation/jito-solana/blob/master/core/src/bundle.rs#L100"

    def test_firedancer_with_repo_param(self):
        """Firedancer uses main branch."""
        url = _source_to_github_url("src/disco/consensus.c", 50, repo="firedancer")
        assert url == "https://github.com/firedancer-io/firedancer/blob/main/src/disco/consensus.c#L50"

    def test_simds_path_based(self):
        """SIMDs use main branch (path-based matching)."""
        url = _source_to_github_url(
            "solana-improvement-documents/proposals/0123-example.md"
        )
        assert url == (
            "https://github.com/solana-foundation/solana-improvement-documents"
            "/blob/main/proposals/0123-example.md"
        )

    def test_alpenglow_with_repo_param(self):
        """Alpenglow uses master branch."""
        url = _source_to_github_url("src/lib.rs", 1, repo="alpenglow")
        assert url == "https://github.com/anza-xyz/alpenglow/blob/master/src/lib.rs#L1"

    def test_jito_programs_with_repo_param(self):
        """Jito programs use master branch."""
        url = _source_to_github_url("mev-programs/src/lib.rs", repo="jito-programs")
        assert url == (
            "https://github.com/jito-foundation/jito-programs"
            "/blob/master/mev-programs/src/lib.rs"
        )

    def test_without_line_number(self):
        """URL without line number."""
        url = _source_to_github_url("runtime/src/bank.rs", repo="agave")
        assert url == "https://github.com/anza-xyz/agave/blob/master/runtime/src/bank.rs"
        assert "#L" not in url

    def test_unknown_repo_returns_none(self):
        """Unknown repos return None when using path-based matching."""
        url = _source_to_github_url("unknown-repo/src/lib.rs")
        assert url is None

    def test_empty_path_returns_none(self):
        """Empty path returns None."""
        url = _source_to_github_url("")
        assert url is None

    def test_path_based_fallback(self):
        """Path-based matching works for SIMDs without repo param."""
        url = _source_to_github_url("solana-improvement-documents/proposals/0001.md")
        assert "solana-foundation/solana-improvement-documents" in url
        assert "/blob/main/" in url
