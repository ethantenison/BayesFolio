"""Tests for deterministic chat knowledge retrieval provider."""

from __future__ import annotations

from pathlib import Path

from bayesfolio.io.providers.chat_knowledge_provider import ChatKnowledgeProvider


def test_chat_knowledge_provider_returns_ranked_snippets(tmp_path: Path) -> None:
    readme_path = tmp_path / "README.md"
    readme_path.write_text("Riskfolio objective Sharpe and risk measure CVaR are supported.", encoding="utf-8")

    provider = ChatKnowledgeProvider(
        workspace_root=tmp_path,
        corpus_paths=["README.md"],
    )

    payload = provider.retrieve_and_suggest(query="optimize with cvar and sharpe", top_k=3)

    snippets = payload["snippets"]
    assert isinstance(snippets, list)
    assert len(snippets) >= 1
    assert snippets[0]["score"] >= 0.0
    assert snippets[0]["source"] == "README.md"


def test_chat_knowledge_provider_suggests_canonical_overrides() -> None:
    provider = ChatKnowledgeProvider(corpus_paths=[])

    suggestions = provider.suggest_overrides("Please run min risk with variance")

    assert suggestions["objective"] == "MinRisk"
    assert suggestions["risk_measure"] == "MV"


def test_chat_knowledge_provider_suggests_extended_riskfolio_fields() -> None:
    provider = ChatKnowledgeProvider(corpus_paths=[])

    suggestions = provider.suggest_overrides("Use black litterman model with rf 2%, hist false and kelly exact")

    assert suggestions["model"] == "BL"
    assert suggestions["rf"] == 0.02
    assert suggestions["hist"] is False
    assert suggestions["kelly"] == "exact"


def test_chat_knowledge_provider_handles_empty_query() -> None:
    provider = ChatKnowledgeProvider(corpus_paths=[])

    snippets = provider.retrieve(query="", top_k=5)

    assert snippets == []
