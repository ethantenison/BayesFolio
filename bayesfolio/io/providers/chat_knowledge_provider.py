"""IO provider for deterministic Riskfolio knowledge retrieval.

Boundary responsibility: this module builds a read-only knowledge corpus from
local documentation sources and static Riskfolio reference snippets, then
returns ranked evidence snippets plus normalized parameter suggestions.
No optimization business logic or external network calls are performed.
"""

from __future__ import annotations

import re
from pathlib import Path


class ChatKnowledgeProvider:
    """Provide deterministic retrieval and normalization hints for chat tools.

    Attributes:
        _workspace_root: Repository root used to resolve local corpus files.
        _corpus_paths: Relative corpus file paths for local ingestion.
        _max_chunk_chars: Maximum approximate size of each text chunk.
    """

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        corpus_paths: list[str] | None = None,
        max_chunk_chars: int = 900,
    ) -> None:
        """Initialize the provider with local corpus configuration.

        Args:
            workspace_root: Repository root directory. Defaults to current
                working directory.
            corpus_paths: Relative paths for local documentation ingestion.
            max_chunk_chars: Approximate max chunk size for retrieval indexing.
        """

        self._workspace_root = Path(workspace_root) if workspace_root is not None else Path.cwd()
        self._corpus_paths = corpus_paths or [
            "README.md",
            "docs/copilot_architecture.md",
            "docs/package_map.md",
            ".github/copilot-instructions.md",
        ]
        self._max_chunk_chars = max_chunk_chars

    def retrieve_and_suggest(self, query: str, top_k: int = 5) -> dict[str, object]:
        """Retrieve evidence snippets and suggestion hints for a user query.

        Args:
            query: User query text.
            top_k: Maximum number of ranked snippets to return.

        Returns:
            Dictionary with snippets, suggested overrides, and retrieval metadata.
        """

        snippets = self.retrieve(query=query, top_k=top_k)
        suggested_overrides = self.suggest_overrides(query=query)
        return {
            "query": query,
            "top_k": top_k,
            "snippets": snippets,
            "suggested_overrides": suggested_overrides,
            "corpus_scope": "docs-only",
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, object]]:
        """Return ranked snippets from the local knowledge corpus.

        Args:
            query: User query text.
            top_k: Maximum number of snippets to return.

        Returns:
            Ranked snippets with source path/URL, topic, score, and text.
        """

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scored: list[dict[str, object]] = []
        for chunk in self._build_chunks():
            chunk_terms = _tokenize(chunk["text"])
            if not chunk_terms:
                continue
            overlap = len(query_terms & chunk_terms)
            if overlap == 0:
                continue

            score = overlap / len(query_terms)
            scored.append(
                {
                    "source": chunk["source"],
                    "topic": chunk["topic"],
                    "score": round(float(score), 4),
                    "text": chunk["text"],
                }
            )

        scored.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item["topic"]),
                str(item["source"]),
                str(item["text"]),
            )
        )
        return scored[: max(int(top_k), 1)]

    def suggest_overrides(self, query: str) -> dict[str, object]:
        """Infer canonical Riskfolio parameter overrides from free-form query.

        Args:
            query: User query text.

        Returns:
            Dictionary containing canonical keys (for example ``objective`` and
            ``risk_measure``) when inferable from text.
        """

        lowered = query.lower()
        objective = _match_alias(
            lowered,
            {
                "MinRisk": ["minrisk", "min risk", "minimum risk"],
                "MaxRet": ["maxret", "max ret", "max return", "maximum return"],
                "Utility": ["utility", "risk aversion"],
                "Sharpe": ["sharpe", "risk adjusted", "risk-adjusted"],
            },
        )
        risk_measure = _match_alias(
            lowered,
            {
                "CVaR": ["cvar", "conditional value at risk"],
                "MV": ["mv", "variance", "standard deviation", "stdev"],
                "MAD": ["mad", "mean absolute deviation"],
                "CDaR": ["cdar", "conditional drawdown at risk"],
                "EVaR": ["evar", "entropic value at risk"],
                "RLVaR": ["rlvar", "relativistic value at risk"],
                "MDD": ["mdd", "max drawdown", "maximum drawdown"],
                "UCI": ["ulcer index", "uci"],
                "GMD": ["gmd", "gini mean difference"],
                "SLPM": ["sortino", "slpm", "second lower partial moment"],
                "FLPM": ["omega", "flpm", "first lower partial moment"],
            },
        )
        model = _match_alias(
            lowered,
            {
                "Classic": ["classic model", "classic"],
                "BL": ["black litterman", "bl model", "model bl"],
                "FM": ["factor model", "fm model", "model fm"],
                "BLFM": ["blfm", "black litterman factor model", "model blfm"],
            },
        )
        kelly = _match_alias(
            lowered,
            {
                "exact": ["kelly exact", "exact kelly"],
                "approx": ["kelly approx", "approx kelly", "approximate kelly"],
            },
        )
        hist = _infer_hist_flag(lowered)
        rf = _extract_rf_decimal(lowered)

        overrides: dict[str, object] = {}
        if objective is not None:
            overrides["objective"] = objective
        if risk_measure is not None:
            overrides["risk_measure"] = risk_measure
        if model is not None:
            overrides["model"] = model
        if kelly is not None:
            overrides["kelly"] = kelly
        if hist is not None:
            overrides["hist"] = hist
        if rf is not None:
            overrides["rf"] = rf
        return overrides

    def _build_chunks(self) -> list[dict[str, str]]:
        """Build the retrieval corpus from local docs and static references."""

        chunks: list[dict[str, str]] = []
        for relative_path in self._corpus_paths:
            full_path = self._workspace_root / relative_path
            if not full_path.exists() or not full_path.is_file():
                continue

            try:
                content = full_path.read_text(encoding="utf-8")
            except OSError:
                continue

            topic = Path(relative_path).stem
            chunks.extend(
                _chunk_text(
                    content=content,
                    source=relative_path,
                    topic=topic,
                    max_chars=self._max_chunk_chars,
                )
            )

        chunks.extend(_static_riskfolio_reference_chunks())
        return chunks


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) >= 2}


def _chunk_text(content: str, source: str, topic: str, max_chars: int) -> list[dict[str, str]]:
    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", content) if segment.strip()]
    chunks: list[dict[str, str]] = []
    current: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current and current_length + paragraph_length > max_chars:
            chunks.append({"source": source, "topic": topic, "text": "\n\n".join(current)})
            current = [paragraph]
            current_length = paragraph_length
        else:
            current.append(paragraph)
            current_length += paragraph_length

    if current:
        chunks.append({"source": source, "topic": topic, "text": "\n\n".join(current)})
    return chunks


def _match_alias(lowered_query: str, alias_map: dict[str, list[str]]) -> str | None:
    for canonical, aliases in alias_map.items():
        if any(alias in lowered_query for alias in aliases):
            return canonical
    return None


def _static_riskfolio_reference_chunks() -> list[dict[str, str]]:
    return [
        {
            "source": "https://riskfolio-lib.readthedocs.io/en/latest/portfolio.html",
            "topic": "riskfolio_objectives",
            "text": (
                "Riskfolio objective codes include MinRisk, Utility, Sharpe, and MaxRet. "
                "Use these canonical values when mapping user intent to optimization parameters."
            ),
        },
        {
            "source": "https://riskfolio-lib.readthedocs.io/en/latest/portfolio.html",
            "topic": "riskfolio_risk_measures",
            "text": (
                "Riskfolio risk measure codes include MV, CVaR, MAD, CDaR, EVaR, RLVaR, MDD, "
                "UCI, and others. Use canonical rm values when normalizing free-form requests."
            ),
        },
        {
            "source": "https://riskfolio-lib.readthedocs.io/en/latest/portfolio.html",
            "topic": "riskfolio_defaults",
            "text": (
                "Reasonable defaults for mean-risk optimization are model Classic, objective Sharpe, "
                "risk measure MV, risk-free rate 0, and hist=True. Kelly is optional and defaults to None."
            ),
        },
        {
            "source": "https://riskfolio-lib.readthedocs.io/en/latest/index.html",
            "topic": "riskfolio_solver_notes",
            "text": (
                "Some relativistic or entropic risk measures can be solver-sensitive. "
                "RLVaR and RLDaR are often more stable with MOSEK for large or difficult problems."
            ),
        },
    ]


def _infer_hist_flag(lowered_query: str) -> bool | None:
    if "hist false" in lowered_query or "non historical" in lowered_query or "no historical" in lowered_query:
        return False
    if "hist true" in lowered_query or "historical" in lowered_query:
        return True
    return None


def _extract_rf_decimal(lowered_query: str) -> float | None:
    match = re.search(r"\b(?:rf|risk free(?: rate)?)\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*(%)?", lowered_query)
    if match is None:
        return None

    value = float(match.group(1))
    if match.group(2) == "%" or value > 1.0:
        value = value / 100.0
    return value
