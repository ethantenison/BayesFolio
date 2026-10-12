"""External LLM transport adapters for structured BayesFolio workflows."""

from bayesfolio.io.llm.openai_compatible import OpenAICompatibleChatModel, OpenAICompatibleModelError

__all__ = ["OpenAICompatibleChatModel", "OpenAICompatibleModelError"]
