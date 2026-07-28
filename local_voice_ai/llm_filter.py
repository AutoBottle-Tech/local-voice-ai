"""Shared LLM streaming helpers for voice agents."""

import os
import re

from livekit.agents import Agent, llm
from livekit.agents.types import FlushSentinel
from livekit.agents.voice.agent import ModelSettings

_THINKING_TAG = "redacted_thinking"
_THINKING_OPEN = f"<{_THINKING_TAG}>"
_THINKING_CLOSE = f"</{_THINKING_TAG}>"
_THINKING_RE = re.compile(
    rf"{re.escape(_THINKING_OPEN)}.*?{re.escape(_THINKING_CLOSE)}\s*",
    re.DOTALL,
)


def strip_minimax_thinking(text: str) -> str:
    return _THINKING_RE.sub("", text)


class MiniMaxThinkingFilter:
    """Remove MiniMax thinking markup from streamed LLM text (tags may span chunks)."""

    def __init__(self) -> None:
        self._pending = ""
        self._in_thinking = False

    def push(self, text: str) -> str:
        self._pending += text
        out: list[str] = []
        while self._pending:
            if self._in_thinking:
                close = self._pending.find(_THINKING_CLOSE)
                if close == -1:
                    break
                self._pending = self._pending[close + len(_THINKING_CLOSE) :].lstrip()
                self._in_thinking = False
                continue

            open_idx = self._pending.find(_THINKING_OPEN)
            if open_idx == -1:
                hold = 0
                for i in range(1, len(_THINKING_OPEN) + 1):
                    if self._pending.endswith(_THINKING_OPEN[:i]):
                        hold = i
                if hold:
                    if len(self._pending) > hold:
                        out.append(self._pending[:-hold])
                        self._pending = self._pending[-hold:]
                    break
                out.append(self._pending)
                self._pending = ""
                break

            out.append(self._pending[:open_idx])
            self._pending = self._pending[open_idx + len(_THINKING_OPEN) :]
            self._in_thinking = True
        return "".join(out)

    def flush(self) -> str:
        if self._in_thinking:
            self._pending = ""
            self._in_thinking = False
            return ""
        result = self._pending
        self._pending = ""
        return result


async def stream_llm_with_thinking_filter(
    agent: Agent,
    chat_ctx: llm.ChatContext,
    tools: list[llm.Tool],
    model_settings: ModelSettings,
):
    stream = Agent.default.llm_node(agent, chat_ctx, tools, model_settings)
    strip_thinking = os.getenv("LLM_PROVIDER", "llama").lower() == "minimax"
    thinking_filter = MiniMaxThinkingFilter() if strip_thinking else None

    async for chunk in stream:
        if not strip_thinking or thinking_filter is None:
            yield chunk
            continue

        if isinstance(chunk, str):
            cleaned = thinking_filter.push(chunk)
            if cleaned:
                yield cleaned
        elif isinstance(chunk, llm.ChatChunk):
            if chunk.delta and chunk.delta.content:
                chunk.delta.content = thinking_filter.push(chunk.delta.content)
            yield chunk
        elif isinstance(chunk, FlushSentinel):
            remainder = thinking_filter.flush()
            if remainder:
                yield remainder
            yield chunk
        else:
            yield chunk
