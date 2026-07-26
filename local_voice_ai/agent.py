"""LiveKit Agents worker.

Moved verbatim from ``livekit_agent/src/agent.py``. The only change is that the
default base URLs are loopback (``127.0.0.1``) instead of Docker service names —
the supervisor spawns the inference children on loopback ports, so this is
correct for both single-image deployment and bare-metal local runs.
"""

import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    llm,
)
from livekit.agents.types import FlushSentinel
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv()
load_dotenv(".env.local")

_THINKING_TAG = "redacted_thinking"
_THINKING_OPEN = f"<{_THINKING_TAG}>"
_THINKING_CLOSE = f"</{_THINKING_TAG}>"
_THINKING_RE = re.compile(
    rf"{re.escape(_THINKING_OPEN)}.*?{re.escape(_THINKING_CLOSE)}\s*",
    re.DOTALL,
)


def _strip_minimax_thinking(text: str) -> str:
    return _THINKING_RE.sub("", text)


class _MiniMaxThinkingFilter:
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


def _is_loopback(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    return host in {"", "localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _llm_tls_verify_enabled() -> bool:
    explicit = os.getenv("LLM_TLS_VERIFY_SSL")
    if explicit is not None and explicit.strip() != "":
        return explicit.strip().lower() not in ("0", "false", "no", "off")
    if os.getenv("TLS_INTERCEPT_MODE", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return True


def _resolve_llm() -> tuple[str, str, str, bool]:
    """Return ``(base_url, model, api_key, is_local)`` for the configured LLM."""
    provider = os.getenv("LLM_PROVIDER", "llama").lower()
    if provider == "minimax":
        base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
        model = os.getenv("MINIMAX_MODEL", "MiniMax-M3")
        api_key = (
            os.getenv("MINIMAX_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        if not api_key.strip():
            logger.warning(
                "MINIMAX_API_KEY is empty; set it in .env for MiniMax cloud LLM"
            )
        return base_url, model, api_key or "no-key-needed", False

    base_url = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    model = os.getenv("LLAMA_MODEL", "gemma-4-e2b")
    api_key = os.getenv("LLAMA_API_KEY", "no-key-needed")
    return base_url, model, api_key, _is_loopback(base_url)


def _build_llm_plugin(base_url: str, model: str, api_key: str, is_local: bool) -> openai.LLM:
    provider = os.getenv("LLM_PROVIDER", "llama").lower()
    # MiniMax M3 rejects instruction-only turns with empty user content and, by
    # default, emits long "thinking" tokens that are dead air before TTS gets text.
    llm_kwargs: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
    }
    if provider == "minimax":
        llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    if is_local or _llm_tls_verify_enabled():
        return openai.LLM(**llm_kwargs)

    logger.warning(
        "LLM TLS verification disabled (LLM_TLS_VERIFY_SSL); use only if you trust the network."
    )
    timeout = httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=5.0)
    limits = httpx.Limits(
        max_connections=50,
        max_keepalive_connections=50,
        keepalive_expiry=120,
    )
    return openai.LLM(
        model=model,
        extra_body=llm_kwargs.get("extra_body"),
        client=AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(
                verify=False,
                timeout=timeout,
                follow_redirects=True,
                limits=limits,
            ),
        ),
    )


def _minimax_tts_base_url() -> str:
    """Host root for the MiniMax TTS plugin (no ``/v1`` suffix).

    The LiveKit plugin appends ``/ws/v1/t2a_v2`` (WebSocket) and ``/v1/t2a_v2``
    (HTTP). ``MINIMAX_BASE_URL`` is shared with the OpenAI-compatible LLM client
    and typically ends in ``/v1`` — strip that here to avoid ``/v1/v1/...``.
    """
    base = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1").rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    return base


def _build_tts_plugin():
    tts_provider = os.getenv("TTS_PROVIDER", "kokoro").lower()
    if tts_provider == "minimax":
        from livekit.plugins import minimax

        api_key = (
            os.getenv("MINIMAX_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        model = os.getenv("MINIMAX_TTS_MODEL", "speech-2.8-turbo")
        voice = os.getenv("MINIMAX_TTS_VOICE", "English_Insightful_Speaker")
        if not api_key.strip():
            logger.warning(
                "MINIMAX_API_KEY is empty; set it in .env for MiniMax cloud TTS"
            )
        return minimax.TTS(
            model=model,
            voice=voice,
            api_key=api_key or None,
            base_url=_minimax_tts_base_url(),
            language_boost="English",
            sample_rate=24000,
            audio_format="pcm",
        )

    tts_base_url = os.getenv("TTS_BASE_URL", "http://127.0.0.1:8880/v1")
    tts_voice = os.getenv("TTS_VOICE", "af_nova")
    tts_api_key = os.getenv("TTS_API_KEY", "no-key-needed")
    return openai.TTS(
        base_url=tts_base_url,
        model="tts-1",
        voice=tts_voice,
        api_key=tts_api_key,
    )


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant. The user is interacting with you via "
                "voice, even if you perceive the conversation as text. You eagerly assist "
                "users with their questions by providing information from your extensive "
                "knowledge. Your responses are concise, to the point, and without any "
                "emojis, lists, or other special symbols. "
                "You are curious, friendly, and have a sense of humor."
            ),
        )

    @function_tool()
    async def multiply_numbers(
        self,
        context: RunContext,
        number1: int,
        number2: int,
    ) -> dict[str, Any]:
        """Multiply two numbers.

        Args:
            number1: The first number to multiply.
            number2: The second number to multiply.
        """
        return f"The product of {number1} and {number2} is {number1 * number2}."

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ):
        stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        strip_thinking = os.getenv("LLM_PROVIDER", "llama").lower() == "minimax"
        thinking_filter = _MiniMaxThinkingFilter() if strip_thinking else None

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


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    llm_base_url, llm_model, llm_api_key, llm_local = _resolve_llm()
    llm_provider = os.getenv("LLM_PROVIDER", "llama").lower()

    stt_provider = os.getenv("STT_PROVIDER", "nemotron").lower()
    if stt_provider == "whisper":
        default_stt_base_url = "http://127.0.0.1:8000/v1"
        default_stt_model = "Systran/faster-whisper-small"
    else:
        default_stt_base_url = "http://127.0.0.1:8000/v1"
        default_stt_model = "nemotron-speech-streaming"

    stt_base_url = os.getenv("STT_BASE_URL", default_stt_base_url)
    stt_model = os.getenv("STT_MODEL", default_stt_model)
    stt_api_key = os.getenv("STT_API_KEY", "no-key-needed")
    tts_provider = os.getenv("TTS_PROVIDER", "kokoro").lower()

    logger.info(
        "agent session: stt=%s/%s llm=%s/%s/%s tts=%s",
        stt_provider, stt_model, llm_provider, llm_base_url, llm_model, tts_provider,
    )

    wake_word = os.getenv("WAKE_WORD", "").strip().lower() in {"1", "true", "yes", "on"}
    wake_word_model = os.getenv("WAKE_WORD_MODEL", "/app/models/wakeword/hey_livekit.onnx")
    wake_word_threshold = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))

    session = AgentSession(
        stt=openai.STT(base_url=stt_base_url, model=stt_model, api_key=stt_api_key),
        llm=_build_llm_plugin(llm_base_url, llm_model, llm_api_key, llm_local),
        tts=_build_tts_plugin(),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=llm_provider != "minimax",
    )

    # session.start(room=...) connects to LiveKit internally; do not call ctx.connect() again.
    await session.start(agent=Assistant(), room=ctx.room)

    if wake_word:
        # Join deaf, wait for the wake phrase, then wake up and greet.
        from .wakeword import wait_for_wake_word

        session.input.set_audio_enabled(False)
        participant = await ctx.wait_for_participant()
        try:
            await wait_for_wake_word(participant, wake_word_model, wake_word_threshold)
        except Exception:
            # Fail open: a broken detector shouldn't brick the assistant.
            logger.exception("wake word detection failed; enabling audio input")
        session.input.set_audio_enabled(True)
        session.generate_reply(
            user_input="Hello",
            instructions=(
                "You just woke up because the user said the wake phrase. "
                "Greet them very briefly and ask how you can help."
            ),
        )
    else:
        # MiniMax requires non-empty user content; a placeholder turn is enough.
        session.input.set_audio_enabled(False)
        handle = session.generate_reply(
            user_input="Hello",
            instructions=(
                "Greet the user warmly in one short sentence and invite them "
                "to ask you anything."
            ),
            allow_interruptions=False,
        )
        await handle.wait_for_playout()
        session.input.set_audio_enabled(True)


if __name__ == "__main__":
    cli.run_app(server)
