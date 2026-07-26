"""Environment-driven configuration for the local-voice-ai supervisor.

A single ``Config`` object is constructed at startup from environment variables
and shared with every subsystem (supervisor, agent, FastAPI routes).

The "manage X" flags decide whether the supervisor will spawn a given service
as a child process. They default to ``True`` when the matching base URL is a
loopback address (or unset), and ``False`` otherwise — pointing any base URL
at a remote endpoint automatically disables the local child.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

# Provider option lists exposed to the settings UI.
LLM_PROVIDERS = ("llama", "minimax")
STT_PROVIDERS = ("nemotron", "whisper")
TTS_PROVIDERS = ("kokoro", "minimax")
MINIMAX_TTS_MODELS = (
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
)
MINIMAX_TTS_MODEL_LABELS: dict[str, str] = {
    "speech-2.8-hd": "Speech 2.8 HD — ultra-realistic, sound tags",
    "speech-2.8-turbo": "Speech 2.8 Turbo — fast, natural flow",
    "speech-2.6-hd": "Speech 2.6 HD — low latency, enhanced naturalness",
    "speech-2.6-turbo": "Speech 2.6 Turbo — faster, agent-friendly",
    "speech-02-hd": "Speech 02 HD — superior rhythm and stability",
    "speech-02-turbo": "Speech 02 Turbo — multilingual, stable",
}
KOKORO_VOICES = (
    "af_nova",
    "af_bella",
    "af_sarah",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
)
MINIMAX_TTS_VOICES = (
    "socialmedia_female_2_v1",
    "socialmedia_female_1_v1",
    "voice_agent_Female_Phone_4",
    "voice_agent_Male_Phone_1",
    "English_WiseScholar",
    "English_Insightful_Speaker",
    "English_Persuasive_Man",
    "English_radiant_girl",
)

# Settings-page keys persisted to ``.env.local`` (never secrets).
PERSISTABLE_ENV_KEYS = (
    "LLM_PROVIDER",
    "STT_PROVIDER",
    "TTS_PROVIDER",
    "TTS_VOICE",
    "MINIMAX_TTS_MODEL",
    "MINIMAX_TTS_VOICE",
    "WAKE_WORD",
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_bool_opt(name: str) -> Optional[bool]:
    """Like ``_env_bool`` but returns ``None`` when the var is unset, so callers
    can distinguish "not configured" (auto) from an explicit true/false."""
    raw = os.getenv(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_loopback(url: str) -> bool:
    """Return True if ``url`` points at the local machine."""
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    return host in {"", "localhost", "127.0.0.1", "0.0.0.0", "::1"}


@dataclass
class Config:
    # --- Web (FastAPI in the supervisor process) -------------------------
    web_host: str = "0.0.0.0"
    web_port: int = 8080
    frontend_dir: Optional[str] = None  # path to a Next.js static export dir

    # --- LiveKit ---------------------------------------------------------
    livekit_url: str = "ws://127.0.0.1:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"
    livekit_bind_port: int = 7880
    livekit_rtc_port: int = 7881  # WebRTC over TCP (ICE/TCP fallback)
    livekit_udp_port: int = 7882  # WebRTC over UDP (preferred media transport)
    # IP the managed dev server advertises in ICE candidates. 127.0.0.1 is
    # reachable both from a browser on the host (via Docker-published ports) and
    # from the in-container agent (via loopback). Override (LIVEKIT_NODE_IP) when
    # running the server on a remote host reached over the network.
    livekit_node_ip: str = "127.0.0.1"
    manage_livekit: bool = True

    # --- LLM (llama.cpp local by default; set LLM_PROVIDER=minimax for cloud)
    llm_provider: str = "llama"  # "llama" | "minimax"
    minimax_base_url: str = "https://api.minimax.io/v1"
    minimax_model: str = "MiniMax-M3"
    minimax_api_key: str = ""
    #
    llama_base_url: str = "http://127.0.0.1:11434/v1"
    llama_model: str = "gemma-4-e2b"
    llama_api_key: str = "no-key-needed"
    # Quantization-aware-trained quant — holds up much better at 4-bit than
    # post-hoc quantization. The :tag suffix selects the quant within the repo.
    llama_hf_repo: str = "unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL"
    # Path to a local .gguf. When set, llama-server loads it directly with -m
    # instead of resolving --hf-repo against Hugging Face (works fully offline).
    llama_model_path: str = ""
    # Pass --offline to llama-server: use only cached files, never hit the
    # network. Lets a previously-downloaded --hf-repo model start with no
    # internet. ``None`` (the default) means auto: enable --offline when the
    # model is already cached, otherwise allow the first-run download.
    # See https://github.com/ShayneP/local-voice-ai/issues/9
    llama_offline: Optional[bool] = None
    llama_model_alias: str = "gemma-4-e2b"
    llama_ctx_size: int = 16384
    llama_n_gpu_layers: int = 0
    llama_bind_port: int = 11434
    manage_llama: bool = True

    # --- STT (Nemotron by default) --------------------------------------
    stt_provider: str = "nemotron"  # "nemotron" | "whisper"
    stt_base_url: str = "http://127.0.0.1:8000/v1"
    stt_model: str = "nemotron-speech-streaming"
    stt_api_key: str = "no-key-needed"
    stt_bind_port: int = 8000
    manage_stt: bool = True

    # Nemotron-specific
    nemotron_model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    nemotron_model_id: str = "nemotron-speech-streaming"

    # Whisper (faster-whisper) specific
    whisper_model: str = "Systran/faster-whisper-small"

    # --- TTS (Kokoro local by default; set TTS_PROVIDER=minimax for cloud)
    tts_provider: str = "kokoro"  # "kokoro" | "minimax"
    tts_base_url: str = "http://127.0.0.1:8880/v1"
    tts_voice: str = "af_nova"
    tts_api_key: str = "no-key-needed"
    tts_bind_port: int = 8880
    manage_tts: bool = True
    minimax_tts_model: str = "speech-2.8-turbo"
    minimax_tts_voice: str = "English_Insightful_Speaker"

    # --- Wake word (off by default) --------------------------------------
    # When enabled the agent joins deaf and only starts listening after the
    # wake phrase ("hey livekit") is detected on the user's microphone.
    wake_word: bool = False
    wake_word_model: str = "/app/models/wakeword/hey_livekit.onnx"
    wake_word_threshold: float = 0.5

    # --- Device ---------------------------------------------------------
    device: str = "cpu"  # cpu | cuda | mps

    # --- Misc -----------------------------------------------------------
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Build the config from ``os.environ`` with sane defaults."""
        livekit_url = os.getenv("LIVEKIT_URL", cls.livekit_url)
        llama_base_url = os.getenv("LLAMA_BASE_URL", cls.llama_base_url)
        stt_base_url = os.getenv("STT_BASE_URL")
        tts_base_url = os.getenv("TTS_BASE_URL", cls.tts_base_url)

        stt_provider = os.getenv("STT_PROVIDER", cls.stt_provider).lower()
        if stt_base_url is None:
            # Default STT URL depends on provider
            stt_base_url = (
                "http://127.0.0.1:8000/v1"
                if stt_provider != "whisper"
                else "http://127.0.0.1:8000/v1"
            )

        default_stt_model = (
            "Systran/faster-whisper-small"
            if stt_provider == "whisper"
            else "nemotron-speech-streaming"
        )

        llm_provider = os.getenv("LLM_PROVIDER", cls.llm_provider).lower()
        tts_provider = os.getenv("TTS_PROVIDER", cls.tts_provider).lower()

        return cls(
            web_host=os.getenv("WEB_HOST", cls.web_host),
            web_port=int(os.getenv("WEB_PORT", str(cls.web_port))),
            frontend_dir=os.getenv("FRONTEND_DIR"),
            #
            llm_provider=llm_provider,
            minimax_base_url=os.getenv("MINIMAX_BASE_URL", cls.minimax_base_url),
            minimax_model=os.getenv("MINIMAX_MODEL", cls.minimax_model),
            minimax_api_key=os.getenv("MINIMAX_API_KEY", cls.minimax_api_key),
            #
            livekit_url=livekit_url,
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", cls.livekit_api_key),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", cls.livekit_api_secret),
            livekit_bind_port=int(os.getenv("LIVEKIT_BIND_PORT", str(cls.livekit_bind_port))),
            livekit_rtc_port=int(os.getenv("LIVEKIT_RTC_PORT", str(cls.livekit_rtc_port))),
            livekit_udp_port=int(os.getenv("LIVEKIT_UDP_PORT", str(cls.livekit_udp_port))),
            livekit_node_ip=os.getenv("LIVEKIT_NODE_IP", cls.livekit_node_ip),
            manage_livekit=_env_bool("MANAGE_LIVEKIT", _is_loopback(livekit_url)),
            #
            llama_base_url=llama_base_url,
            llama_model=os.getenv("LLAMA_MODEL", cls.llama_model),
            llama_api_key=os.getenv("LLAMA_API_KEY", cls.llama_api_key),
            llama_hf_repo=os.getenv("LLAMA_HF_REPO", cls.llama_hf_repo),
            llama_model_path=os.getenv("LLAMA_MODEL_PATH", cls.llama_model_path),
            llama_offline=_env_bool_opt("LLAMA_OFFLINE"),
            llama_model_alias=os.getenv("LLAMA_MODEL_ALIAS", cls.llama_model_alias),
            llama_ctx_size=int(os.getenv("LLAMA_CTX_SIZE", str(cls.llama_ctx_size))),
            llama_n_gpu_layers=int(os.getenv("LLAMA_N_GPU_LAYERS", str(cls.llama_n_gpu_layers))),
            llama_bind_port=int(os.getenv("LLAMA_BIND_PORT", str(cls.llama_bind_port))),
            manage_llama=(
                False
                if llm_provider == "minimax"
                else _env_bool("MANAGE_LLAMA", _is_loopback(llama_base_url))
            ),
            #
            stt_provider=stt_provider,
            stt_base_url=stt_base_url,
            stt_model=os.getenv("STT_MODEL", default_stt_model),
            stt_api_key=os.getenv("STT_API_KEY", cls.stt_api_key),
            stt_bind_port=int(os.getenv("STT_BIND_PORT", str(cls.stt_bind_port))),
            manage_stt=_env_bool("MANAGE_STT", _is_loopback(stt_base_url)),
            nemotron_model_name=os.getenv("NEMOTRON_MODEL_NAME", cls.nemotron_model_name),
            nemotron_model_id=os.getenv("NEMOTRON_MODEL_ID", cls.nemotron_model_id),
            whisper_model=os.getenv("WHISPER_MODEL", cls.whisper_model),
            #
            wake_word=_env_bool("WAKE_WORD", cls.wake_word),
            wake_word_model=os.getenv("WAKE_WORD_MODEL", cls.wake_word_model),
            wake_word_threshold=float(
                os.getenv("WAKE_WORD_THRESHOLD", str(cls.wake_word_threshold))
            ),
            #
            tts_provider=tts_provider,
            tts_base_url=tts_base_url,
            tts_voice=os.getenv("TTS_VOICE", cls.tts_voice),
            tts_api_key=os.getenv("TTS_API_KEY", cls.tts_api_key),
            tts_bind_port=int(os.getenv("TTS_BIND_PORT", str(cls.tts_bind_port))),
            manage_tts=(
                False
                if tts_provider == "minimax"
                else _env_bool("MANAGE_TTS", _is_loopback(tts_base_url))
            ),
            minimax_tts_model=os.getenv("MINIMAX_TTS_MODEL", cls.minimax_tts_model),
            minimax_tts_voice=os.getenv("MINIMAX_TTS_VOICE", cls.minimax_tts_voice),
            #
            device=os.getenv("DEVICE", cls.device).lower(),
            log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
        )

    def agent_env(self) -> dict[str, str]:
        """Environment variables to pass to the agent worker subprocess."""
        env: dict[str, str] = {
            "LIVEKIT_URL": self.livekit_url,
            "LIVEKIT_API_KEY": self.livekit_api_key,
            "LIVEKIT_API_SECRET": self.livekit_api_secret,
            "LLM_PROVIDER": self.llm_provider,
            "LLAMA_BASE_URL": self.llama_base_url,
            "LLAMA_MODEL": self.llama_model,
            "LLAMA_API_KEY": self.llama_api_key,
            "MINIMAX_BASE_URL": self.minimax_base_url,
            "MINIMAX_MODEL": self.minimax_model,
            "MINIMAX_API_KEY": self.minimax_api_key,
            "STT_PROVIDER": self.stt_provider,
            "STT_BASE_URL": self.stt_base_url,
            "STT_MODEL": self.stt_model,
            "STT_API_KEY": self.stt_api_key,
            "WAKE_WORD": "1" if self.wake_word else "0",
            "WAKE_WORD_MODEL": self.wake_word_model,
            "WAKE_WORD_THRESHOLD": str(self.wake_word_threshold),
            "TTS_PROVIDER": self.tts_provider,
            "TTS_BASE_URL": self.tts_base_url,
            "TTS_VOICE": self.tts_voice,
            "TTS_API_KEY": self.tts_api_key,
            "MINIMAX_TTS_MODEL": self.minimax_tts_model,
            "MINIMAX_TTS_VOICE": self.minimax_tts_voice,
        }
        return env

    @property
    def llm_model(self) -> str:
        return self.minimax_model if self.llm_provider == "minimax" else self.llama_model

    def to_public_dict(self) -> dict[str, Any]:
        """Read-only settings snapshot for ``GET /api/config`` (no secrets)."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "stt_provider": self.stt_provider,
            "tts_provider": self.tts_provider,
            "tts_voice": self.tts_voice,
            "minimax_tts_model": self.minimax_tts_model,
            "minimax_tts_voice": self.minimax_tts_voice,
            "wake_word": self.wake_word,
            "minimax_api_key_set": bool(self.minimax_api_key.strip()),
            "options": {
                "llm_providers": list(LLM_PROVIDERS),
                "stt_providers": list(STT_PROVIDERS),
                "tts_providers": list(TTS_PROVIDERS),
                "minimax_tts_models": list(MINIMAX_TTS_MODELS),
                "minimax_tts_model_labels": MINIMAX_TTS_MODEL_LABELS,
                "kokoro_voices": list(KOKORO_VOICES),
                "minimax_voices": list(MINIMAX_TTS_VOICES),
            },
        }

    def to_env_local(self) -> dict[str, str]:
        """Writable settings as env var name → value for ``.env.local``."""
        return {
            "LLM_PROVIDER": self.llm_provider,
            "STT_PROVIDER": self.stt_provider,
            "TTS_PROVIDER": self.tts_provider,
            "TTS_VOICE": self.tts_voice,
            "MINIMAX_TTS_MODEL": self.minimax_tts_model,
            "MINIMAX_TTS_VOICE": self.minimax_tts_voice,
            "WAKE_WORD": "1" if self.wake_word else "0",
        }

    def apply_settings(self, patch: dict[str, Any]) -> "Config":
        """Return a new config with validated settings merged in."""
        updated = replace(self)

        if "llm_provider" in patch:
            provider = str(patch["llm_provider"]).lower()
            if provider not in LLM_PROVIDERS:
                raise ValueError(f"invalid llm_provider: {provider}")
            updated.llm_provider = provider
            updated.manage_llama = (
                False
                if provider == "minimax"
                else _env_bool("MANAGE_LLAMA", _is_loopback(updated.llama_base_url))
            )

        if "stt_provider" in patch:
            provider = str(patch["stt_provider"]).lower()
            if provider not in STT_PROVIDERS:
                raise ValueError(f"invalid stt_provider: {provider}")
            updated.stt_provider = provider
            updated.stt_model = (
                "Systran/faster-whisper-small"
                if provider == "whisper"
                else "nemotron-speech-streaming"
            )

        if "tts_provider" in patch:
            provider = str(patch["tts_provider"]).lower()
            if provider not in TTS_PROVIDERS:
                raise ValueError(f"invalid tts_provider: {provider}")
            updated.tts_provider = provider
            updated.manage_tts = (
                False
                if provider == "minimax"
                else _env_bool("MANAGE_TTS", _is_loopback(updated.tts_base_url))
            )

        if "tts_voice" in patch:
            updated.tts_voice = str(patch["tts_voice"])

        if "minimax_tts_model" in patch:
            model = str(patch["minimax_tts_model"])
            if model not in MINIMAX_TTS_MODELS:
                raise ValueError(f"invalid minimax_tts_model: {model}")
            updated.minimax_tts_model = model

        if "minimax_tts_voice" in patch:
            updated.minimax_tts_voice = str(patch["minimax_tts_voice"])

        if "wake_word" in patch:
            updated.wake_word = bool(patch["wake_word"])

        return updated

    def persist_env_local(self, path: Path | str = ".env.local") -> None:
        """Write persistable settings to ``.env.local``, preserving other keys."""
        path = Path(path)
        new_vars = self.to_env_local()
        existing: dict[str, str] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, _, value = stripped.partition("=")
                existing[key.strip()] = value.strip()

        existing.update(new_vars)
        lines = [f"{key}={value}" for key, value in existing.items()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        for key, value in new_vars.items():
            os.environ[key] = value
