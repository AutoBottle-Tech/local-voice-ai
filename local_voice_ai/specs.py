"""Child process specs for the supervisor, built from ``Config``."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from .config import Config
from .supervisor import ChildSpec

logger = logging.getLogger("main")


def _llama_cache_dir(env: dict[str, str]) -> Path:
    if env.get("LLAMA_CACHE"):
        return Path(env["LLAMA_CACHE"])
    if env.get("XDG_CACHE_HOME"):
        return Path(env["XDG_CACHE_HOME"]) / "llama.cpp"
    return Path.home() / ".cache" / "llama.cpp"


def _hf_hub_dir(env: dict[str, str]) -> Path:
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "hub"
    if env.get("XDG_CACHE_HOME"):
        return Path(env["XDG_CACHE_HOME"]) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _llama_repo_cached(repo: str, env: dict[str, str]) -> bool:
    spec, tag = [*repo.rsplit(":", 1), "latest"][:2]
    hub_repo = _hf_hub_dir(env) / f"models--{spec.replace('/', '--')}"
    if hub_repo.is_dir():
        pattern = f"*{tag}*.gguf" if tag != "latest" else "*.gguf"
        if any(hub_repo.glob(f"snapshots/*/{pattern}")):
            return True
    cache = _llama_cache_dir(env)
    if not cache.is_dir():
        return False
    manifest = cache / f"manifest={spec.replace('/', '=')}={tag}.json"
    if manifest.is_file():
        return True
    prefix = spec.replace("/", "_")
    return any(p.suffix == ".gguf" for p in cache.glob(f"{prefix}*.gguf"))


def build_specs(cfg: Config) -> list[ChildSpec]:
    specs: list[ChildSpec] = []
    py = sys.executable

    if cfg.manage_livekit:
        livekit_bin = os.getenv("LIVEKIT_BIN", "livekit-server")
        specs.append(
            ChildSpec(
                name="livekit",
                argv=[
                    livekit_bin,
                    "--dev",
                    "--bind", "0.0.0.0",
                    "--port", str(cfg.livekit_bind_port),
                    "--rtc.tcp_port", str(cfg.livekit_rtc_port),
                    "--udp-port", str(cfg.livekit_udp_port),
                    "--node-ip", cfg.livekit_node_ip,
                ],
                ready_url=None,
                ready_timeout=30.0,
            )
        )

    if cfg.manage_llama:
        llama_bin = os.getenv("LLAMA_BIN", "llama-server")
        llama_env = {
            "HF_HOME": os.getenv("HF_HOME", "/models"),
            "XDG_CACHE_HOME": os.getenv("XDG_CACHE_HOME", "/models"),
        }
        if cfg.llama_model_path:
            model_argv = ["-m", cfg.llama_model_path]
        else:
            model_argv = ["--hf-repo", cfg.llama_hf_repo]
        if cfg.llama_offline is not None:
            offline = cfg.llama_offline
        elif cfg.llama_model_path:
            offline = False
        else:
            offline = _llama_repo_cached(cfg.llama_hf_repo, llama_env)
            if offline:
                logger.info("llama: %s found in cache; starting --offline", cfg.llama_hf_repo)
        specs.append(
            ChildSpec(
                name="llama",
                argv=[
                    llama_bin,
                    "--host", "127.0.0.1",
                    "--port", str(cfg.llama_bind_port),
                    *model_argv,
                    *(["--offline"] if offline else []),
                    "--alias", cfg.llama_model_alias,
                    "--ctx-size", str(cfg.llama_ctx_size),
                    "--n-gpu-layers", str(cfg.llama_n_gpu_layers),
                    "--reasoning", "off",
                ],
                env=llama_env,
                ready_url=f"http://127.0.0.1:{cfg.llama_bind_port}/v1/models",
                ready_timeout=900.0,
            )
        )

    if cfg.manage_stt:
        if cfg.stt_provider == "whisper":
            specs.append(
                ChildSpec(
                    name="whisper",
                    argv=[
                        py, "-m", "local_voice_ai.services.whisper.server",
                        "--host", "127.0.0.1",
                        "--port", str(cfg.stt_bind_port),
                    ],
                    env={
                        "WHISPER_MODEL": cfg.whisper_model,
                        "DEVICE": cfg.device,
                    },
                    ready_url=f"http://127.0.0.1:{cfg.stt_bind_port}/health",
                    ready_timeout=600.0,
                )
            )
        else:
            specs.append(
                ChildSpec(
                    name="nemotron",
                    argv=[
                        py, "-m", "local_voice_ai.services.nemotron.server",
                        "--host", "127.0.0.1",
                        "--port", str(cfg.stt_bind_port),
                    ],
                    env={
                        "NEMOTRON_MODEL_NAME": cfg.nemotron_model_name,
                        "NEMOTRON_MODEL_ID": cfg.nemotron_model_id,
                        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                    },
                    ready_url=f"http://127.0.0.1:{cfg.stt_bind_port}/health",
                    ready_timeout=600.0,
                )
            )

    if cfg.manage_tts:
        specs.append(
            ChildSpec(
                name="kokoro",
                argv=[
                    py, "-m", "local_voice_ai.services.kokoro.server",
                    "--host", "127.0.0.1",
                    "--port", str(cfg.tts_bind_port),
                ],
                ready_url=f"http://127.0.0.1:{cfg.tts_bind_port}/v1/models",
                ready_timeout=600.0,
            )
        )

    specs.append(
        ChildSpec(
            name="agent",
            argv=[py, "-m", "local_voice_ai.agent", "start"],
            env=cfg.agent_env(),
            ready_log_substring="registered worker",
            ready_timeout=120.0,
        )
    )

    return specs
