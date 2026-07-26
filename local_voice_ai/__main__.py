"""Entry point: ``python -m local_voice_ai [serve|download-models|console]``.

The default ``serve`` command:
  1. Builds child specs based on the config (skipping any service whose base
     URL is external).
  2. Spawns all children, waits for readiness.
  3. Starts the FastAPI app (token route + static frontend) on the same loop.
  4. Blocks on SIGTERM/SIGINT, then shuts everything down cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from .api import build_app
from .config import Config
from .settings_manager import SettingsManager
from .specs import build_specs
from .supervisor import Supervisor, configure_logging

logger = logging.getLogger("main")


def _hf_hub_dir(env: dict[str, str]) -> Path:
    """The Hugging Face hub cache current llama-server downloads into."""
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "hub"
    if env.get("XDG_CACHE_HOME"):
        return Path(env["XDG_CACHE_HOME"]) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _dir_size(path: Path) -> int:
    """Total bytes under ``path`` (0 if missing). Cheap: /models holds few files."""
    total = 0
    if path.is_dir():
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    return total


def _hub_repo_dir(repo: str) -> str:
    """HF hub cache dir name for a repo id (tag/quant suffix stripped)."""
    return "models--" + repo.split(":", 1)[0].replace("/", "--")


def make_status_provider(supervisor: Supervisor, cfg: Config):
    """Wrap ``supervisor.status()`` with per-child download detail.

    Model downloads dominate first boot, so while a child isn't ready we
    report how many bytes its models occupy on disk. Totals aren't knowable
    up front (llama resolves the quant at runtime), so this is a growing
    byte count rather than a fake percentage.
    """
    hub = _hf_hub_dir(dict(os.environ))
    repo_for_child = {
        "llama": _hub_repo_dir(cfg.llama_hf_repo),
        "nemotron": _hub_repo_dir(cfg.nemotron_model_name),
        "whisper": _hub_repo_dir(cfg.whisper_model),
        "kokoro": _hub_repo_dir("hexgrad/Kokoro-82M"),
    }

    def status() -> list[dict[str, object]]:
        children = supervisor.status()
        for child in children:
            repo = repo_for_child.get(str(child["name"]))
            if child["ready"] or repo is None:
                continue
            size = _dir_size(hub / repo)
            if size > 1_000_000:  # only meaningful once a download has begun
                child["detail"] = (
                    f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"
                )
        return children

    return status


def _startup_line(children: list[dict[str, object]]) -> str:
    """One compact line per poll: ``llama … 1.2 GB | nemotron ✓ | …``"""
    parts = []
    for c in children:
        mark = "✓" if c["ready"] else "…"
        detail = f" {c['detail']}" if c.get("detail") else ""
        parts.append(f"{c['name']} {mark}{detail}")
    return " | ".join(parts)


async def _serve(cfg: Config) -> int:
    specs = build_specs(cfg)
    supervisor = Supervisor(specs)

    logger.info(
        "supervisor managing %d children (livekit=%s llama=%s stt=%s tts=%s)",
        len(specs),
        cfg.manage_livekit, cfg.manage_llama, cfg.manage_stt, cfg.manage_tts,
    )

    status_provider = make_status_provider(supervisor, cfg)
    settings_manager = SettingsManager(cfg=cfg, supervisor=supervisor)
    app = build_app(
        cfg,
        status_provider=status_provider,
        settings_manager=settings_manager,
    )
    uv_config = uvicorn.Config(
        app,
        host=cfg.web_host,
        port=cfg.web_port,
        log_level=cfg.log_level.lower(),
        access_log=False,
    )
    uv_server = uvicorn.Server(uv_config)

    # Start the web server BEFORE the children: first boot can spend a long
    # time downloading model weights, and the frontend polls /api/status to
    # show per-child progress instead of a dead page. run_until_signal also
    # starts now so SIGTERM/SIGINT during a slow startup aborts cleanly (the
    # stop event makes each pending readiness wait raise).
    web_task = asyncio.create_task(uv_server.serve(), name="web")
    sup_task = asyncio.create_task(supervisor.run_until_signal(), name="supervisor")
    startup_task = asyncio.create_task(supervisor.start_all(), name="startup")

    async def _report_startup() -> None:
        # A compact heartbeat so `docker compose up` shows startup at a glance
        # instead of a wall of interleaved child logs.
        while True:
            await asyncio.sleep(10)
            logger.info("starting: %s", _startup_line(status_provider()))

    reporter_task = asyncio.create_task(_report_startup(), name="startup-reporter")

    done, _ = await asyncio.wait(
        {web_task, sup_task, startup_task}, return_when=asyncio.FIRST_COMPLETED
    )
    reporter_task.cancel()

    if startup_task in done and startup_task.exception() is not None:
        logger.error("startup failed; shutting down", exc_info=startup_task.exception())
        uv_server.should_exit = True
        await supervisor.shutdown()
        for task in (web_task, sup_task):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        return 1

    if not startup_task.done():
        # web or supervisor exited first (signal during startup, port clash…)
        startup_task.cancel()
        try:
            await startup_task
        except (asyncio.CancelledError, Exception):
            pass
    elif startup_task in done:
        # The line first-time users are looking for — make it unmissable.
        logger.info(
            "\n\n"
            "  ┌────────────────────────────────────────────────┐\n"
            "  │                                                │\n"
            "  │   ✅  local-voice-ai is ready                  │\n"
            "  │                                                │\n"
            "  │   👉  Open  http://localhost:%-5d             │\n"
            "  │       and click “Start call”                   │\n"
            "  │                                                │\n"
            "  └────────────────────────────────────────────────┘\n",
            cfg.web_port,
        )
        done, _ = await asyncio.wait(
            {web_task, sup_task}, return_when=asyncio.FIRST_COMPLETED
        )

    # Whatever finished first triggers a coordinated shutdown.
    uv_server.should_exit = True
    if not sup_task.done():
        await supervisor.shutdown()
    for task in (web_task, sup_task):
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    return 0


def _download_models(cfg: Config) -> int:
    """Pre-download VAD, turn-detector, Nemotron weights so first run is warm."""
    logger.info("downloading agent prewarm models (silero VAD, turn detector)")
    # Reuse livekit-agents' built-in download-files command
    import subprocess
    rc = subprocess.call([sys.executable, "-m", "local_voice_ai.agent", "download-files"])
    if rc != 0:
        return rc

    if cfg.manage_stt and cfg.stt_provider == "nemotron":
        logger.info("downloading nemotron model %s", cfg.nemotron_model_name)
        import nemo.collections.asr as nemo_asr  # type: ignore[import]
        nemo_asr.models.ASRModel.from_pretrained(cfg.nemotron_model_name)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="local_voice_ai")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="run the full supervised stack (default)")
    sub.add_parser("download-models", help="pre-download model weights")
    sub.add_parser("console", help="run the agent in interactive console mode")

    args = parser.parse_args(argv)
    load_dotenv()
    load_dotenv(".env.local")
    cfg = Config.from_env()
    configure_logging(cfg.log_level)

    cmd = args.cmd or "serve"
    if cmd == "serve":
        return asyncio.run(_serve(cfg))
    if cmd == "download-models":
        return _download_models(cfg)
    if cmd == "console":
        os.execv(
            sys.executable,
            [sys.executable, "-m", "local_voice_ai.agent", "console"],
        )
    parser.error(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
