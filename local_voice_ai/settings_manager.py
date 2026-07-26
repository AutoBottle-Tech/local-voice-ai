"""Runtime settings updates: persist config and reconfigure supervisor children."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .config import Config
from .specs import build_specs
from .supervisor import Supervisor

logger = logging.getLogger("settings")


@dataclass
class SettingsManager:
    cfg: Config
    supervisor: Supervisor
    _lock: asyncio.Lock | None = None

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()

    async def apply(self, patch: dict[str, Any]) -> Config:
        """Validate, persist, and hot-reconfigure affected children."""
        assert self._lock is not None
        async with self._lock:
            updated = self.cfg.apply_settings(patch)
            updated.persist_env_local()
            new_specs = build_specs(updated)
            logger.info(
                "reconfiguring stack: llm=%s stt=%s tts=%s wake_word=%s",
                updated.llm_provider,
                updated.stt_provider,
                updated.tts_provider,
                updated.wake_word,
            )
            await self.supervisor.reconfigure(new_specs)
            self.cfg = updated
            return updated
