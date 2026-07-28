"""FastAPI app served from the supervisor process.

Two responsibilities:
  1. ``POST /api/connection-details`` — mints a LiveKit access token. This is
     the Python port of ``frontend/app/api/connection-details/route.ts``.
  2. ``GET /*`` — serves the statically-exported Next.js frontend, when
     ``Config.frontend_dir`` is set.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from livekit import api as lk_api

from .config import Config

if TYPE_CHECKING:
    from .settings_manager import SettingsManager

logger = logging.getLogger("api")


def _mint_token(cfg: Config, agent_name: str | None) -> dict[str, Any]:
    participant_name = "user"
    participant_identity = f"voice_assistant_user_{random.randint(0, 9999)}"
    room_name = f"voice_assistant_room_{random.randint(0, 9999)}"

    token = (
        lk_api.AccessToken(cfg.livekit_api_key, cfg.livekit_api_secret)
        .with_identity(participant_identity)
        .with_name(participant_name)
        .with_ttl(timedelta(minutes=15))
        .with_grants(
            lk_api.VideoGrants(
                room=room_name,
                room_join=True,
                can_publish=True,
                can_publish_data=True,
                can_subscribe=True,
            )
        )
    )

    if agent_name:
        token = token.with_room_config(
            lk_api.RoomConfiguration(agents=[lk_api.RoomAgentDispatch(agent_name=agent_name)])
        )

    return {
        "serverUrl": cfg.livekit_url,
        "roomName": room_name,
        "participantName": participant_name,
        "participantToken": token.to_jwt(),
    }


def build_app(
    cfg: Config,
    status_provider: Callable[[], list[dict[str, Any]]] | None = None,
    settings_manager: "SettingsManager | None" = None,
) -> FastAPI:
    app = FastAPI(title="local-voice-ai", version="0.1.0")

    def current_cfg() -> Config:
        if settings_manager is not None:
            return settings_manager.cfg
        return cfg

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        """Per-child readiness, polled by the frontend's first-boot splash.

        The web server starts before the children are ready (first boot can
        spend a long time downloading model weights), so this is how the UI
        knows whether the stack is usable yet.
        """
        children = status_provider() if status_provider is not None else []
        active = current_cfg()
        return {
            "ready": all(c["ready"] for c in children),
            "children": children,
            "wake_word": active.wake_word,
        }

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        return current_cfg().to_public_dict()

    @app.put("/api/settings")
    async def put_settings(request: Request) -> JSONResponse:
        if settings_manager is None:
            raise HTTPException(status_code=501, detail="settings not available")
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")
        try:
            await settings_manager.apply(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("settings apply failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse({"restarting": True})

    @app.post("/api/connection-details")
    async def connection_details(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}

        agent_name: str | None = None
        try:
            agent_name = body.get("room_config", {}).get("agents", [{}])[0].get("agent_name")
        except (AttributeError, IndexError, TypeError):
            agent_name = None

        # Stale static frontends may omit room_config; match the habits worker registration.
        if not agent_name and os.getenv("AGENT_PROFILE", "").lower() == "habits":
            agent_name = "habits"

        try:
            data = _mint_token(current_cfg(), agent_name)
        except Exception as exc:
            logger.exception("token minting failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    if cfg.frontend_dir:
        settings_html = f"{cfg.frontend_dir}/settings.html"

        @app.get("/settings")
        async def settings_page() -> FileResponse:
            return FileResponse(settings_html)

        # SPA-style: serve static export, falling back to index.html for unknown paths.
        static = StaticFiles(directory=cfg.frontend_dir, html=True)

        @app.get("/{path:path}")
        async def spa(path: str, request: Request) -> Any:
            for candidate in (path or "index.html", f"{path}.html" if path else None):
                if not candidate:
                    continue
                try:
                    return await static.get_response(candidate, request.scope)
                except Exception:
                    continue
            return FileResponse(f"{cfg.frontend_dir}/index.html")

    return app
