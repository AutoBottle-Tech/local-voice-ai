"""Async process supervisor.

Spawns a list of child processes, line-prefixes their stdout/stderr through the
parent logger, polls per-child readiness URLs, and propagates SIGTERM/SIGINT to
all children for a clean shutdown. A child that dies before becoming ready is a
fatal error; a child that dies after becoming ready is restarted with linear
backoff up to a cap.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("supervisor")


@dataclass
class ChildSpec:
    name: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    ready_url: str | None = None
    # When set, readiness waits for this substring on stdout/stderr (e.g. LiveKit
    # agent worker registration) instead of marking ready immediately.
    ready_log_substring: str | None = None
    ready_timeout: float = 180.0
    max_restarts: int = 5


@dataclass
class _Child:
    spec: ChildSpec
    process: asyncio.subprocess.Process | None = None
    restart_count: int = 0
    ready: bool = False
    ready_log_event: asyncio.Event = field(default_factory=asyncio.Event)
    pump_task: asyncio.Task | None = None
    watch_task: asyncio.Task | None = None


class Supervisor:
    def __init__(self, specs: list[ChildSpec]) -> None:
        self._children: list[_Child] = [_Child(spec=spec) for spec in specs]
        self._stop_event = asyncio.Event()
        self._http: httpx.AsyncClient | None = None

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()

    def status(self) -> list[dict[str, object]]:
        """Per-child snapshot for the /api/status endpoint (first-boot UI)."""
        return [
            {
                "name": child.spec.name,
                "ready": child.ready,
                "running": bool(child.process and child.process.returncode is None),
                "restarts": child.restart_count,
            }
            for child in self._children
        ]

    async def start_all(self) -> None:
        """Spawn every child and wait for each to pass its readiness probe."""
        if not self._children:
            return

        self._http = httpx.AsyncClient(timeout=httpx.Timeout(2.0))

        # Spawn all children in parallel; each one's readiness wait is independent.
        await asyncio.gather(*(self._start(child) for child in self._children))
        await asyncio.gather(*(self._await_ready(child) for child in self._children))

    async def run_until_signal(self) -> int:
        """Wait until a signal arrives or any child exceeds restart budget."""
        loop = asyncio.get_running_loop()

        def _handle(signum: int) -> None:
            logger.info("received signal %s; shutting down", signal.Signals(signum).name)
            self._stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _handle, sig)
            except NotImplementedError:  # pragma: no cover — Windows
                pass

        await self._stop_event.wait()
        await self.shutdown()
        return 0

    async def reconfigure(self, new_specs: list[ChildSpec]) -> None:
        """Stop changed/removed children and start new/changed ones."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(2.0))

        new_by_name = {spec.name: spec for spec in new_specs}
        old_by_name = {child.spec.name: child for child in self._children}

        for name, child in list(old_by_name.items()):
            new_spec = new_by_name.get(name)
            if new_spec is None or _spec_changed(child.spec, new_spec):
                await self._terminate_one(child)

        rebuilt: list[_Child] = []
        for spec in new_specs:
            existing = old_by_name.get(spec.name)
            if (
                existing is not None
                and not _spec_changed(existing.spec, spec)
                and existing.process is not None
                and existing.process.returncode is None
            ):
                rebuilt.append(existing)
            else:
                rebuilt.append(_Child(spec=spec))

        self._children = rebuilt

        to_start = [
            child
            for child in self._children
            if child.process is None or child.process.returncode is not None
        ]
        if not to_start:
            return

        await asyncio.gather(*(self._start(child) for child in to_start))
        await asyncio.gather(*(self._await_ready(child) for child in to_start))

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Terminate all children. SIGTERM, wait, SIGKILL if needed."""
        self._stop_event.set()

        for child in self._children:
            if child.process and child.process.returncode is None:
                logger.info("[%s] terminating", child.spec.name)
                try:
                    child.process.terminate()
                except ProcessLookupError:
                    pass

        deadline = asyncio.get_running_loop().time() + timeout
        for child in self._children:
            if not child.process:
                continue
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            try:
                await asyncio.wait_for(child.process.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning("[%s] killing (did not exit in %.1fs)", child.spec.name, timeout)
                try:
                    child.process.kill()
                except ProcessLookupError:
                    pass
                await child.process.wait()

        for child in self._children:
            for task in (child.pump_task, child.watch_task):
                if task and not task.done():
                    task.cancel()

        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ---------------- internals ----------------

    async def _terminate_one(self, child: _Child, timeout: float = 10.0) -> None:
        for task in (child.pump_task, child.watch_task):
            if task and not task.done():
                task.cancel()
        child.pump_task = None
        child.watch_task = None

        if child.process and child.process.returncode is None:
            logger.info("[%s] terminating for reconfigure", child.spec.name)
            try:
                child.process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(child.process.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                try:
                    child.process.kill()
                except ProcessLookupError:
                    pass
                await child.process.wait()

        child.process = None
        child.ready = False
        child.ready_log_event = asyncio.Event()

    async def _start(self, child: _Child) -> None:
        child.ready_log_event = asyncio.Event()
        env = {**os.environ, **child.spec.env}
        logger.info("[%s] starting: %s", child.spec.name, " ".join(child.spec.argv))
        child.process = await asyncio.create_subprocess_exec(
            *child.spec.argv,
            cwd=child.spec.cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        child.pump_task = asyncio.create_task(
            self._pump_output(child),
            name=f"pump:{child.spec.name}",
        )
        child.watch_task = asyncio.create_task(
            self._watch_exit(child),
            name=f"watch:{child.spec.name}",
        )

    async def _pump_output(self, child: _Child) -> None:
        assert child.process is not None
        prefix = f"[{child.spec.name}]"

        ready_marker = child.spec.ready_log_substring

        async def pump(stream: asyncio.StreamReader | None, level: int) -> None:
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    return
                text = line.decode(errors="replace").rstrip()
                logger.log(level, "%s %s", prefix, text)
                if ready_marker and ready_marker in text and not child.ready_log_event.is_set():
                    child.ready_log_event.set()

        await asyncio.gather(
            pump(child.process.stdout, logging.INFO),
            pump(child.process.stderr, logging.WARNING),
        )

    async def _await_ready(self, child: _Child) -> None:
        if child.spec.ready_url is None and child.spec.ready_log_substring is None:
            child.ready = True
            return

        assert self._http is not None or child.spec.ready_url is None
        deadline = asyncio.get_running_loop().time() + child.spec.ready_timeout
        delay = 0.5

        while True:
            if self.stopping:
                raise RuntimeError(f"{child.spec.name}: aborted during readiness wait")
            if child.process and child.process.returncode is not None:
                raise RuntimeError(
                    f"{child.spec.name}: exited (rc={child.process.returncode}) before ready"
                )

            if child.spec.ready_log_substring and child.ready_log_event.is_set():
                child.ready = True
                logger.info("[%s] ready (log marker %r)", child.spec.name, child.spec.ready_log_substring)
                return

            if child.spec.ready_url is not None:
                try:
                    resp = await self._http.get(child.spec.ready_url)
                    if resp.status_code < 400:
                        child.ready = True
                        logger.info("[%s] ready", child.spec.name)
                        return
                except (httpx.RequestError, httpx.HTTPError):
                    pass

            now = asyncio.get_running_loop().time()
            if now >= deadline:
                if child.spec.ready_log_substring:
                    raise TimeoutError(
                        f"{child.spec.name}: log marker {child.spec.ready_log_substring!r} did not "
                        f"appear within {child.spec.ready_timeout:.0f}s"
                    )
                raise TimeoutError(
                    f"{child.spec.name}: readiness probe {child.spec.ready_url} did not "
                    f"succeed within {child.spec.ready_timeout:.0f}s"
                )
            await asyncio.sleep(min(delay, deadline - now))
            delay = min(delay * 1.5, 5.0)

    async def _watch_exit(self, child: _Child) -> None:
        assert child.process is not None
        rc = await child.process.wait()
        if self.stopping:
            return
        if not child.ready:
            return  # _await_ready will report
        logger.warning("[%s] exited (rc=%s); restart_count=%d", child.spec.name, rc, child.restart_count)

        if child.restart_count >= child.spec.max_restarts:
            logger.error(
                "[%s] exceeded max_restarts=%d; signalling supervisor shutdown",
                child.spec.name,
                child.spec.max_restarts,
            )
            self._stop_event.set()
            return

        child.restart_count += 1
        backoff = min(2.0 * child.restart_count, 10.0)
        await asyncio.sleep(backoff)
        if self.stopping:
            return

        child.ready = False
        await self._start(child)
        try:
            await self._await_ready(child)
        except Exception as exc:
            logger.error("[%s] failed to recover: %s", child.spec.name, exc)
            self._stop_event.set()


def _spec_changed(old: ChildSpec, new: ChildSpec) -> bool:
    return (
        old.argv != new.argv
        or old.env != new.env
        or old.cwd != new.cwd
        or old.ready_url != new.ready_url
        or old.ready_log_substring != new.ready_log_substring
    )


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
