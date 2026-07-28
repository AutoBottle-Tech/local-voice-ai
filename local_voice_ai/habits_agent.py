"""Habits voice coach — habits-api context and unified tool execution."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from livekit.agents import Agent, RunContext, function_tool

from .llm_filter import stream_llm_with_thinking_filter

logger = logging.getLogger("habits-agent")


def _habits_api_url() -> str:
    return os.getenv("HABITS_API_URL", "http://host.docker.internal:8787").rstrip("/")


def _habits_bearer() -> str:
    return os.getenv("HABITS_INTERNAL_BEARER", "")

HABITS_BASE_INSTRUCTIONS = """You are the Habits coach — a concise voice assistant for food tracking,
habit accountability, and scheduling. The user speaks via phone.

Use tools for all data actions. When food is vague, search first and ask ONE clarifying question
before logging. Before scheduling, list calendar events to check conflicts.
Keep replies short and spoken-friendly — no markdown, emojis, or bullet lists.
When logging food, confirm what was recorded and today's protein total if available."""


async def _habits_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    headers = kwargs.pop("headers", {})
    bearer = _habits_bearer()
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.request(
            method, f"{_habits_api_url()}{path}", headers=headers, **kwargs
        )
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return {}


async def _execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    data = await _habits_request(
        "POST",
        "/api/agent/tools/execute",
        json={"name": name, "args": args},
    )
    result = data.get("result") or {}
    if isinstance(result, dict) and result.get("error"):
        raise ValueError(str(result["error"]))
    return result if isinstance(result, dict) else {"message": str(result)}


def _spoken_result(result: dict[str, Any], fallback: str) -> str:
    if result.get("message"):
        return str(result["message"])
    if result.get("summary"):
        summary = result["summary"]
        if isinstance(summary, dict):
            protein = summary.get("protein_g", "?")
            target = summary.get("protein_target_g", "?")
            return f"Today: {protein} grams protein of {target} target."
    return fallback


async def build_habits_context() -> str:
    parts: list[str] = []

    try:
        summary = await _habits_request("GET", "/api/future-self/summary")
        if s := summary.get("summary"):
            parts.append(f"Daily overview: {s}")
    except Exception as exc:
        logger.warning("future-self summary unavailable: %s", exc)

    try:
        food = await _habits_request("GET", "/api/food/today")
        protein = food.get("protein_g", "?")
        target = food.get("protein_target_g", "?")
        calories = food.get("calories", "?")
        items = food.get("items") or []
        recent = ", ".join(
            f"{i.get('quantity_g')}g {i.get('food')}" for i in items[-3:]
        )
        parts.append(
            f"Nutrition today: {protein}g protein of {target}g target, {calories} calories. "
            f"Recent meals: {recent or 'nothing logged yet'}."
        )
    except Exception as exc:
        logger.warning("food today unavailable: %s", exc)

    try:
        habits = await _habits_request("GET", "/api/habits/today")
        metrics = habits.get("metrics") or {}
        metric_str = ", ".join(f"{k}={v}h" for k, v in metrics.items() if v is not None)
        if metric_str:
            parts.append(
                f"Habit tracker today ({habits.get('weekday', '')}): {metric_str}."
            )
    except Exception as exc:
        logger.warning("habits today unavailable: %s", exc)

    try:
        cal = await _habits_request("GET", "/api/calendar/today")
        events = cal.get("events") or []
        if events:
            ev_str = "; ".join(
                f"{e.get('summary')} at {e.get('start')}" for e in events[:5]
            )
            parts.append(f"Calendar today: {ev_str}.")
        else:
            parts.append("Calendar today: no events scheduled.")
    except Exception as exc:
        logger.warning("calendar today unavailable: %s", exc)

    if not parts:
        return (
            "User data is temporarily unavailable; you can still help log food "
            "and schedule events."
        )
    return "\n".join(parts)


class HabitsAssistant(Agent):
    def __init__(self, context_block: str = "") -> None:
        instructions = HABITS_BASE_INSTRUCTIONS
        if context_block:
            instructions += f"\n\nCurrent user context:\n{context_block}"
        super().__init__(instructions=instructions)

    @function_tool()
    async def search_food(self, context: RunContext, query: str) -> str:
        """Search the food database before logging."""
        try:
            result = await _execute_tool("search_food", {"query": query})
            results = result.get("results") or []
            if not results:
                return f"No foods found matching {query}."
            names = ", ".join(r.get("name", "?") for r in results[:5])
            return f"Found: {names}."
        except Exception as exc:
            return f"Food search failed: {exc}"

    @function_tool()
    async def log_meal(
        self,
        context: RunContext,
        description: str,
        meal_type: str = "other",
    ) -> str:
        """Log food from a natural language description."""
        try:
            result = await _execute_tool(
                "log_food", {"description": description, "meal_type": meal_type}
            )
            return _spoken_result(result, "Meal logged.")
        except Exception as exc:
            logger.exception("log_meal failed")
            return f"Could not log meal: {exc}"

    @function_tool()
    async def log_food_item(
        self,
        context: RunContext,
        food_name: str,
        quantity_g: float,
        meal_type: str = "other",
    ) -> str:
        """Log a specific food with exact grams."""
        try:
            result = await _execute_tool(
                "log_food_item",
                {
                    "food_name": food_name,
                    "quantity_g": quantity_g,
                    "meal_type": meal_type,
                },
            )
            return _spoken_result(result, "Food logged.")
        except Exception as exc:
            return f"Could not log food: {exc}"

    @function_tool()
    async def get_food_today(self, context: RunContext) -> str:
        """Get today's protein, calories, and logged food items."""
        try:
            result = await _execute_tool("get_food_today", {})
            return _spoken_result(result, "Fetched today's nutrition.")
        except Exception as exc:
            return f"Could not fetch nutrition: {exc}"

    @function_tool()
    async def get_calendar_today(self, context: RunContext) -> str:
        """List today's calendar events."""
        try:
            result = await _execute_tool("list_calendar_events", {})
            events = result.get("events") or []
            if not events:
                return "No events on the calendar today."
            parts = [f"{e.get('summary')} at {e.get('start')}" for e in events]
            return "Today: " + "; ".join(parts) + "."
        except Exception as exc:
            return f"Could not fetch calendar: {exc}"

    @function_tool()
    async def create_calendar_event(
        self,
        context: RunContext,
        title: str,
        start_iso: str,
        duration_minutes: int = 60,
        description: str = "",
        location: str = "",
    ) -> str:
        """Schedule a Google Calendar event."""
        try:
            args: dict[str, Any] = {
                "title": title,
                "start": start_iso,
                "duration_minutes": duration_minutes,
            }
            if description:
                args["description"] = description
            if location:
                args["location"] = location
            result = await _execute_tool("create_event", args)
            summary = result.get("summary", title)
            start = result.get("start", start_iso)
            return f"Scheduled {summary} at {start}."
        except Exception as exc:
            logger.exception("create_calendar_event failed")
            return f"Could not schedule event: {exc}"

    @function_tool()
    async def update_calendar_event(
        self,
        context: RunContext,
        event_id: str,
        title: str = "",
        start_iso: str = "",
        duration_minutes: int = 0,
        description: str = "",
        location: str = "",
    ) -> str:
        """Update or reschedule a calendar event."""
        try:
            args: dict[str, Any] = {"event_id": event_id}
            if title:
                args["title"] = title
            if start_iso:
                args["start"] = start_iso
            if duration_minutes:
                args["duration_minutes"] = duration_minutes
            if description:
                args["description"] = description
            if location:
                args["location"] = location
            result = await _execute_tool("update_calendar_event", args)
            return f"Updated {result.get('summary', 'event')}."
        except Exception as exc:
            return f"Could not update event: {exc}"

    @function_tool()
    async def delete_calendar_event(self, context: RunContext, event_id: str) -> str:
        """Delete a calendar event."""
        try:
            await _execute_tool("delete_calendar_event", {"event_id": event_id})
            return "Event removed from calendar."
        except Exception as exc:
            return f"Could not delete event: {exc}"

    @function_tool()
    async def update_habit(self, context: RunContext, metric: str, value: float) -> str:
        """Update today's habit metric hours."""
        try:
            await _execute_tool("update_habit", {"metric": metric, "value": value})
            return f"Updated {metric} to {value} hours."
        except Exception as exc:
            return f"Could not update habit: {exc}"

    @function_tool()
    async def add_card(
        self,
        context: RunContext,
        card_type: str,
        title: str,
        body: str,
    ) -> str:
        """Add a sickness, notes, or strategy card."""
        try:
            await _execute_tool(
                "add_card",
                {"card_type": card_type, "title": title, "body": body},
            )
            return f"Added {card_type} card: {title}."
        except Exception as exc:
            return f"Could not add card: {exc}"

    async def llm_node(
        self,
        chat_ctx,
        tools,
        model_settings,
    ):
        async for chunk in stream_llm_with_thinking_filter(
            self, chat_ctx, tools, model_settings
        ):
            yield chunk
