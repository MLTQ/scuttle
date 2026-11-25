"""Scuttle Hermes plugin: fast 0.93 GiB swarm search over Markdown piles, dataswamps, and SQLite DBs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tools import Client, Settings, register_tools

__all__ = ["register"]


def register(ctx: Any) -> None:
    settings = Settings.from_context(ctx)
    client = Client(settings)
    register_tools(ctx, client)
    skill_path = Path(__file__).parent / "skills" / "search" / "SKILL.md"
    if skill_path.exists():
        ctx.register_skill(
            name="scuttle-search",
            path=skill_path,
            description=(
                "Use when searching a user's Markdown/Obsidian pile, messy dataswamp "
                "(PDFs/HTML/code/images), read-only SQLite database, or linked document graph with Scuttle."
            ),
        )
