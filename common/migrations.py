from __future__ import annotations

from typing import Set

from sqlalchemy import inspect, text

from .database import engine


def _get_columns(table_name: str) -> Set[str]:
    inspector = inspect(engine)
    return {column["name"] for column in inspector.get_columns(table_name)}


def _add_column(table: str, ddl: str) -> None:
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))


def apply_schema_patches() -> None:
    """Lightweight, idempotent migrations for existing SQLite installs."""
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    if "group_settings" in tables:
        columns = _get_columns("group_settings")
        if "title" not in columns:
            _add_column("group_settings", "title VARCHAR(255)")
        if "group_type" not in columns:
            _add_column("group_settings", "group_type VARCHAR(32)")
        if "last_active" not in columns:
            _add_column("group_settings", "last_active DATETIME")
            with engine.begin() as connection:
                connection.execute(
                    text("UPDATE group_settings SET last_active = COALESCE(last_updated, CURRENT_TIMESTAMP)")
                )

    if "events" in tables:
        columns = _get_columns("events")
        if "reason" not in columns:
            _add_column("events", "reason VARCHAR(255)")
        if "manual_label" not in columns:
            _add_column("events", "manual_label VARCHAR(32)")
        if "manual_verified" not in columns:
            _add_column("events", "manual_verified BOOLEAN DEFAULT 0")
        if "manual_verified_at" not in columns:
            _add_column("events", "manual_verified_at DATETIME")

    if "member_moderations" in tables:
        columns = _get_columns("member_moderations")
        if "expires_at" not in columns:
            _add_column("member_moderations", "expires_at DATETIME")

    if "action_counter_resets" not in tables:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE action_counter_resets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id BIGINT NOT NULL,
                        user_id BIGINT NOT NULL,
                        action VARCHAR(32) NOT NULL,
                        reset_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(chat_id, user_id, action)
                    )
                    """
                )
            )
