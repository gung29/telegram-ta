from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from sqlalchemy import BigInteger, Boolean, CheckConstraint, Column, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .config import settings
from .database import Base, engine
from .migrations import apply_schema_patches
from .timezone import now_local


class GroupMode(str, Enum):
    precision = "precision"
    balanced = "balanced"
    recall = "recall"


class GroupSettings(Base):
    __tablename__ = "group_settings"

    chat_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    threshold: Mapped[float] = mapped_column(Float, default=settings.default_threshold)
    mode: Mapped[GroupMode] = mapped_column(SAEnum(GroupMode), default=GroupMode.balanced)
    retention_days: Mapped[int] = mapped_column(Integer, default=settings.retention_days)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=now_local)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    group_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    last_active: Mapped[datetime] = mapped_column(DateTime, default=now_local)


class ModerationEvent(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prob_hate: Mapped[float] = mapped_column(Float)
    prob_nonhate: Mapped[float] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String(64))
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local, index=True)
    manual_label: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    manual_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    manual_verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class GroupAdmin(Base):
    __tablename__ = "group_admins"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("group_settings.chat_id", ondelete="CASCADE"))
    user_id: Mapped[int] = mapped_column(BigInteger)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=now_local)

    __table_args__ = (
        UniqueConstraint("chat_id", "user_id", name="uq_group_admin_member"),
        CheckConstraint("user_id > 0", name="ck_admin_user_positive"),
    )


class MemberStatus(str, Enum):
    muted = "muted"
    banned = "banned"


class MemberModeration(Base):
    __tablename__ = "member_moderations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("group_settings.chat_id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(BigInteger)
    username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[MemberStatus] = mapped_column(SAEnum(MemberStatus))
    reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_local)

    __table_args__ = (
        UniqueConstraint("chat_id", "user_id", "status", name="uq_member_status"),
        CheckConstraint("user_id > 0", name="ck_member_user_positive"),
    )


def ensure_tables() -> None:
    Base.metadata.create_all(bind=engine)
    apply_schema_patches()


def prune_old_events(retention_days: int | None = None) -> int:
    """Delete events older than the configured retention."""
    from sqlalchemy import delete
    from sqlalchemy.orm import Session

    cutoff_days = retention_days or settings.retention_days
    cutoff = now_local() - timedelta(days=cutoff_days)

    with Session(engine) as session:
        stmt = delete(ModerationEvent).where(ModerationEvent.created_at < cutoff)
        result = session.execute(stmt)
        session.commit()
        return result.rowcount or 0
