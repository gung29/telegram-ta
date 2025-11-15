from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field

from .models import GroupMode, MemberStatus


class PredictionRequest(BaseModel):
    text: str
    chat_id: Optional[int] = None


class PredictionResponse(BaseModel):
    prob_hate: float
    prob_nonhate: float
    pred: int
    label: str


class SettingsPayload(BaseModel):
    enabled: Optional[bool] = None
    threshold: Optional[float] = Field(None, ge=0, le=1)
    mode: Optional[GroupMode] = None
    retention_days: Optional[int] = Field(None, ge=1, le=90)


class SettingsResponse(BaseModel):
    chat_id: int
    enabled: bool
    threshold: float
    mode: GroupMode
    retention_days: int
    updated_at: datetime


class EventSchema(BaseModel):
    id: int
    chat_id: int
    user_id: Optional[int]
    username: Optional[str]
    prob_hate: float
    prob_nonhate: float
    action: str
    text: Optional[str]
    reason: Optional[str]
    created_at: datetime
    manual_label: Optional[str]
    manual_verified: bool = False
    manual_verified_at: Optional[datetime]

    class Config:
        from_attributes = True


class StatsResponse(BaseModel):
    chat_id: int
    window: str
    total_events: int
    blocked: int
    warned: int
    deleted: int
    top_offenders: List[str] = Field(default_factory=list)


class AdminTestRequest(BaseModel):
    text: str
    threshold: Optional[float] = None


class EventCreate(BaseModel):
    chat_id: int
    user_id: Optional[int] = None
    username: Optional[str] = None
    message_id: Optional[int] = None
    prob_hate: float
    prob_nonhate: float
    action: str
    text: Optional[str] = None
    reason: Optional[str] = None
    manual_label: Optional[str] = None
    manual_verified: Optional[bool] = None
    manual_verified_at: Optional[datetime] = None


class EventVerificationPayload(BaseModel):
    chat_id: int
    label: Literal["hate", "non-hate"]


class AdminEntry(BaseModel):
    id: int
    chat_id: int
    user_id: int
    added_at: datetime

    class Config:
        from_attributes = True


class AdminPayload(BaseModel):
    user_id: int


class MemberModerationSchema(BaseModel):
    id: int
    chat_id: int
    user_id: int
    username: Optional[str]
    status: MemberStatus
    reason: Optional[str]
    expires_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class MemberModerationPayload(BaseModel):
    user_id: int
    username: Optional[str] = None
    status: MemberStatus
    duration_minutes: Optional[int] = Field(None, ge=1, le=10080)
    reason: Optional[str] = Field(None, max_length=200)


class GroupSyncPayload(BaseModel):
    title: Optional[str] = None
    group_type: Optional[str] = None


class GroupSummary(BaseModel):
    chat_id: int
    enabled: bool
    threshold: float
    mode: GroupMode
    updated_at: datetime
    title: Optional[str] = None
    group_type: Optional[str] = None
    last_active: datetime


class TimelinePoint(BaseModel):
    date: datetime
    deleted: int
    warned: int
    blocked: int


class ActivityResponse(BaseModel):
    chat_id: int
    points: List[TimelinePoint] = Field(default_factory=list)


class UserActionSummary(BaseModel):
    user_id: int
    username: Optional[str]
    warnings_today: int
    mutes_total: int
    last_warning: Optional[datetime]
    last_mute: Optional[datetime]


class ActionResetPayload(BaseModel):
    action: Literal["warned", "muted"]
