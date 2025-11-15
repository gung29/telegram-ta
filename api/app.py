from __future__ import annotations

import csv
import io
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Union

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from sqlalchemy import case, desc, func
from sqlalchemy.orm import Session

from common.config import settings as app_settings
from common.database import get_db
from common.models import (
    GroupAdmin,
    GroupMode,
    GroupSettings,
    MemberModeration,
    MemberStatus,
    ModerationEvent,
    ActionCounterReset,
    ensure_tables,
)
from common.timezone import LOCAL_TIMEZONE, day_bounds, now_local
from common.schemas import (
    AdminTestRequest,
    AdminEntry,
    AdminPayload,
    ActivityResponse,
    EventCreate,
    EventSchema,
    EventVerificationPayload,
    GroupSummary,
    GroupSyncPayload,
    MemberModerationPayload,
    MemberModerationSchema,
    PredictionRequest,
    PredictionResponse,
    SettingsPayload,
    SettingsResponse,
    StatsResponse,
    UserActionSummary,
    ActionResetPayload,
)

from .model_loader import classifier

app = FastAPI(title="Hate Guard Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=app_settings.cors_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

PREDICTION_COUNTER = Counter("hate_guard_predictions_total", "Total number of predictions served")
PREDICTION_LATENCY = Histogram("hate_guard_prediction_latency_seconds", "Latency of prediction endpoint")
API_HEALTH = Gauge("hate_guard_api_status", "API health flag", multiprocess_mode="livesum")
EPOCH_START = datetime(1970, 1, 1, tzinfo=LOCAL_TIMEZONE)

ACTION_IGNORE = {"allowed", "bypassed_admin"}
ACTION_WARNED = {"warned", "muted"}
ACTION_BLOCKED = {"blocked", "banned"}


def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    if x_api_key != app_settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def _get_or_create_settings(db: Session, chat_id: int) -> GroupSettings:
    settings = db.get(GroupSettings, chat_id)
    if not settings:
        settings = GroupSettings(
            chat_id=chat_id,
            enabled=True,
            threshold=app_settings.default_threshold,
            mode=GroupMode.balanced,
            retention_days=app_settings.retention_days,
        )
        db.add(settings)
        db.commit()
        db.refresh(settings)
        # seed default admins
        for admin_id in app_settings.admin_ids:
            exists = (
                db.query(GroupAdmin)
                .filter(GroupAdmin.chat_id == chat_id, GroupAdmin.user_id == admin_id)
                .first()
            )
            if not exists:
                db.add(GroupAdmin(chat_id=chat_id, user_id=admin_id))
        db.commit()
    return settings


def _serialize_group(settings_row: GroupSettings) -> GroupSummary:
    return GroupSummary(
        chat_id=settings_row.chat_id,
        enabled=settings_row.enabled,
        threshold=settings_row.threshold,
        mode=settings_row.mode,
        updated_at=settings_row.last_updated,
        title=settings_row.title,
        group_type=settings_row.group_type,
        last_active=settings_row.last_active,
    )


def _require_group_chat(chat_id: int) -> None:
    if chat_id >= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chat_id harus berupa ID grup Telegram (negatif)")


def _normalize_day(value: Union[str, datetime, date]) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    return parsed.date()


@app.on_event("startup")
def on_startup() -> None:
    ensure_tables()
    API_HEALTH.set(1)


@app.on_event("shutdown")
def on_shutdown() -> None:
    API_HEALTH.set(0)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "model_loaded": str(bool(classifier.session))}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, _: None = Depends(require_api_key)) -> PredictionResponse:
    start = time.perf_counter()
    prob_hate, prob_nonhate, pred = classifier.predict(payload.text)
    latency = time.perf_counter() - start
    PREDICTION_COUNTER.inc()
    PREDICTION_LATENCY.observe(latency)
    label = "hate" if prob_hate >= 0.5 else "non-hate"
    return PredictionResponse(prob_hate=prob_hate, prob_nonhate=prob_nonhate, pred=pred, label=label)


@app.get("/admin/settings/{chat_id}", response_model=SettingsResponse)
def get_settings_endpoint(
    chat_id: int,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> SettingsResponse:
    _require_group_chat(chat_id)
    settings_row = _get_or_create_settings(db, chat_id)
    return SettingsResponse(
        chat_id=settings_row.chat_id,
        enabled=settings_row.enabled,
        threshold=settings_row.threshold,
        mode=settings_row.mode,
        retention_days=settings_row.retention_days,
        updated_at=settings_row.last_updated,
    )


@app.get("/admin/groups", response_model=List[GroupSummary])
def list_groups(
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> List[GroupSummary]:
    groups = (
        db.query(GroupSettings)
        .filter(GroupSettings.chat_id < 0)
        .order_by(desc(GroupSettings.last_active))
        .all()
    )
    return [_serialize_group(group) for group in groups]


@app.post("/admin/groups/{chat_id}/sync", response_model=GroupSummary)
def sync_group(
    chat_id: int,
    payload: GroupSyncPayload,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> GroupSummary:
    _require_group_chat(chat_id)
    settings_row = _get_or_create_settings(db, chat_id)
    if payload.title:
        settings_row.title = payload.title
    if payload.group_type:
        settings_row.group_type = payload.group_type
    settings_row.last_active = now_local()
    db.add(settings_row)
    db.commit()
    db.refresh(settings_row)
    return _serialize_group(settings_row)


@app.post("/admin/settings/{chat_id}", response_model=SettingsResponse)
def update_settings_endpoint(
    chat_id: int,
    payload: SettingsPayload,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> SettingsResponse:
    _require_group_chat(chat_id)
    settings_row = _get_or_create_settings(db, chat_id)
    if payload.enabled is not None:
        settings_row.enabled = payload.enabled
    if payload.threshold is not None:
        settings_row.threshold = payload.threshold
    if payload.mode is not None:
        settings_row.mode = payload.mode
    if payload.retention_days is not None:
        settings_row.retention_days = payload.retention_days
    settings_row.last_updated = now_local()
    db.add(settings_row)
    db.commit()
    db.refresh(settings_row)
    return SettingsResponse(
        chat_id=settings_row.chat_id,
        enabled=settings_row.enabled,
        threshold=settings_row.threshold,
        mode=settings_row.mode,
        retention_days=settings_row.retention_days,
        updated_at=settings_row.last_updated,
    )


@app.get("/admin/groups/{chat_id}/admins", response_model=List[AdminEntry])
def list_group_admins(
    chat_id: int,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> List[AdminEntry]:
    _require_group_chat(chat_id)
    admins = (
        db.query(GroupAdmin)
        .filter(GroupAdmin.chat_id == chat_id)
        .order_by(GroupAdmin.added_at)
        .all()
    )
    return admins


@app.post("/admin/groups/{chat_id}/admins", response_model=AdminEntry)
def add_group_admin(
    chat_id: int,
    payload: AdminPayload,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> AdminEntry:
    _require_group_chat(chat_id)
    _get_or_create_settings(db, chat_id)
    existing = (
        db.query(GroupAdmin)
        .filter(GroupAdmin.chat_id == chat_id, GroupAdmin.user_id == payload.user_id)
        .first()
    )
    if existing:
        return existing
    admin = GroupAdmin(chat_id=chat_id, user_id=payload.user_id)
    db.add(admin)
    db.commit()
    db.refresh(admin)
    return admin


@app.delete("/admin/groups/{chat_id}/admins/{user_id}")
def remove_group_admin(
    chat_id: int,
    user_id: int,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    _require_group_chat(chat_id)
    deleted = (
        db.query(GroupAdmin)
        .filter(GroupAdmin.chat_id == chat_id, GroupAdmin.user_id == user_id)
        .delete()
    )
    db.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail="Admin not found")
    return {"status": "ok"}


@app.get("/admin/stats/{chat_id}", response_model=StatsResponse)
def stats_endpoint(
    chat_id: int,
    window: str = "24h",
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> StatsResponse:
    _require_group_chat(chat_id)
    hours = 24 if window == "24h" else 24 * 7
    cutoff = now_local() - timedelta(hours=hours)
    query = (
        db.query(ModerationEvent)
        .filter(ModerationEvent.chat_id == chat_id, ModerationEvent.created_at >= cutoff)
        .order_by(desc(ModerationEvent.created_at))
    )
    events = query.all()
    actionable = [event for event in events if event.action not in ACTION_IGNORE]
    blocked = sum(1 for event in actionable if event.action in ACTION_BLOCKED)
    warned = sum(1 for event in actionable if event.action in ACTION_WARNED)
    deleted = len(actionable)
    offenders = (
        db.query(ModerationEvent.username, func.count(ModerationEvent.id).label("cnt"))
        .filter(
            ModerationEvent.chat_id == chat_id,
            ModerationEvent.created_at >= cutoff,
            ModerationEvent.username.isnot(None),
            ~ModerationEvent.action.in_(tuple(ACTION_IGNORE)),
        )
        .group_by(ModerationEvent.username)
        .order_by(desc("cnt"))
        .limit(5)
        .all()
    )
    top_offenders = [f"{username} ({count})" for username, count in offenders if username]
    return StatsResponse(
        chat_id=chat_id,
        window=window,
        total_events=len(actionable),
        blocked=blocked,
        warned=warned,
        deleted=deleted,
        top_offenders=top_offenders,
    )


@app.get("/admin/export/{chat_id}")
def export_endpoint(
    chat_id: int,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    _require_group_chat(chat_id)
    query = (
        db.query(
            ModerationEvent.id,
            ModerationEvent.chat_id,
            ModerationEvent.user_id,
            ModerationEvent.username,
            ModerationEvent.prob_hate,
            ModerationEvent.prob_nonhate,
            ModerationEvent.action,
            ModerationEvent.reason,
            ModerationEvent.created_at,
            ModerationEvent.text,
        )
        .filter(ModerationEvent.chat_id == chat_id)
        .order_by(desc(ModerationEvent.created_at))
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "chat_id", "user_id", "username", "prob_hate", "prob_nonhate", "action", "reason", "created_at", "text"])
    for row in query.all():
        writer.writerow(row)
    buffer.seek(0)
    filename = f"hate_guard_{chat_id}.csv"
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.post("/admin/test", response_model=PredictionResponse)
def test_text(
    payload: AdminTestRequest,
    _: None = Depends(require_api_key),
) -> PredictionResponse:
    prob_hate, prob_nonhate, pred_idx = classifier.predict(payload.text)
    threshold = payload.threshold or app_settings.default_threshold
    return PredictionResponse(
        prob_hate=prob_hate,
        prob_nonhate=prob_nonhate,
        pred=pred_idx,
        label="hate" if prob_hate >= threshold else "non-hate",
    )


@app.post("/admin/events")
def create_event(
    payload: EventCreate,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    _require_group_chat(payload.chat_id)
    settings_row = _get_or_create_settings(db, payload.chat_id)
    event = ModerationEvent(
        chat_id=payload.chat_id,
        user_id=payload.user_id,
        username=payload.username,
        message_id=payload.message_id,
        prob_hate=payload.prob_hate,
        prob_nonhate=payload.prob_nonhate,
        action=payload.action,
        text=payload.text,
        reason=payload.reason,
    )
    db.add(event)
    settings_row.last_active = now_local()
    db.add(settings_row)
    db.commit()
    db.refresh(event)
    return {"status": "ok", "id": event.id}


@app.get("/admin/groups/{chat_id}/members", response_model=List[MemberModerationSchema])
def list_member_moderations(
    chat_id: int,
    status: MemberStatus | None = None,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> List[MemberModerationSchema]:
    _require_group_chat(chat_id)
    query = db.query(MemberModeration).filter(MemberModeration.chat_id == chat_id)
    if status:
        query = query.filter(MemberModeration.status == status)
    members = query.order_by(desc(MemberModeration.created_at)).all()
    return members


@app.post("/admin/groups/{chat_id}/members", response_model=MemberModerationSchema)
def upsert_member_moderation(
    chat_id: int,
    payload: MemberModerationPayload,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> MemberModerationSchema:
    _require_group_chat(chat_id)
    _get_or_create_settings(db, chat_id)
    expires_at = None
    if payload.duration_minutes:
        expires_at = now_local() + timedelta(minutes=payload.duration_minutes)
    record = (
        db.query(MemberModeration)
        .filter(MemberModeration.chat_id == chat_id, MemberModeration.user_id == payload.user_id, MemberModeration.status == payload.status)
        .first()
    )
    if record:
        record.username = payload.username or record.username
        record.reason = payload.reason
        record.expires_at = expires_at
    else:
        record = MemberModeration(
            chat_id=chat_id,
            user_id=payload.user_id,
            username=payload.username,
            status=payload.status,
            reason=payload.reason,
            expires_at=expires_at,
        )
        db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.delete("/admin/groups/{chat_id}/members/{user_id}")
def remove_member_moderation(
    chat_id: int,
    user_id: int,
    status: MemberStatus,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
):
    _require_group_chat(chat_id)
    deleted = (
        db.query(MemberModeration)
        .filter(
            MemberModeration.chat_id == chat_id,
            MemberModeration.user_id == user_id,
            MemberModeration.status == status,
        )
        .delete()
    )
    db.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail="Member status not found")
    return {"status": "ok"}


@app.get("/admin/activity/{chat_id}", response_model=ActivityResponse)
def activity_endpoint(
    chat_id: int,
    days: int = 7,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> ActivityResponse:
    _require_group_chat(chat_id)
    cutoff = now_local() - timedelta(days=days)
    warned_case = case((ModerationEvent.action.in_(tuple(ACTION_WARNED)), 1), else_=0)
    blocked_case = case((ModerationEvent.action.in_(tuple(ACTION_BLOCKED)), 1), else_=0)
    rows = (
        db.query(
            func.date(ModerationEvent.created_at).label("day"),
            func.count(ModerationEvent.id).label("deleted"),
            func.sum(warned_case).label("warned"),
            func.sum(blocked_case).label("blocked"),
        )
        .filter(
            ModerationEvent.chat_id == chat_id,
            ModerationEvent.created_at >= cutoff,
            ~ModerationEvent.action.in_(tuple(ACTION_IGNORE)),
        )
        .group_by(func.date(ModerationEvent.created_at))
        .order_by(func.date(ModerationEvent.created_at))
        .all()
    )
    points = []
    for row in rows:
        normalized_day = _normalize_day(row.day)
        points.append(
            {
                "date": datetime.combine(normalized_day, datetime.min.time(), tzinfo=LOCAL_TIMEZONE),
                "deleted": row.deleted,
                "warned": row.warned,
                "blocked": row.blocked,
            }
        )
    return ActivityResponse(chat_id=chat_id, points=points)


@app.get("/admin/action_count/{chat_id}/{user_id}")
def action_count(
    chat_id: int,
    user_id: int,
    action: str = "warned",
    period: str = "day",
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> Dict[str, int]:
    _require_group_chat(chat_id)
    action = action.lower()
    period = (period or "day").lower()
    reset_at = (
        db.query(ActionCounterReset.reset_at)
        .filter(
            ActionCounterReset.chat_id == chat_id,
            ActionCounterReset.user_id == user_id,
            ActionCounterReset.action == action,
        )
        .scalar()
    )
    reset_at = _ensure_local_datetime(reset_at)
    query = (
        db.query(func.count(ModerationEvent.id))
        .filter(
            ModerationEvent.chat_id == chat_id,
            ModerationEvent.user_id == user_id,
            ModerationEvent.action == action,
        )
    )
    lower_bound = EPOCH_START
    upper_bound: datetime | None = None
    if period == "day":
        start, end = day_bounds()
        lower_bound = start
        upper_bound = end
    if reset_at:
        lower_bound = max(lower_bound, reset_at)
    query = query.filter(ModerationEvent.created_at >= lower_bound)
    if upper_bound:
        query = query.filter(ModerationEvent.created_at < upper_bound)
    count = query.scalar() or 0
    return {"count": count}


def _build_action_summary_entry(user_id: int, username: str | None) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "username": username,
        "warnings_today": 0,
        "mutes_total": 0,
        "last_warning": None,
        "last_mute": None,
    }


@app.get("/admin/user_actions/{chat_id}", response_model=List[UserActionSummary])
def list_user_actions(
    chat_id: int,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> List[UserActionSummary]:
    _require_group_chat(chat_id)
    start_day, end_day = day_bounds()
    resets = {
        (reset.user_id, reset.action): _ensure_local_datetime(reset.reset_at) or EPOCH_START
        for reset in db.query(ActionCounterReset).filter(ActionCounterReset.chat_id == chat_id).all()
    }
    summaries: Dict[int, Dict[str, Any]] = {}

    warning_events = (
        db.query(ModerationEvent.user_id, ModerationEvent.username, ModerationEvent.created_at)
        .filter(
            ModerationEvent.chat_id == chat_id,
            ModerationEvent.action == "warned",
            ModerationEvent.created_at >= start_day,
            ModerationEvent.created_at < end_day,
        )
        .all()
    )
    for event in warning_events:
        if not event.user_id:
            continue
        event_time = _ensure_local_datetime(event.created_at)
        cutoff = max(start_day, resets.get((event.user_id, "warned"), start_day))
        if not event_time or event_time < cutoff:
            continue
        entry = summaries.setdefault(event.user_id, _build_action_summary_entry(event.user_id, event.username))
        if event.username and not entry["username"]:
            entry["username"] = event.username
        entry["warnings_today"] += 1
        if not entry["last_warning"] or event_time > entry["last_warning"]:
            entry["last_warning"] = event_time

    mute_events = (
        db.query(ModerationEvent.user_id, ModerationEvent.username, ModerationEvent.created_at)
        .filter(ModerationEvent.chat_id == chat_id, ModerationEvent.action == "muted")
        .all()
    )
    for event in mute_events:
        if not event.user_id:
            continue
        event_time = _ensure_local_datetime(event.created_at)
        cutoff = resets.get((event.user_id, "muted"), EPOCH_START)
        if not event_time or event_time < cutoff:
            continue
        entry = summaries.setdefault(event.user_id, _build_action_summary_entry(event.user_id, event.username))
        if event.username and not entry["username"]:
            entry["username"] = event.username
        entry["mutes_total"] += 1
        if not entry["last_mute"] or event_time > entry["last_mute"]:
            entry["last_mute"] = event_time

    sorted_entries = sorted(
        summaries.values(),
        key=lambda item: (-(item["warnings_today"] or 0), -(item["mutes_total"] or 0), item["user_id"]),
    )
    return [UserActionSummary(**entry) for entry in sorted_entries]


@app.post("/admin/user_actions/{chat_id}/{user_id}/reset")
def reset_user_action(
    chat_id: int,
    user_id: int,
    payload: ActionResetPayload,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    _require_group_chat(chat_id)
    action = payload.action.lower()
    if action not in {"warned", "muted"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Action harus warned atau muted")
    now = now_local()
    existing = (
        db.query(ActionCounterReset)
        .filter(
            ActionCounterReset.chat_id == chat_id,
            ActionCounterReset.user_id == user_id,
            ActionCounterReset.action == action,
        )
        .first()
    )
    if existing:
        existing.reset_at = now
    else:
        db.add(ActionCounterReset(chat_id=chat_id, user_id=user_id, action=action, reset_at=now))
    db.commit()
    return {"status": "ok", "action": action, "reset_at": now}


@app.get("/admin/events/{chat_id}", response_model=List[EventSchema])
def recent_events(
    chat_id: int,
    limit: int = 50,
    offset: int = 0,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> List[EventSchema]:
    _require_group_chat(chat_id)
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    query = (
        db.query(ModerationEvent)
        .filter(ModerationEvent.chat_id == chat_id)
        .order_by(desc(ModerationEvent.created_at))
        .offset(offset)
        .limit(limit)
    )
    return query.all()


@app.get("/admin/events/{chat_id}/count")
def events_count(
    chat_id: int,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> Dict[str, int]:
    _require_group_chat(chat_id)
    total = (
        db.query(func.count(ModerationEvent.id))
        .filter(ModerationEvent.chat_id == chat_id)
        .scalar()
    ) or 0
    return {"total": total}


@app.post("/admin/events/{event_id}/verify", response_model=EventSchema)
def verify_event(
    event_id: int,
    payload: EventVerificationPayload,
    _: None = Depends(require_api_key),
    db: Session = Depends(get_db),
) -> EventSchema:
    _require_group_chat(payload.chat_id)
    event = db.get(ModerationEvent, event_id)
    if not event or event.chat_id != payload.chat_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event tidak ditemukan")
    event.manual_label = payload.label
    event.manual_verified = True
    event.manual_verified_at = now_local()
    db.add(event)
    db.commit()
    db.refresh(event)
    return event
def _ensure_local_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=LOCAL_TIMEZONE)
    return value.astimezone(LOCAL_TIMEZONE)
