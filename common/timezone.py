from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Tuple

LOCAL_TIMEZONE = timezone(timedelta(hours=8))


def now_local() -> datetime:
    return datetime.now(LOCAL_TIMEZONE)


def day_bounds(moment: datetime | None = None) -> Tuple[datetime, datetime]:
    current = (moment or now_local()).astimezone(LOCAL_TIMEZONE)
    start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end
