from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import urllib.parse
import logging
from typing import Any, Dict

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from common.config import settings as app_settings

DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
INDEX_FILE = DIST_DIR / "index.html"

logger = logging.getLogger("hate-guard-web")
logger.setLevel(logging.INFO)

app = FastAPI(title="Hate Guard Mini App")

http_client = httpx.AsyncClient(
    base_url=app_settings.inference_api_url,
    headers={"X-API-Key": app_settings.api_key},
    timeout=10,
)


def verify_init_data(init_data: str) -> Dict[str, Any]:
    parsed = dict(urllib.parse.parse_qsl(init_data, keep_blank_values=True))
    if "hash" not in parsed:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing hash")

    hash_value = parsed.pop("hash")
    data_check_string = "\n".join(f"{k}={parsed[k]}" for k in sorted(parsed.keys()))
    secret_key = hmac.new(b"WebAppData", app_settings.bot_token.encode(), hashlib.sha256).digest()
    expected_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_hash, hash_value):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid WebApp signature")
    if "user" in parsed:
        parsed["user"] = json.loads(parsed["user"])
    if "chat" in parsed:
        parsed["chat"] = json.loads(parsed["chat"])
    return parsed


async def get_context(
    request: Request,
    x_init_data: str | None = Header(default=None, alias="X-Init-Data"),
    x_debug_chat_id: str | None = Header(default=None, alias="X-Debug-Chat-Id"),
    x_debug_user_id: str | None = Header(default=None, alias="X-Debug-User-Id"),
) -> Dict[str, Any]:
    init_data = x_init_data or request.query_params.get("initData")
    debug_chat = x_debug_chat_id or request.query_params.get("chat_id")
    debug_user = x_debug_user_id or request.query_params.get("debug_user_id")

    if init_data:
        return verify_init_data(init_data)

    if app_settings.mini_app_dev_mode and debug_chat:
        ctx: Dict[str, Any] = {"chat_id": int(debug_chat), "chat": {"id": int(debug_chat)}, "dev_mode": True}
        if debug_user:
            ctx["user"] = {"id": int(debug_user)}
        elif app_settings.admin_ids:
            ctx["user"] = {"id": app_settings.admin_ids[0]}
        return ctx

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Init data diperlukan")


def _assert_chat_access(ctx: Dict[str, Any], chat_id: int) -> None:
    allowed_id = None
    if "chat" in ctx and isinstance(ctx["chat"], dict):
        allowed_id = ctx["chat"].get("id")
    elif "chat_id" in ctx:
        allowed_id = ctx["chat_id"]
    if allowed_id is not None and int(allowed_id) != int(chat_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tidak boleh mengakses grup lain")


async def ensure_admin_access(chat_id: int, ctx: Dict[str, Any]) -> None:
    user = ctx.get("user")
    if not isinstance(user, dict) or "id" not in user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Dashboard hanya untuk admin grup")
    user_id = int(user["id"])
    admins = await proxy_core_api("GET", f"/admin/groups/{chat_id}/admins")
    allowed_ids = {int(entry["user_id"]) for entry in admins if isinstance(entry, dict) and entry.get("user_id") is not None}
    if not allowed_ids and app_settings.admin_ids:
        allowed_ids.update(app_settings.admin_ids)
    if user_id not in allowed_ids:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Dashboard hanya untuk admin grup")


async def proxy_core_api(
    method: str,
    path: str,
    json_data: Dict[str, Any] | None = None,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    try:
        response = await http_client.request(method, path, json=json_data, params=params)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail: Any
        try:
            payload = exc.response.json()
            detail = payload.get("detail", payload)
        except ValueError:
            detail = exc.response.text or exc.response.reason_phrase
        raise HTTPException(status_code=exc.response.status_code, detail=detail)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Core API unreachable: {exc}",
        ) from exc

    if not response.content:
        return {"status": "ok"}

    try:
        return response.json()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Invalid response from core API",
        )


async def register_bot_webhook() -> None:
    if not app_settings.webhook_url or not app_settings.bot_token:
        logger.warning("Webhook URL or bot token missing; skipping webhook registration")
        return
    payload = {"url": app_settings.webhook_url, "secret_token": app_settings.webhook_secret}
    url = f"https://api.telegram.org/bot{app_settings.bot_token}/setWebhook"
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Failed to register webhook: {data}")
    logger.info("Webhook registered automatically from .env", extra={"url": app_settings.webhook_url})


@app.on_event("startup")
async def startup_event() -> None:
    try:
        await register_bot_webhook()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to set webhook: %s", exc)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await http_client.aclose()


@app.get("/api/settings")
async def api_get_settings(chat_id: int, ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("GET", f"/admin/settings/{chat_id}")


@app.post("/api/settings")
async def api_update_settings(chat_id: int, payload: Dict[str, Any], ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("POST", f"/admin/settings/{chat_id}", json_data=payload)


@app.get("/api/stats")
async def api_stats(chat_id: int, window: str = "24h", ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("GET", f"/admin/stats/{chat_id}", params={"window": window})


@app.get("/api/groups")
async def api_groups(ctx: Dict[str, Any] = Depends(get_context)):
    # List of groups is safe to share with any authenticated WebApp user; granular access is enforced per-chat.
    return await proxy_core_api("GET", "/admin/groups")


@app.get("/api/admins")
async def api_admins(chat_id: int, ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("GET", f"/admin/groups/{chat_id}/admins")


@app.post("/api/admins")
async def api_admin_add(chat_id: int, payload: Dict[str, Any], ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("POST", f"/admin/groups/{chat_id}/admins", json_data=payload)


@app.delete("/api/admins/{user_id}")
async def api_admin_remove(user_id: int, chat_id: int, ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("DELETE", f"/admin/groups/{chat_id}/admins/{user_id}")


@app.get("/api/members")
async def api_members(chat_id: int, status: str | None = None, ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    params = {"status": status} if status else None
    return await proxy_core_api("GET", f"/admin/groups/{chat_id}/members", params=params)


@app.post("/api/members")
async def api_member_add(chat_id: int, payload: Dict[str, Any], ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("POST", f"/admin/groups/{chat_id}/members", json_data=payload)


@app.delete("/api/members/{user_id}")
async def api_member_remove(user_id: int, chat_id: int, status: str, ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("DELETE", f"/admin/groups/{chat_id}/members/{user_id}", params={"status": status})


@app.get("/api/activity")
async def api_activity(chat_id: int, days: int = 7, ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("GET", f"/admin/activity/{chat_id}", params={"days": days})


@app.get("/api/events")
async def api_events(chat_id: int, limit: int = 50, offset: int = 0, ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("GET", f"/admin/events/{chat_id}", params={"limit": limit, "offset": offset})


@app.get("/api/events/count")
async def api_events_count(chat_id: int, ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("GET", f"/admin/events/{chat_id}/count")


@app.post("/api/events/{event_id}/verify")
async def api_verify_event(event_id: int, chat_id: int, payload: Dict[str, Any], ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    if "label" not in payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Label verifikasi diperlukan")
    data = {"chat_id": chat_id, "label": payload["label"]}
    return await proxy_core_api("POST", f"/admin/events/{event_id}/verify", json_data=data)


@app.get("/api/export")
async def api_export(chat_id: int, ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    response = await http_client.get(f"/admin/export/{chat_id}")
    response.raise_for_status()
    filename = f"hate_guard_{chat_id}.csv"
    return Response(content=response.content, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/api/user_actions")
async def api_user_actions(chat_id: int, ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("GET", f"/admin/user_actions/{chat_id}")


@app.post("/api/user_actions/{user_id}/reset")
async def api_reset_user_actions(chat_id: int, user_id: int, payload: Dict[str, Any], ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    return await proxy_core_api("POST", f"/admin/user_actions/{chat_id}/{user_id}/reset", json_data=payload)


@app.post("/api/test")
async def api_test(chat_id: int, payload: Dict[str, Any], ctx: Dict[str, Any] = Depends(get_context)):
    await ensure_admin_access(chat_id, ctx)
    _assert_chat_access(ctx, chat_id)
    payload.setdefault("chat_id", chat_id)
    return await proxy_core_api("POST", "/admin/test", json_data=payload)


if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
else:
    @app.get("/", response_class=HTMLResponse)
    async def placeholder() -> HTMLResponse:
        return HTMLResponse(
            "<h1>Mini App belum dibuild</h1><p>Jalankan <code>cd web/frontend && npm install && npm run build</code> untuk menghasilkan aset.</p>",
        )
