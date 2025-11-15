#!/usr/bin/env python3
"""Utility to spin up HTTPS tunnels (Mini App + webhook) via ngrok and update .env."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict

from dotenv import dotenv_values
from pyngrok import conf, ngrok

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def ensure_auth_token(token: str | None) -> None:
    token = token or os.getenv("NGROK_AUTHTOKEN")
    if not token:
        print("⚠️  ngrok auth token is not configured. Run `ngrok config add-authtoken <token>` or pass --auth-token.")
        return
    conf.get_default().auth_token = token


def update_env_file(updates: Dict[str, str]) -> None:
    if not ENV_PATH.exists():
        print(f"⚠️  .env not found at {ENV_PATH}. Skipping automatic update.")
        return

    lines = ENV_PATH.read_text().splitlines()
    applied = {key: False for key in updates}

    for idx, line in enumerate(lines):
        if not line or line.strip().startswith("#"):
            continue
        key = line.split("=", 1)[0]
        if key in updates:
            lines[idx] = f"{key}={updates[key]}"
            applied[key] = True

    for key, done in applied.items():
        if not done:
            lines.append(f"{key}={updates[key]}")

    ENV_PATH.write_text("\n".join(lines) + "\n")
    print(f"✅ Updated .env with: {', '.join(updates.keys())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch ngrok tunnels for Mini App + webhook and update .env automatically.")
    parser.add_argument("--mini-port", type=int, default=8080, help="Local port serving the Mini App (default: 8080)")
    parser.add_argument("--webhook-port", type=int, default=8081, help="Local port for the Telegram bot webhook (default: 8081)")
    parser.add_argument("--auth-token", type=str, default=None, help="ngrok auth token (optional, falls back to env/CLI config)")
    parser.add_argument("--update-env", action="store_true", help="Write discovered HTTPS URLs into .env")
    parser.add_argument("--webhook-path", type=str, default=None, help="Override webhook path when updating .env (default: value from .env or /webhook)")
    parser.add_argument("--mini-only", action="store_true", help="Skip creating webhook tunnel (only Mini App)")
    parser.add_argument("--exit", action="store_true", help="Create tunnels, print URLs, then exit (tunnels will shut down)")
    args = parser.parse_args()

    ensure_auth_token(args.auth_token)

    existing_env = dotenv_values(ENV_PATH) if ENV_PATH.exists() else {}
    webhook_path = args.webhook_path or existing_env.get("WEBHOOK_PATH") or "/webhook"

    try:
        mini_tunnel = ngrok.connect(args.mini_port, "http", bind_tls=True)
        print(f"🌐 Mini App tunnel: {mini_tunnel.public_url}")

        webhook_tunnel = None
        if not args.mini_only:
            webhook_tunnel = ngrok.connect(args.webhook_port, "http", bind_tls=True)
            print(f"🤖 Webhook tunnel: {webhook_tunnel.public_url}{webhook_path}")

        if args.update_env:
            updates = {"MINI_APP_BASE_URL": mini_tunnel.public_url}
            if webhook_tunnel:
                updates["WEBHOOK_URL"] = f"{webhook_tunnel.public_url}{webhook_path}"
            update_env_file(updates)

        if args.exit:
            return

        print("\nTunnels are live. Press Ctrl+C to stop.")
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopping tunnels...")
    finally:
        try:
            ngrok.kill()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
