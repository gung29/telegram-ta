#!/usr/bin/env bash
set -e

BASE="/www/wwwroot/telegram-ta"
VENV="$BASE/.venv/bin"
LOG_DIR="$BASE/logs"
BOT_NAME="hate-guard-bot"

start_core() {
  source "$BASE/.venv/bin/activate"
  nohup "$VENV/uvicorn" api.app:app --host 0.0.0.0 --port 8000 > "$LOG_DIR/core-api.out" 2>&1 &
}

stop_core() {
  pkill -f "uvicorn.*api.app:app" || true
}

start_mini() {
  source "$BASE/.venv/bin/activate"
  nohup "$VENV/uvicorn" web.main:app --host 0.0.0.0 --port 8080 > "$LOG_DIR/mini-api.out" 2>&1 &
}

stop_mini() {
  pkill -f "uvicorn.*web.main:app" || true
}

start_bot() {
  cd "$BASE/bot"
  npm install
  npm run build
  pm2 start npm --name "$BOT_NAME" -- run start >/dev/null || pm2 restart "$BOT_NAME"
}

stop_bot() {
  pm2 delete "$BOT_NAME" >/dev/null 2>&1 || true
}

case "$1" in
  start)
    start_core
    start_mini
    start_bot
    ;;
  stop)
    stop_bot
    stop_mini
    stop_core
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    pgrep -a "uvicorn.*api.app:app" || echo "core API not running"
    pgrep -a "uvicorn.*web.main:app" || echo "mini app not running"
    pm2 list
    ;;
  *)
    echo "Usage: scripts/manage.sh {start|stop|restart|status}"
    exit 1
    ;;
esac
