#!/usr/bin/env bash
set -e

BASE="/www/wwwroot/telegram-ta"
VENV="$BASE/.venv/bin"
LOG_DIR="$BASE/logs"
BOT_NAME="hate-guard-bot"
MIN_CORE_RAM_MB="${MIN_CORE_RAM_MB:-1800}"

ensure_dirs() {
  mkdir -p "$LOG_DIR"
}

build_frontend() {
  cd "$BASE/referensi-desain"
  npm install
  npm run build
}

build_bot() {
  cd "$BASE/bot"
  npm install
  npm run build
}

build_all() {
  build_frontend
  build_bot
}

env_value() {
  local key="$1"
  if [ -f "$BASE/.env" ]; then
    grep -E "^${key}=" "$BASE/.env" | tail -n 1 | cut -d= -f2- || true
  fi
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

assert_core_memory_safe() {
  local skip_model
  skip_model="$(env_value SKIP_MODEL_LOAD)"
  if is_truthy "$skip_model"; then
    return 0
  fi

  if is_truthy "${ALLOW_LOW_MEMORY_CORE:-}"; then
    echo "WARNING: ALLOW_LOW_MEMORY_CORE is enabled; starting Core API even though VPS may freeze."
    return 0
  fi

  local mem_total_mb
  mem_total_mb="$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)"
  if [ "$mem_total_mb" -lt "$MIN_CORE_RAM_MB" ]; then
    cat <<EOF
Refusing to start Core API: this VPS has ${mem_total_mb}MB RAM, below safe minimum ${MIN_CORE_RAM_MB}MB.
The model is loaded by PyTorch/Transformers and can freeze a 1GB VPS.

Options:
  1. Upgrade VPS to at least 2GB RAM, recommended 4GB.
  2. Add SKIP_MODEL_LOAD=1 to .env for dashboard/API smoke test only (predictions become dummy).
  3. Override at your own risk:
     ALLOW_LOW_MEMORY_CORE=1 ./scripts/manage-new.sh start-runtime
EOF
    return 1
  fi
}

start_core() {
  assert_core_memory_safe
  local host
  local port
  host="$(env_value INFERENCE_API_HOST)"
  port="$(env_value INFERENCE_API_PORT)"
  host="${host:-127.0.0.1}"
  port="${port:-8000}"
  source "$BASE/.venv/bin/activate"
  (
    cd "$BASE"
    nohup "$VENV/uvicorn" api.app:app --host "$host" --port "$port" > "$LOG_DIR/core-api.out" 2>&1 &
  )
}

stop_core() {
  pkill -f "uvicorn.*api.app:app" || true
}

start_mini() {
  local host
  local port
  host="$(env_value MINI_APP_HOST)"
  port="$(env_value MINI_APP_PORT)"
  host="${host:-127.0.0.1}"
  port="${port:-8080}"
  source "$BASE/.venv/bin/activate"
  (
    cd "$BASE"
    nohup "$VENV/uvicorn" web.main:app --host "$host" --port "$port" > "$LOG_DIR/mini-api.out" 2>&1 &
  )
}

stop_mini() {
  pkill -f "uvicorn.*web.main:app" || true
}

start_bot() {
  cd "$BASE/bot"
  pm2 start npm --name "$BOT_NAME" -- run start >/dev/null || pm2 restart "$BOT_NAME"
}

stop_bot() {
  pm2 delete "$BOT_NAME" >/dev/null 2>&1 || true
}

show_log_tail() {
  local file="$1"
  if [ -s "$file" ]; then
    echo "---- last lines: $file ----"
    tail -n 40 "$file"
  else
    echo "log file empty or missing: $file"
  fi
}

http_reachable() {
  curl -fsS --max-time 3 "$1" >/dev/null 2>&1
}

show_status() {
  if pgrep -a "uvicorn.*api.app:app"; then
    :
  elif http_reachable "http://127.0.0.1:8000/healthz"; then
    echo "core API reachable on http://127.0.0.1:8000 (no local uvicorn process; likely SSH tunnel)"
  else
    echo "core API not running"
    show_log_tail "$LOG_DIR/core-api.out"
  fi

  if pgrep -a "uvicorn.*web.main:app"; then
    :
  elif http_reachable "http://127.0.0.1:8080/"; then
    echo "mini app reachable on http://127.0.0.1:8080 (process name did not match pgrep pattern)"
  else
    echo "mini app not running"
    show_log_tail "$LOG_DIR/mini-api.out"
  fi

  pm2 list
}

show_logs() {
  show_log_tail "$LOG_DIR/core-api.out"
  show_log_tail "$LOG_DIR/mini-api.out"
  pm2 logs "$BOT_NAME" --lines 40 --nostream || true
}

diagnose_mini() {
  local local_url="http://127.0.0.1:8080"
  local public_url
  public_url="$(env_value MINI_APP_BASE_URL)"
  local html
  local asset

  echo "== Mini App local HTML =="
  html="$(curl -fsS --max-time 5 "$local_url/" || true)"
  if [ -z "$html" ]; then
    echo "Cannot fetch $local_url/"
  else
    if echo "$html" | grep -q 'src="/index.tsx"'; then
      echo "Problem: serving source index.html (/index.tsx), not built dist."
    elif echo "$html" | grep -q '/assets/.*\.js'; then
      echo "OK: built asset reference found."
    else
      echo "Warning: no /assets/*.js reference found."
    fi

    asset="$(echo "$html" | grep -o '/assets/[^"]*\.js' | head -n 1 || true)"
    if [ -n "$asset" ]; then
      echo "Local JS asset: $asset"
      curl -I --max-time 5 "$local_url$asset" || true
    fi
  fi

  if [ -n "$public_url" ]; then
    echo "== Mini App public URL =="
    curl -I --max-time 8 "$public_url/" || true
    if [ -n "$asset" ]; then
      curl -I --max-time 8 "$public_url$asset" || true
    fi
  fi
}

start_runtime() {
  ensure_dirs
  start_core
  start_mini
  start_bot
}

start_edge() {
  ensure_dirs
  start_mini
  start_bot
}

case "$1" in
  build)
    build_all
    ;;
  start)
    ensure_dirs
    build_all
    start_runtime
    ;;
  start-runtime)
    start_runtime
    ;;
  start-edge)
    start_edge
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
  restart-runtime)
    "$0" stop
    "$0" start-runtime
    ;;
  restart-edge)
    "$0" stop
    "$0" start-edge
    ;;
  status)
    show_status
    ;;
  logs)
    show_logs
    ;;
  diagnose-mini)
    diagnose_mini
    ;;
  *)
    echo "Usage: scripts/manage-new.sh {build|start|start-runtime|start-edge|stop|restart|restart-runtime|restart-edge|status|logs|diagnose-mini}"
    exit 1
    ;;
esac
