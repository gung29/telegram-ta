# Hate Guard Bot

Telegram hate-speech moderation stack yang terdiri dari:

- **FastAPI inference service** (`api/app.py`) menggunakan ONNXRuntime + `IndoBERTweet` (via `model/model.onnx`).
- **Aiogram bot** (`bot/main.py`) dengan mode webhook, logging SQLite, dan perintah admin lengkap.
- **Telegram Mini App dashboard** (`web/main.py`) untuk mengubah threshold/mode, melihat statistik, mengelola whitelist, dan melakukan simulasi teks.

> Plan implementasi mengacu pada dokumen `telegram-hate-speech-bot-plan.md`.

## Fitur Utama

- Moderasi otomatis (hapus + mute + ban) berbasis threshold per grup, dengan eskalasi otomatis dan pencatatan anggota yang dimute/diblokir.
- `/moderation_on|off`, `/set_threshold`, `/set_mode`, `/stats`, `/test`, `/admins`, `/admin_add|remove`, `/mute`, `/ban`, `/why` untuk kontrol via bot.
- Dashboard Mini App React menampilkan statistik real-time (chart aktivitas), pengaturan threshold/mode, daftar grup yang dijaga, simulasi teks, ekspor CSV, serta manajemen admin & anggota (mute/ban manual).
- REST API FastAPI menyediakan endpoint agregasi statistik, timeline aktivitas, manajemen admin/anggota, sinkronisasi grup, dan Prometheus metrics.
- Node.js bot (node-telegram-bot-api) berkomunikasi dengan API melalui HTTP + API key, sehingga model inferensi tetap berjalan di Python/ONNXRuntime.

## Struktur Folder

```
├── api/          # FastAPI inference & admin endpoints (Python / ONNXRuntime)
├── bot/          # Node.js Telegram bot (node-telegram-bot-api + axios)
├── web/
│   ├── main.py          # FastAPI proxy + static hosting
│   └── frontend/        # Vite + React (Telegram Mini App UI)
├── common/       # Shared config, DB, models, schemas, logging
├── data/         # SQLite DB + model/tokenizer artefacts
└── .env.example
```

## Persiapan Lingkungan

1. Salin `.env.example` ke `.env`, isi token/bot secret/model path.
2. Buat virtualenv & instal dependensi:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Taruh model ONNX (`data/model/model.onnx`) dan tokenizer HuggingFace (`tokenizer.json`).

## Menjalankan Komponen

### Alternatif: Docker / Docker Compose

Cara paling cepat menjalankan keseluruhan stack (API + bot) adalah memakai Docker. Repository ini sudah memiliki `Dockerfile.api`, `Dockerfile.bot`, dan `docker-compose.yml`. Langkahnya:

1. Pastikan `.env` diisi lengkap (token bot, API key, URL webhook/mini-app, dsb).
2. Bangun dan jalankan container:

   ```bash
   docker compose up -d --build
   ```

   - Service `api` otomatis membuild frontend (Vite) lalu menjalankan `uvicorn web.main:app` pada port 8080 (forward ke host).
   - Service `bot` menjalankan compile TypeScript (`npm run build`) dan start dari `dist/index.js`.
   - Volume `./data:/app/data` memastikan SQLite tetap persist.

3. Untuk update kode dari lokal → server cukup `git pull` lalu `docker compose up -d --build`.

Jika ingin mengganti port atau menambahkan reverse proxy (Nginx, Caddy, aaPanel), arahkan saja ke `localhost:8080` (API + Mini App) dan port webhook yang Anda expose di `.env`.

### 1. Inference API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoint penting:

- `POST /predict` – `{ text }` → probabilitas hate/non-hate.
- `POST /admin/settings/{chat_id}` – ubah threshold/mode, butuh `X-API-Key`.
- `GET /metrics` – Prometheus scrape.

### 2. Telegram Bot (Node + node-telegram-bot-api)

```bash
cd bot
npm install
npm run dev      # hot reload (ts-node-dev + polling)
npm run build    # compile to dist/
npm start        # run compiled bot
```

- Konfigurasi diambil dari `.env` (token, API URL, API key, mini-app URL, dll).
- Default menjalankan **polling**. Jika `WEBHOOK_URL` diset, bot otomatis beralih ke webhook mode pada port `BOT_WEBHOOK_PORT` (fallback `WEBHOOK_PORT`).
- Fitur moderasi, perintah admin, whitelist, dan logging sepenuhnya dilakukan lewat panggilan HTTP ke FastAPI sehingga model tetap berada di Python/ONNXRuntime.

### 3. Mini App Dashboard (Vite + Telegram WebApp)

1. **Frontend (npx / Vite)**

   ```bash
   cd web/frontend
   npm install
   npm run dev        # atau: npx vite
   npm run build      # produksi → output ke web/frontend/dist
   ```

   Mode dev menjalankan Vite pada port 5173 (ideal untuk styling). Build production akan dipakai oleh FastAPI secara otomatis.

2. **Proxy + Static Host**

   ```bash
   cd ../../
   uvicorn web.main:app --host 0.0.0.0 --port 8080 --reload
   ```

   - FastAPI akan melayani `/api/*` (proxy ke inference API dengan validasi `initData`) dan menyajikan bundle React yang sudah dibuild dari `web/frontend/dist`.
   - Untuk pengembangan lokal tanpa Telegram, set `MINI_APP_DEV_MODE=true` dan tambahkan `?chat_id=...` pada URL (atau gunakan header `X-Debug-Chat-Id`) sehingga backend mengizinkan akses tanpa HMAC.
   - UI sudah seluruhnya dark-mode, responsive, dan mengikuti pedoman Telegram Mini App terbaru (menggunakan `window.Telegram.WebApp` API saat `npm run dev` / `npx vite` maupun produksi).
   - Fitur dashboard:
     - Switch antar grup yang diawasi (list otomatis dari bot).
     - Kontrol moderasi (on/off, threshold, mode).
     - Statistik agregat + grafik timeline + ekspor CSV.
     - Simulasi teks langsung via model.
     - Manajemen admin tambahan, daftar anggota yang dimute/diban, dan form untuk mute/ban manual.

#### HTTPS tunneling (ngrok helper)

Telegram mengharuskan URL WebApp & webhook memakai HTTPS. Gunakan helper berikut supaya tidak perlu mengatur reverse-proxy manual setiap kali pengembangan:

```bash
# Pastikan sudah login ngrok: ngrok config add-authtoken <token>
pip install -r requirements.txt   # pyngrok sudah termasuk
python scripts/dev_tunnel.py --update-env
```

- Script akan membuka tunnel untuk Mini App (`mini-port=8080`) dan webhook (`webhook-port=8081`), lalu menuliskan `MINI_APP_BASE_URL` dan `WEBHOOK_URL` ke `.env`.
- Gunakan `--mini-only` jika hanya butuh Mini App, atau `--exit` bila hanya ingin mengetahui URL tanpa mempertahankan proses.
- Jalankan script ini di terminal terpisah; selama aktif, kamu bisa menjalankan `uvicorn web.main:app` + `npm run dev` dan bot Node.js dengan URL HTTPS yang valid.

### 4. Nginx Reverse Proxy (contoh)

```
bot.gungzy.xyz   → proxy_pass http://127.0.0.1:8081 (bot webhook server)
api.gungzy.xyz   → proxy_pass http://127.0.0.1:8000 (FastAPI inference)
mini.gungzy.xyz  → proxy_pass http://127.0.0.1:8080 (Mini App dashboard)
```

Aktifkan SSL + Telegram IP whitelist untuk webhook.

## Database

- Default: SQLite `data/db.sqlite3`.
- Ganti `DATABASE_URL` untuk PostgreSQL (`postgresql+psycopg://...`).
- Tabel:
  - `group_settings`
  - `events`
  - `whitelist`

## Testing & Debugging

- Jalankan `uvicorn` + `python bot/main.py --mode polling` di lokal dengan ngrok untuk webhook.
- Gunakan `/test <teks>` untuk cek skor manual.
- Gunakan `/stats 24h` atau `/stats 7d` di grup untuk ringkasan.
- Mini App menyediakan export CSV via tombol “Export” (endpoint `/admin/export/{chat_id}` jika perlu integrasi tambahan).

## Pengembangan Lanjutan

- Tambah worker untuk adaptive threshold / shadow mode.
- Sambungkan ke Prometheus + Grafana untuk observasi latensi/QPS.
- Migrasi ke PostgreSQL + Alembic untuk multi-tenant.

Selamat membangun platform moderasi Telegram Anda! 🎯
