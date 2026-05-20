# Panduan Running Hybrid: Model di Laptop, Bot dan Mini App di VPS

Panduan ini untuk skenario VPS kecil, misalnya 1GB RAM / 1 CPU. Core API dan model dijalankan di laptop lokal, sedangkan VPS hanya menjalankan Telegram bot, Mini App, Nginx aaPanel, SSL, dan webhook.

## 1. Arsitektur

Alur request:

```text
Telegram
  -> bot.gungzy.xyz/webhook
  -> VPS bot Node.js port 8081
  -> http://127.0.0.1:8000 di VPS
  -> SSH reverse tunnel
  -> Core API di laptop port 8000

Telegram Mini App
  -> mini.gungzy.xyz
  -> VPS Mini App port 8080
  -> http://127.0.0.1:8000 di VPS
  -> SSH reverse tunnel
  -> Core API di laptop port 8000
```

Yang berjalan di laptop:

- `uvicorn api.app:app --host 127.0.0.1 --port 8000`
- Model PyTorch/Transformers
- Database Core API jika `DATABASE_URL=sqlite:///./data/db.sqlite3`

Yang berjalan di VPS:

- `web.main:app` di port `8080`
- bot Node.js di port `8081`
- aaPanel + Nginx + SSL
- SSH reverse tunnel listener di `127.0.0.1:8000`

Kelemahan mode ini:

- Laptop harus menyala terus.
- Laptop jangan sleep/hibernate.
- Internet laptop harus stabil.
- Jika laptop atau tunnel mati, bot dan dashboard di VPS masih online tetapi Core API tidak bisa diakses.

## 2. Persiapan DNS

Di DNS domain, arahkan:

| Type | Name | Value |
| --- | --- | --- |
| A | `mini` | IP publik VPS |
| A | `bot` | IP publik VPS |

Cek dari laptop:

```powershell
nslookup mini.gungzy.xyz
nslookup bot.gungzy.xyz
```

Keduanya harus mengarah ke IP VPS.

## 3. Konfigurasi aaPanel

Gunakan aaPanel hanya untuk domain, Nginx, SSL, dan firewall. Jangan pakai Python Manager atau Node Project aaPanel untuk menjalankan aplikasi ini.

### 3.1. Add site

Buat dua website:

```text
aaPanel -> Website -> Add site
```

Site pertama:

| Field | Isi |
| --- | --- |
| Domain name | `mini.gungzy.xyz` |
| Root directory | `/www/wwwroot/mini.gungzy.xyz` |
| PHP | Static / Pure static / Not set |
| Database | Tidak perlu |

Site kedua:

| Field | Isi |
| --- | --- |
| Domain name | `bot.gungzy.xyz` |
| Root directory | `/www/wwwroot/bot.gungzy.xyz` |
| PHP | Static / Pure static / Not set |
| Database | Tidak perlu |

### 3.2. SSL

Untuk masing-masing site:

```text
aaPanel -> Website -> pilih domain -> SSL -> Let's Encrypt
```

Aktifkan `Force HTTPS`.

### 3.3. Nginx reverse proxy

Buka:

```text
aaPanel -> Website -> mini.gungzy.xyz -> Config
```

Ubah atau tambahkan blok `location /`:

```nginx
location / {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 300;
    proxy_connect_timeout 300;
    proxy_send_timeout 300;
}
```

Buka:

```text
aaPanel -> Website -> bot.gungzy.xyz -> Config
```

Tambahkan blok:

```nginx
location = /webhook {
    proxy_pass http://127.0.0.1:8081;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 300;
    proxy_connect_timeout 300;
    proxy_send_timeout 300;
}

location / {
    return 404;
}
```

Pastikan tidak ada dua blok `location /` yang aktif di satu server block.

Test dan reload Nginx dari SSH VPS:

```bash
/www/server/nginx/sbin/nginx -t
systemctl reload nginx
```

Jika `systemctl reload nginx` tidak cocok dengan instalasi aaPanel, reload dari:

```text
aaPanel -> App Store -> Nginx -> Restart/Reload
```

## 4. Firewall

### 4.1. Firewall VPS

Di aaPanel:

```text
aaPanel -> Security
```

Buka hanya port berikut:

| Port | Fungsi |
| ---: | --- |
| `22` atau port SSH kamu | SSH dan reverse tunnel |
| `80` | HTTP / Let's Encrypt |
| `443` | HTTPS |
| port aaPanel | Login panel, biasanya `8888` jika belum diganti |

Jangan buka port berikut ke publik:

| Port | Alasan |
| ---: | --- |
| `8000` | Core API hanya lewat SSH tunnel lokal |
| `8080` | Mini App API hanya untuk Nginx lokal |
| `8081` | Bot webhook listener hanya untuk Nginx lokal |

Port `8000`, `8080`, dan `8081` cukup listen di VPS dan diakses oleh `127.0.0.1`.

### 4.2. Firewall Windows

Untuk mode reverse tunnel, laptop tidak perlu menerima koneksi masuk dari internet. Laptop hanya membuat koneksi keluar SSH ke VPS.

Yang perlu dipastikan:

- PowerShell/Windows Terminal boleh melakukan koneksi keluar ke port SSH VPS.
- Jika SSH VPS di port `22`, outbound TCP `22` tidak diblokir.
- Jika SSH VPS memakai port custom, outbound TCP port custom itu tidak diblokir.

Tidak perlu membuka inbound port `8000` di Windows Firewall karena Core API hanya listen di `127.0.0.1`.

## 5. Setup project di laptop

Masuk ke project:

```powershell
cd "D:\Data cokagung\Skripsi\tele-new"
```

Siapkan virtualenv jika belum:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Pastikan model ada:

```powershell
dir data\model
```

Minimal ada:

```text
config.json
model.safetensors
tokenizer.json
vocab.txt
```

## 6. `.env` laptop

Laptop menjalankan Core API, jadi `.env` laptop harus punya token dan API key yang sama dengan VPS.

Contoh bagian penting:

```env
BOT_TOKEN=ISI_TOKEN_BOT_TELEGRAM
API_KEY=ISI_API_KEY_YANG_SAMA_DENGAN_VPS
SECRET_KEY=ISI_SECRET_KEY

MODEL_PATH=./data/model
TOKENIZER_PATH=./data/model/tokenizer.json
USE_CUDA=0

INFERENCE_API_URL=http://127.0.0.1:8000

DATABASE_URL=sqlite:///./data/db.sqlite3
RETENTION_DAYS=30
DEFAULT_THRESHOLD=0.83
ADMIN_IDS=ISI_USER_ID_TELEGRAM_ADMIN

MINI_APP_BASE_URL=https://mini.gungzy.xyz
MINI_APP_DEV_MODE=false
ALLOWED_ORIGINS=https://mini.gungzy.xyz
```

Catatan:

- `API_KEY` laptop dan VPS harus sama.
- `BOT_TOKEN` tetap diperlukan di laptop karena beberapa endpoint Core API bisa memanggil Telegram API.
- `DATABASE_URL` di laptop berarti data settings/stats tersimpan di laptop.

## 7. Jalankan Core API di laptop

Terminal PowerShell pertama:

```powershell
cd "D:\Data cokagung\Skripsi\tele-new"
.\.venv\Scripts\Activate.ps1
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Cek dari PowerShell kedua:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

Status yang diharapkan:

```text
status: ok
model_loaded: True
tokenizer_loaded: True
```

Jika laptop RAM cukup tetapi load awal lama, tunggu sampai uvicorn selesai start. Jangan jalankan model di VPS 1GB.

## 8. Buka SSH reverse tunnel dari laptop ke VPS

Terminal PowerShell kedua:

```powershell
ssh -N `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -R 127.0.0.1:8000:127.0.0.1:8000 `
  root@IP_VPS_BARU
```

Jika SSH memakai port custom, contoh:

```powershell
ssh -p 2222 -N `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -R 127.0.0.1:8000:127.0.0.1:8000 `
  root@IP_VPS_BARU
```

Biarkan terminal tunnel ini tetap hidup.

Test dari VPS:

```bash
curl http://127.0.0.1:8000/healthz
```

Jika berhasil, response itu berasal dari Core API di laptop.

Jika gagal, cek di VPS:

```bash
ss -ltnp | grep ':8000'
```

Jika port `8000` sudah dipakai proses lain di VPS, matikan Core API lokal:

```bash
pkill -f "uvicorn.*api.app:app" || true
```

Lalu ulangi command SSH tunnel dari laptop.

## 9. Setup project di VPS

Masuk SSH VPS:

```bash
ssh root@IP_VPS_BARU
```

Pastikan dependency runtime tersedia:

```bash
apt update
apt install -y git curl unzip build-essential python3 python3-venv python3-pip sqlite3
```

Install Node.js dari OS, bukan aaPanel:

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt install -y nodejs
node -v
npm -v
```

Project ada di:

```bash
cd /www/wwwroot/telegram-ta
```

Install Python dependency untuk Mini App API:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Build frontend dan bot. Untuk VPS 1GB, lebih aman build di laptop lalu upload hasil build. Kalau harus build di VPS, jalankan saat Core API tidak hidup:

```bash
pkill -f "uvicorn.*api.app:app" || true
npm install --prefix referensi-desain
NODE_OPTIONS=--max-old-space-size=768 npm run build --prefix referensi-desain
npm install --prefix bot
npm run build --prefix bot
```

## 10. `.env` VPS

VPS tidak menjalankan Core API, tetapi Mini App dan bot tetap memanggil Core API melalui tunnel di `127.0.0.1:8000`.

Contoh bagian penting:

```env
BOT_TOKEN=ISI_TOKEN_BOT_TELEGRAM
API_KEY=ISI_API_KEY_YANG_SAMA_DENGAN_LAPTOP
SECRET_KEY=ISI_SECRET_KEY

INFERENCE_API_URL=http://127.0.0.1:8000

WEBHOOK_URL=https://bot.gungzy.xyz/webhook
WEBHOOK_HOST=127.0.0.1
WEBHOOK_PORT=8081
BOT_WEBHOOK_PORT=8081
WEBHOOK_PATH=/webhook
WEBHOOK_SECRET=ISI_SECRET_WEBHOOK_RANDOM

MINI_APP_BASE_URL=https://mini.gungzy.xyz
MINI_APP_DEV_MODE=false
ALLOWED_ORIGINS=https://mini.gungzy.xyz

DATABASE_URL=sqlite:///./data/db.sqlite3
ADMIN_IDS=ISI_USER_ID_TELEGRAM_ADMIN
```

Catatan:

- `API_KEY` harus sama dengan laptop.
- `WEBHOOK_URL` harus domain VPS.
- `MINI_APP_BASE_URL` harus domain VPS.
- `DATABASE_URL` di VPS tidak terlalu dipakai oleh bot/Mini App untuk data utama jika Core API ada di laptop, tetapi tetap boleh ada.

Amankan `.env`:

```bash
chmod 600 .env
```

## 11. Jalankan edge service di VPS

Pastikan Core API lokal VPS tidak berjalan:

```bash
pkill -f "uvicorn.*api.app:app" || true
```

Jalankan mode edge:

```bash
cd /www/wwwroot/telegram-ta
mkdir -p logs
chmod +x scripts/manage-new.sh
./scripts/manage-new.sh restart-edge
```

Cek status:

```bash
./scripts/manage-new.sh status
```

Catatan: status bisa menampilkan `core API not running`. Itu normal untuk mode edge, karena Core API tidak dijalankan sebagai proses VPS. Yang penting command ini berhasil:

```bash
curl http://127.0.0.1:8000/healthz
```

Dan proses ini berjalan:

```bash
pgrep -a "uvicorn.*web.main:app"
pm2 list
```

## 12. Register webhook Telegram

Bot akan otomatis set webhook saat start jika `WEBHOOK_URL` terisi.

Cek:

```bash
cd /www/wwwroot/telegram-ta
BOT_TOKEN=$(grep '^BOT_TOKEN=' .env | cut -d= -f2-)
curl "https://api.telegram.org/bot${BOT_TOKEN}/getWebhookInfo"
```

Hasil yang benar:

- `url` berisi `https://bot.gungzy.xyz/webhook`
- `last_error_message` kosong
- `pending_update_count` tidak terus naik

Jika masih mengarah ke VPS lama:

```bash
curl "https://api.telegram.org/bot${BOT_TOKEN}/deleteWebhook?drop_pending_updates=true"
./scripts/manage-new.sh restart-edge
curl "https://api.telegram.org/bot${BOT_TOKEN}/getWebhookInfo"
```

## 13. BotFather Mini App URL

Di Telegram:

```text
@BotFather
/mybots
Pilih bot
Bot Settings
Menu Button atau Configure Mini App
Set URL ke https://mini.gungzy.xyz
```

Command `/start` juga memakai `MINI_APP_BASE_URL` dari `.env` VPS, jadi restart bot setelah mengubah URL:

```bash
cd /www/wwwroot/telegram-ta
./scripts/manage-new.sh restart-edge
```

## 14. Urutan running harian

Urutan paling aman:

1. Nyalakan laptop dan pastikan tidak sleep.
2. Jalankan Core API di laptop:

```powershell
cd "D:\Data cokagung\Skripsi\tele-new"
.\.venv\Scripts\Activate.ps1
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

3. Jalankan reverse tunnel dari laptop:

```powershell
ssh -N `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -R 127.0.0.1:8000:127.0.0.1:8000 `
  root@IP_VPS_BARU
```

4. Di VPS, test tunnel:

```bash
curl http://127.0.0.1:8000/healthz
```

5. Di VPS, jalankan edge service:

```bash
cd /www/wwwroot/telegram-ta
./scripts/manage-new.sh restart-edge
```

6. Test Telegram:

- Kirim `/start` ke bot.
- Buka dashboard.
- Jalankan `/test teks percobaan`.

## 15. Optional: tunnel auto-reconnect di laptop

Manual terminal cukup untuk testing. Untuk pemakaian lebih stabil, buat script PowerShell.

Buat file:

```text
D:\Data cokagung\Skripsi\tele-new\Start-Tunnel.ps1
```

Isi:

```powershell
$VpsHost = "root@IP_VPS_BARU"

while ($true) {
  ssh -N `
    -o ExitOnForwardFailure=yes `
    -o ServerAliveInterval=30 `
    -o ServerAliveCountMax=3 `
    -R 127.0.0.1:8000:127.0.0.1:8000 `
    $VpsHost

  Start-Sleep -Seconds 5
}
```

Jalankan:

```powershell
powershell -ExecutionPolicy Bypass -File "D:\Data cokagung\Skripsi\tele-new\Start-Tunnel.ps1"
```

Untuk autostart Windows, pakai Task Scheduler:

```text
Task Scheduler -> Create Task
Trigger: At log on
Action: Start a program
Program: powershell.exe
Arguments: -ExecutionPolicy Bypass -File "D:\Data cokagung\Skripsi\tele-new\Start-Tunnel.ps1"
```

Tetap pastikan Core API juga dijalankan setelah laptop menyala.

## 16. Power settings laptop

Agar bot tidak putus:

```text
Windows Settings -> System -> Power & battery
```

Set:

- Sleep: Never saat plugged in
- Hibernate: off jika perlu
- Matikan "turn off network adapter to save power" jika koneksi sering putus

Opsional dari PowerShell admin:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

## 17. Troubleshooting

### VPS tidak bisa akses `127.0.0.1:8000`

Cek tunnel dari VPS:

```bash
ss -ltnp | grep ':8000'
curl http://127.0.0.1:8000/healthz
```

Jika tidak ada listener, tunnel belum jalan atau gagal bind.

Solusi:

```bash
pkill -f "uvicorn.*api.app:app" || true
```

Lalu ulangi command SSH tunnel dari laptop.

### SSH tunnel gagal dengan `remote port forwarding failed`

Biasanya port `8000` di VPS sudah dipakai.

Cek:

```bash
ss -ltnp | grep ':8000'
```

Matikan proses yang memakai port:

```bash
pkill -f "uvicorn.*api.app:app" || true
```

Jika SSH server menolak forwarding, cek `/etc/ssh/sshd_config` di VPS:

```bash
grep -E 'AllowTcpForwarding|GatewayPorts' /etc/ssh/sshd_config
```

Nilai aman:

```text
AllowTcpForwarding yes
```

Untuk mode ini tidak perlu `GatewayPorts yes`, karena tunnel hanya bind ke `127.0.0.1`.

Restart SSH jika mengubah config:

```bash
systemctl restart ssh
```

### Mini App error `Core API unreachable`

Di VPS:

```bash
curl http://127.0.0.1:8000/healthz
tail -n 120 /www/wwwroot/telegram-ta/logs/mini-api.out
```

Penyebab umum:

- Core API laptop belum jalan.
- SSH tunnel mati.
- `API_KEY` laptop dan VPS berbeda.
- `.env` VPS belum memakai `INFERENCE_API_URL=http://127.0.0.1:8000`.

### Bot tidak membalas atau moderasi gagal

Di VPS:

```bash
cd /www/wwwroot/telegram-ta
pm2 list
pm2 logs hate-guard-bot --lines 120
BOT_TOKEN=$(grep '^BOT_TOKEN=' .env | cut -d= -f2-)
curl "https://api.telegram.org/bot${BOT_TOKEN}/getWebhookInfo"
```

Pastikan:

- bot PM2 online
- webhook URL benar
- tunnel ke Core API hidup
- bot sudah admin di grup Telegram

### Nginx 502 di `mini.gungzy.xyz`

Cek service Mini App di VPS:

```bash
pgrep -a "uvicorn.*web.main:app"
tail -n 120 /www/wwwroot/telegram-ta/logs/mini-api.out
tail -n 80 /www/wwwlogs/mini.gungzy.xyz.error.log
```

Restart edge:

```bash
cd /www/wwwroot/telegram-ta
./scripts/manage-new.sh restart-edge
```

### Dashboard menolak akses admin

Pastikan `ADMIN_IDS` di `.env` laptop berisi user ID Telegram kamu, karena Core API admin check berjalan di laptop.

Cek user ID dengan:

```text
/getid
```

Setelah edit `.env` laptop, restart Core API di laptop.

### Core API laptop stuck saat start

Ini normal beberapa saat saat model pertama kali load. Jika laptop juga freeze, laptop RAM kurang untuk model PyTorch.

Solusi:

- Tutup aplikasi berat.
- Pastikan `USE_CUDA=0` jika tidak memakai GPU.
- Gunakan laptop dengan RAM lebih besar.
- Refactor inference ke ONNX/quantized model jika ingin lebih ringan.

## 18. Checklist akhir

Sebelum dipakai:

- `https://mini.gungzy.xyz` SSL valid.
- `https://bot.gungzy.xyz/webhook` reachable dari internet.
- aaPanel firewall membuka `22`, `80`, `443`, dan port panel.
- aaPanel firewall tidak membuka `8000`, `8080`, `8081`.
- Core API laptop berjalan di `127.0.0.1:8000`.
- SSH reverse tunnel laptop ke VPS hidup.
- Dari VPS, `curl http://127.0.0.1:8000/healthz` berhasil.
- Di VPS, `./scripts/manage-new.sh restart-edge` berhasil.
- PM2 menunjukkan `hate-guard-bot` online.
- `getWebhookInfo` mengarah ke `https://bot.gungzy.xyz/webhook`.
- Bot sudah admin di grup Telegram.
- Laptop tidak sleep/hibernate.
