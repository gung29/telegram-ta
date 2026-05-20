# Setup Project di VPS Baru dengan aaPanel

Panduan ini menjelaskan setup dari nol untuk menjalankan bot Telegram, Core API, dan Telegram Mini App di VPS baru.

Asumsi yang dipakai:

- OS VPS: Ubuntu 22.04 atau 24.04.
- VPS sudah memakai aaPanel.
- aaPanel dipakai untuk membuat website/subdomain, SSL, dan konfigurasi Nginx.
- Python dan Node.js tidak memakai runtime dari aaPanel.
- Folder deploy: `/www/wwwroot/telegram-ta`.
- Domain utama: `gungzy.xyz`.
- Mini App: `https://mini.gungzy.xyz`.
- Bot webhook: `https://bot.gungzy.xyz/webhook`.
- Core API hanya dipakai internal VPS di `http://127.0.0.1:8000`.

Project ini menjalankan tiga proses:

| Komponen | Command | Port lokal | Akses publik |
| --- | --- | ---: | --- |
| Core API | `uvicorn api.app:app` | `8000` | Tidak wajib publik |
| Mini App API + static React | `uvicorn web.main:app` | `8080` | `mini.gungzy.xyz` |
| Bot webhook server | `npm run start --prefix bot` | `8081` | `bot.gungzy.xyz/webhook` |

Catatan penting: Telegram Mini App dan webhook harus memakai HTTPS publik. Karena kamu memakai aaPanel, gunakan aaPanel untuk add site/subdomain, SSL Let's Encrypt, dan Nginx reverse proxy. Jangan pakai menu Node/Python bawaan aaPanel untuk menjalankan aplikasi ini karena versi runtime sering kurang cocok dengan dependency project.

## 1. Arahkan DNS subdomain

Di panel DNS domain `gungzy.xyz`, buat record berikut:

| Type | Name | Value | Proxy |
| --- | --- | --- | --- |
| A | `mini` | IP publik VPS baru | DNS only disarankan |
| A | `bot` | IP publik VPS baru | DNS only disarankan |
| A | `api` | IP publik VPS baru | Opsional, DNS only |

`api.gungzy.xyz` opsional. Untuk project ini, Core API sudah cukup diakses internal oleh Mini App dan bot melalui `127.0.0.1:8000`.

Jika VPS punya IPv6, boleh tambahkan record `AAAA` untuk subdomain yang sama.

Cek propagasi DNS:

```bash
dig +short mini.gungzy.xyz
dig +short bot.gungzy.xyz
```

Dari Windows bisa cek dengan:

```powershell
nslookup mini.gungzy.xyz
nslookup bot.gungzy.xyz
```

Pastikan hasilnya mengarah ke IP VPS baru sebelum lanjut membuat SSL.

## 2. Login ke VPS

Dari Windows PowerShell:

```powershell
ssh root@IP_VPS_BARU
```

Update paket dasar:

```bash
apt update && apt upgrade -y
apt install -y git curl unzip build-essential python3 python3-venv python3-pip sqlite3 dnsutils
```

Nginx dan SSL akan diatur dari aaPanel. Jika Nginx belum terpasang di aaPanel, install dari:

```text
aaPanel -> App Store -> Nginx -> Install
```

Jangan install atau menjalankan project lewat:

```text
aaPanel -> App Store -> Python Manager
aaPanel -> App Store -> Node.js version manager / Node Project
```

Python dan Node.js untuk project ini tetap dari SSH/OS supaya versi dan dependency lebih konsisten.

Install Node.js. Gunakan Node 20 atau 22 LTS. Contoh dengan Node 22:

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt install -y nodejs
node -v
npm -v
```

Jika VPS RAM kecil, buat swap agar install dependency dan load model lebih aman:

```bash
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
free -h
```

## 3. Buat folder deploy

Folder ini sengaja dibuat sama seperti `scripts/manage-new.sh`:

```bash
mkdir -p /www/wwwroot
cd /www/wwwroot
```

Jika project ada di Git remote:

```bash
git clone REPO_URL telegram-ta
cd /www/wwwroot/telegram-ta
```

Jika project belum ada di Git remote, upload dari komputer lokal. Dari PowerShell Windows, jalankan:

```powershell
scp -r "D:\Data cokagung\Skripsi\tele-new" root@IP_VPS_BARU:/www/wwwroot/telegram-ta
```

Setelah upload, kembali ke SSH VPS:

```bash
cd /www/wwwroot/telegram-ta
```

Pastikan file model ada:

```bash
ls -lh data/model
```

Minimal folder `data/model` harus punya file seperti:

```text
config.json
model.safetensors
tokenizer.json
vocab.txt
```

Jika ingin membawa database lama, pastikan juga ada:

```bash
ls -lh data/db.sqlite3
```

Jika database belum ada, aplikasi akan membuat tabel saat Core API pertama kali start.

## 4. Buat file `.env`

Buat atau edit `.env`:

```bash
cd /www/wwwroot/telegram-ta
nano .env
```

Template production:

```env
BOT_TOKEN=ISI_TOKEN_BOT_TELEGRAM
API_KEY=ISI_API_KEY_RANDOM
SECRET_KEY=ISI_SECRET_KEY_RANDOM

MODEL_PATH=./data/model
TOKENIZER_PATH=./data/model/tokenizer.json
USE_CUDA=0
INTRA_OP_THREADS=2
INTER_OP_THREADS=1

INFERENCE_API_HOST=127.0.0.1
INFERENCE_API_PORT=8000
INFERENCE_API_URL=http://127.0.0.1:8000

WEBHOOK_URL=https://bot.gungzy.xyz/webhook
WEBHOOK_HOST=127.0.0.1
WEBHOOK_PORT=8081
BOT_WEBHOOK_PORT=8081
WEBHOOK_PATH=/webhook
WEBHOOK_SECRET=ISI_SECRET_WEBHOOK_RANDOM

DATABASE_URL=sqlite:///./data/db.sqlite3
RETENTION_DAYS=30
DEFAULT_THRESHOLD=0.83

ADMIN_IDS=ISI_USER_ID_TELEGRAM_ADMIN
MODERATE_ADMINS=true

MINI_APP_BASE_URL=https://mini.gungzy.xyz
ALLOWED_ORIGINS=https://mini.gungzy.xyz
MINI_APP_DEV_MODE=false

TELEMETRY_ENABLED=true
```

Buat nilai random untuk `API_KEY`, `SECRET_KEY`, dan `WEBHOOK_SECRET`:

```bash
openssl rand -hex 32
```

Catatan:

- `BOT_TOKEN` ambil dari `@BotFather`.
- `ADMIN_IDS` isi dengan user ID Telegram admin. Jika lebih dari satu, pisahkan koma: `123,456`.
- `WEBHOOK_URL` wajib mengarah ke subdomain bot dan path `/webhook`.
- `MINI_APP_BASE_URL` wajib mengarah ke subdomain Mini App.
- `MINI_APP_DEV_MODE=false` untuk VPS production.
- Jangan pakai URL localhost untuk `MINI_APP_BASE_URL` di VPS production.

Amankan `.env`:

```bash
chmod 600 .env
```

## 5. Install dependency Python

```bash
cd /www/wwwroot/telegram-ta
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Test import aplikasi:

```bash
python -c "from common.config import settings; print(settings.mini_app_base_url)"
```

Jika ini gagal, cek lagi `.env` dan dependency Python.

## 6. Install dependency Node dan build frontend

Project ini melayani Mini App dari `referensi-desain/dist`, jadi build folder `referensi-desain`.

```bash
cd /www/wwwroot/telegram-ta
npm install --prefix referensi-desain
npm run build --prefix referensi-desain

npm install --prefix bot
npm run build --prefix bot
```

Cek hasil build:

```bash
ls -lh referensi-desain/dist
ls -lh bot/dist
```

## 7. Buat service systemd

Service systemd membuat aplikasi otomatis hidup setelah reboot dan lebih rapi dibanding `nohup`.

### Core API service

```bash
cat >/etc/systemd/system/hate-guard-core.service <<'EOF'
[Unit]
Description=Hate Guard Core API
After=network.target

[Service]
Type=simple
WorkingDirectory=/www/wwwroot/telegram-ta
EnvironmentFile=/www/wwwroot/telegram-ta/.env
ExecStart=/www/wwwroot/telegram-ta/.venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

### Mini App service

```bash
cat >/etc/systemd/system/hate-guard-mini.service <<'EOF'
[Unit]
Description=Hate Guard Telegram Mini App
After=network.target hate-guard-core.service
Requires=hate-guard-core.service

[Service]
Type=simple
WorkingDirectory=/www/wwwroot/telegram-ta
EnvironmentFile=/www/wwwroot/telegram-ta/.env
ExecStart=/www/wwwroot/telegram-ta/.venv/bin/uvicorn web.main:app --host 127.0.0.1 --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

### Telegram bot service

Cek lokasi `npm`:

```bash
which npm
```

Biasanya hasilnya `/usr/bin/npm`. Jika berbeda, sesuaikan `ExecStart`.

```bash
cat >/etc/systemd/system/hate-guard-bot.service <<'EOF'
[Unit]
Description=Hate Guard Telegram Bot
After=network.target hate-guard-core.service
Requires=hate-guard-core.service

[Service]
Type=simple
WorkingDirectory=/www/wwwroot/telegram-ta/bot
EnvironmentFile=/www/wwwroot/telegram-ta/.env
ExecStart=/usr/bin/npm run start
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

Aktifkan service:

```bash
systemctl daemon-reload
systemctl enable hate-guard-core hate-guard-mini hate-guard-bot
systemctl start hate-guard-core
systemctl start hate-guard-mini
systemctl start hate-guard-bot
```

Cek status:

```bash
systemctl status hate-guard-core --no-pager
systemctl status hate-guard-mini --no-pager
systemctl status hate-guard-bot --no-pager
```

Lihat log:

```bash
journalctl -u hate-guard-core -f
journalctl -u hate-guard-mini -f
journalctl -u hate-guard-bot -f
```

## 8. Test port lokal di VPS

Core API:

```bash
curl http://127.0.0.1:8000/healthz
```

Mini App:

```bash
curl -I http://127.0.0.1:8080
```

Bot webhook listener:

```bash
curl -I http://127.0.0.1:8081/webhook
```

Untuk endpoint webhook, hasil `404`, `401`, atau `405` masih wajar saat memakai `curl -I`, karena route webhook menerima `POST` dan memeriksa secret header. Yang penting port `8081` merespon dan service bot tidak crash.

## 9. Tambahkan subdomain di aaPanel

Di aaPanel, pastikan Nginx sudah terinstall dari `App Store`. Setelah DNS `mini.gungzy.xyz` dan `bot.gungzy.xyz` mengarah ke IP VPS, buat dua website:

```text
aaPanel -> Website -> Add site
```

Website pertama:

| Field | Isi |
| --- | --- |
| Domain name | `mini.gungzy.xyz` |
| Root directory | `/www/wwwroot/mini.gungzy.xyz` |
| FTP | Tidak perlu |
| Database | Tidak perlu |
| PHP version | Static / Pure static / Not set |

Website kedua:

| Field | Isi |
| --- | --- |
| Domain name | `bot.gungzy.xyz` |
| Root directory | `/www/wwwroot/bot.gungzy.xyz` |
| FTP | Tidak perlu |
| Database | Tidak perlu |
| PHP version | Static / Pure static / Not set |

Root directory ini hanya placeholder agar aaPanel mau membuat site config. Aplikasi sebenarnya tetap berjalan dari `/www/wwwroot/telegram-ta` lewat service `systemd`.

Jangan membuat project ini lewat menu Node Project atau Python Manager aaPanel.

## 10. Pasang SSL dari aaPanel

Untuk masing-masing site:

```text
aaPanel -> Website -> pilih domain -> SSL -> Let's Encrypt
```

Pilih verifikasi HTTP/File Verification, lalu apply certificate.

Aktifkan:

```text
Force HTTPS
```

Cek dari browser:

```text
https://mini.gungzy.xyz
https://bot.gungzy.xyz
```

Pada tahap ini halaman boleh masih 404/default aaPanel. Yang penting SSL valid dulu.

## 11. Atur Nginx reverse proxy lewat aaPanel

Buka config site dari aaPanel:

```text
aaPanel -> Website -> mini.gungzy.xyz -> Config
```

Jangan hapus baris `listen`, `server_name`, `ssl_certificate`, `ssl_certificate_key`, log, dan konfigurasi Let's Encrypt yang dibuat aaPanel. Fokus ubah bagian `location /` untuk diarahkan ke Mini App lokal.

Untuk `mini.gungzy.xyz`, pakai blok ini:

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

Lalu buka:

```text
aaPanel -> Website -> bot.gungzy.xyz -> Config
```

Untuk `bot.gungzy.xyz`, arahkan hanya path `/webhook` ke bot:

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

Jika aaPanel menambahkan blok PHP seperti `include enable-php-...conf`, boleh dibiarkan selama tidak mengganggu. Pastikan tidak ada dua blok `location /` yang aktif dalam satu `server` block. Untuk site `bot.gungzy.xyz`, lebih rapi jika request selain `/webhook` memang `404`.

Setelah save config, test dan reload Nginx dari SSH:

```bash
/www/server/nginx/sbin/nginx -t
systemctl reload nginx
```

Jika `systemctl reload nginx` tidak tersedia di instalasi aaPanel tertentu, reload dari panel:

```text
aaPanel -> App Store -> Nginx -> Restart/Reload
```

Cek:

```bash
curl -I https://mini.gungzy.xyz
curl -I https://bot.gungzy.xyz/webhook
```

Untuk `https://bot.gungzy.xyz/webhook`, status `401`, `404`, atau `405` masih wajar dengan request manual. Telegram akan mengirim `POST` dengan secret token.

## 12. Setup firewall dari aaPanel

Gunakan firewall/security aaPanel:

```text
aaPanel -> Security
```

Pastikan port ini terbuka:

| Port | Fungsi |
| ---: | --- |
| `22` atau port SSH kamu | SSH |
| `80` | HTTP dan Let's Encrypt |
| `443` | HTTPS |
| port aaPanel kamu | Login panel, biasanya `8888` jika belum diganti |

Jangan buka port ini ke publik:

| Port | Fungsi |
| ---: | --- |
| `8000` | Core API internal |
| `8080` | Mini App API internal |
| `8081` | Bot webhook listener internal |

Port `8000`, `8080`, dan `8081` cukup diakses oleh Nginx lokal melalui `127.0.0.1`.

## 13. Register webhook Telegram

Saat `hate-guard-bot` start, kode di `bot/src/index.ts` otomatis menjalankan `setWebHook` jika `WEBHOOK_URL` terisi.

Tetap cek status webhook:

```bash
BOT_TOKEN=$(grep '^BOT_TOKEN=' .env | cut -d= -f2-)
curl "https://api.telegram.org/bot${BOT_TOKEN}/getWebhookInfo"
```

Hasil yang benar:

- `url` berisi `https://bot.gungzy.xyz/webhook`.
- `last_error_message` kosong atau tidak ada.
- `pending_update_count` tidak terus bertambah.

Jika webhook masih mengarah ke VPS lama:

```bash
curl "https://api.telegram.org/bot${BOT_TOKEN}/deleteWebhook?drop_pending_updates=true"
systemctl restart hate-guard-bot
curl "https://api.telegram.org/bot${BOT_TOKEN}/getWebhookInfo"
```

## 14. Set Mini App URL di BotFather

Buka Telegram dan masuk ke `@BotFather`.

Jalur umum:

```text
/mybots
Pilih bot
Bot Settings
Menu Button atau Configure Mini App
Set URL ke https://mini.gungzy.xyz
```

Jika bot punya Main Mini App atau tombol menu yang sebelumnya mengarah ke VPS lama, ganti semuanya ke:

```text
https://mini.gungzy.xyz
```

Command `/start` dari bot juga memakai `MINI_APP_BASE_URL`. Setelah `.env` benar dan service bot direstart, tombol dashboard dari `/start` akan memakai URL baru.

## 15. Test dari Telegram

1. Buka chat private dengan bot, kirim `/start`.
2. Pastikan tombol `Open Dashboard` muncul.
3. Buka dashboard.
4. Tambahkan bot ke grup Telegram.
5. Jadikan bot admin grup jika ingin fitur hapus/mute/ban bekerja.
6. Di grup, kirim `/start` atau `/getid`.
7. Jalankan `/test teks percobaan`.
8. Jalankan `/moderation_on`.

Jika dashboard menolak akses admin:

- Pastikan `ADMIN_IDS` berisi user ID Telegram admin.
- Restart Core API dan Mini App setelah mengubah `.env`.
- Jalankan `/getid` untuk memastikan user ID yang benar.

Restart service:

```bash
systemctl restart hate-guard-core hate-guard-mini hate-guard-bot
```

## 16. Optional: subdomain `api.gungzy.xyz`

Secara default Core API tidak perlu publik. Jika tetap ingin subdomain API untuk debugging, buat DNS `api.gungzy.xyz`, lalu add site dari aaPanel seperti site lain.

```text
aaPanel -> Website -> Add site
Domain name: api.gungzy.xyz
Root directory: /www/wwwroot/api.gungzy.xyz
PHP version: Static / Pure static / Not set
Database: Tidak perlu
```

Pasang SSL dari aaPanel:

```text
aaPanel -> Website -> api.gungzy.xyz -> SSL -> Let's Encrypt
```

Lalu buka:

```text
aaPanel -> Website -> api.gungzy.xyz -> Config
```

Ubah `location /` menjadi:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
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

Cek dan reload Nginx:

```bash
/www/server/nginx/sbin/nginx -t
systemctl reload nginx
```

Peringatan: endpoint admin dan predict memakai `X-API-Key`, tetapi membuka Core API ke publik tetap menambah risiko. Untuk production, lebih aman biarkan Core API internal saja.

## 17. Update deploy berikutnya

Jika ada update kode:

```bash
cd /www/wwwroot/telegram-ta
git pull
source .venv/bin/activate
pip install -r requirements.txt
npm install --prefix referensi-desain
npm run build --prefix referensi-desain
npm install --prefix bot
npm run build --prefix bot
systemctl restart hate-guard-core hate-guard-mini hate-guard-bot
```

Cek log setelah restart:

```bash
journalctl -u hate-guard-core -n 80 --no-pager
journalctl -u hate-guard-mini -n 80 --no-pager
journalctl -u hate-guard-bot -n 80 --no-pager
```

## 18. Alternatif hemat RAM: Core API di laptop, bot dan Mini App di VPS

Mode ini cocok untuk VPS 1GB RAM. VPS hanya menjalankan:

- Mini App API di port `8080`
- Bot webhook listener di port `8081`
- Nginx aaPanel untuk `mini.gungzy.xyz` dan `bot.gungzy.xyz`

Core API/model jalan di laptop lokal, lalu diekspos ke VPS melalui SSH reverse tunnel. Dengan cara ini, Core API tetap terlihat sebagai `http://127.0.0.1:8000` dari sisi VPS, tetapi proses model sebenarnya hidup di laptop.

Kelemahan mode ini:

- Laptop harus menyala terus.
- Laptop jangan sleep/hibernate.
- Jika internet laptop mati, bot tidak bisa moderasi karena Core API putus.
- Database SQLite Core API akan berada di laptop jika `DATABASE_URL=sqlite:///./data/db.sqlite3`.

### 18.1. Jalankan Core API di laptop

Di laptop Windows:

```powershell
cd "D:\Data cokagung\Skripsi\tele-new"
.\.venv\Scripts\Activate.ps1
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Cek dari laptop:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

Pastikan `.env` laptop punya `API_KEY`, `BOT_TOKEN`, `ADMIN_IDS`, dan setting model yang benar.

### 18.2. Buka reverse tunnel dari laptop ke VPS

Buka PowerShell baru di laptop:

```powershell
ssh -N `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -R 127.0.0.1:8000:127.0.0.1:8000 `
  root@IP_VPS_BARU
```

Terminal tunnel ini harus tetap hidup. Dari VPS, test:

```bash
curl http://127.0.0.1:8000/healthz
```

Jika berhasil, request itu sebenarnya masuk ke Core API di laptop.

### 18.3. Set `.env` di VPS

Di VPS, `.env` tetap memakai Core API lokal karena port `8000` sudah ditunnel:

```env
INFERENCE_API_URL=http://127.0.0.1:8000
WEBHOOK_URL=https://bot.gungzy.xyz/webhook
MINI_APP_BASE_URL=https://mini.gungzy.xyz
MINI_APP_DEV_MODE=false
```

`API_KEY` di VPS harus sama dengan `API_KEY` di laptop, karena bot dan Mini App VPS mengirim `X-API-Key` ke Core API laptop.

### 18.4. Jalankan hanya edge service di VPS

Jangan start Core API di VPS. Gunakan mode `start-edge`:

```bash
cd /www/wwwroot/telegram-ta
chmod +x scripts/manage-new.sh
./scripts/manage-new.sh restart-edge
./scripts/manage-new.sh status
```

Mode `start-edge` hanya menjalankan Mini App dan bot. Core API tidak diload di VPS, sehingga VPS 1GB tidak freeze.

Jika script di VPS belum punya mode `start-edge`, upload versi terbaru `scripts/manage-new.sh` dari project lokal ini.

## 19. Alternatif: memakai `scripts/manage-new.sh`

Repo ini sudah punya script:

```bash
scripts/manage-new.sh
```

Script tersebut memakai path:

```bash
BASE="/www/wwwroot/telegram-ta"
```

Jika ingin memakai script ini, pastikan project memang ada di path tersebut.

Setup awal:

```bash
cd /www/wwwroot/telegram-ta
mkdir -p logs
chmod +x scripts/manage-new.sh
npm install -g pm2
./scripts/manage-new.sh restart
./scripts/manage-new.sh status
```

Untuk autostart bot PM2:

```bash
pm2 save
pm2 startup
```

Catatan: `manage-new.sh` menjalankan Core API dan Mini App dengan `nohup`, bukan systemd. Untuk VPS production baru, service systemd pada panduan ini lebih mudah diawasi dan lebih aman setelah reboot. Jangan jalankan systemd dan `manage-new.sh` bersamaan karena bisa bentrok port.

## 20. Troubleshooting

### Domain belum bisa SSL

Cek DNS:

```bash
dig +short mini.gungzy.xyz
dig +short bot.gungzy.xyz
```

Pastikan hasilnya IP VPS baru. Jika memakai Cloudflare, pakai mode DNS only dulu sampai certificate berhasil dibuat.

### Mini App terbuka tetapi blank atau asset 404

Build ulang frontend:

```bash
cd /www/wwwroot/telegram-ta
npm run build --prefix referensi-desain
systemctl restart hate-guard-mini
```

### Dashboard error `Core API unreachable`

Pastikan Core API hidup:

```bash
systemctl status hate-guard-core --no-pager
curl http://127.0.0.1:8000/healthz
```

Pastikan `.env`:

```env
INFERENCE_API_URL=http://127.0.0.1:8000
```

Restart:

```bash
systemctl restart hate-guard-mini hate-guard-bot
```

### Bot tidak menerima pesan

Cek webhook:

```bash
BOT_TOKEN=$(grep '^BOT_TOKEN=' .env | cut -d= -f2-)
curl "https://api.telegram.org/bot${BOT_TOKEN}/getWebhookInfo"
```

Cek log bot:

```bash
journalctl -u hate-guard-bot -n 120 --no-pager
```

Pastikan `.env`:

```env
WEBHOOK_URL=https://bot.gungzy.xyz/webhook
WEBHOOK_HOST=127.0.0.1
BOT_WEBHOOK_PORT=8081
WEBHOOK_PATH=/webhook
```

Restart bot:

```bash
systemctl restart hate-guard-bot
```

### Nginx 502 Bad Gateway

Artinya Nginx tidak bisa menghubungi service lokal.

Cek port:

```bash
ss -ltnp | grep -E ':8000|:8080|:8081'
```

Cek service:

```bash
systemctl status hate-guard-core hate-guard-mini hate-guard-bot --no-pager
```

Cek log Nginx aaPanel:

```bash
tail -n 80 /www/wwwlogs/mini.gungzy.xyz.error.log
tail -n 80 /www/wwwlogs/bot.gungzy.xyz.error.log
```

Jika port lokal belum muncul, masalahnya ada di service aplikasi. Jika port lokal muncul tetapi Nginx tetap 502, cek lagi blok proxy di config aaPanel.

### Model gagal load

Cek log:

```bash
journalctl -u hate-guard-core -n 120 --no-pager
```

Pastikan `data/model` lengkap:

```bash
ls -lh /www/wwwroot/telegram-ta/data/model
```

Jika VPS tidak punya GPU:

```env
USE_CUDA=0
```

Setelah edit `.env`:

```bash
systemctl restart hate-guard-core
```

### Setelah edit `.env`, setting tidak berubah

Semua config dibaca saat proses start. Restart service sesuai perubahan:

```bash
systemctl restart hate-guard-core hate-guard-mini hate-guard-bot
```

## 21. Checklist akhir

Gunakan checklist ini sebelum menganggap deploy selesai:

- DNS `mini.gungzy.xyz` mengarah ke IP VPS baru.
- DNS `bot.gungzy.xyz` mengarah ke IP VPS baru.
- `https://mini.gungzy.xyz` bisa dibuka.
- `https://bot.gungzy.xyz/webhook` merespon dari browser/curl walaupun bukan 200.
- `curl http://127.0.0.1:8000/healthz` di VPS mengembalikan status sehat.
- `getWebhookInfo` Telegram berisi `https://bot.gungzy.xyz/webhook`.
- Bot membalas `/start`.
- Tombol dashboard membuka `https://mini.gungzy.xyz`.
- Bot sudah admin di grup target.
- `MINI_APP_DEV_MODE=false` di `.env`.
- Firewall hanya membuka SSH, HTTP, dan HTTPS.
