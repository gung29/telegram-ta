# Menjalankan Bot dan Telegram Mini App di Windows

Panduan ini dibuat untuk menjalankan project ini dari komputer Windows lokal setelah VPS/domain `mini.gungzy.xyz` tidak aktif.

Project ini terdiri dari tiga proses:

| Proses | Folder | Port lokal | Fungsi |
| --- | --- | ---: | --- |
| Core API | `api.app:app` | `8000` | Inference model, database, endpoint admin |
| Mini App API | `web.main:app` | `8080` | Proxy API dan static hosting React dashboard |
| Telegram bot | `bot` | polling | Membaca pesan Telegram dan memanggil Core API |

Catatan penting: Telegram Web App wajib memakai URL HTTPS publik. `localhost` atau `http://127.0.0.1` tidak bisa langsung dipakai dari Telegram. Untuk lokal, gunakan tunnel seperti ngrok. Webhook bot juga butuh HTTPS, tetapi untuk setup lokal paling sederhana bot dijalankan dengan polling sehingga webhook tidak diperlukan.

Referensi resmi:

- Telegram `WebAppInfo.url` harus berupa HTTPS URL: https://core.telegram.org/bots/api#webappinfo
- Telegram webhook memakai HTTPS dan `deleteWebhook` diperlukan saat kembali ke polling: https://core.telegram.org/bots/api#setwebhook

## 1. Prasyarat Windows

Install dulu:

- Python 3.11 atau 3.12 64-bit
- Node.js LTS
- Git
- Akun ngrok dan authtoken ngrok

Cek dari PowerShell:

```powershell
py -3 --version
node -v
npm -v
git --version
```

Jika `py` atau `node` belum dikenali, install ulang dengan opsi "Add to PATH", lalu buka PowerShell baru.

Masuk ke folder project:

```powershell
cd "D:\Data cokagung\Skripsi\tele-new"
```

## 2. Siapkan Python virtualenv

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Jika aktivasi venv diblokir PowerShell:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependency Node dan build frontend

`web.main.py` project ini melayani hasil build dari `referensi-desain/dist`, jadi build folder `referensi-desain`, bukan `web/frontend`.

```powershell
npm install --prefix referensi-desain
npm run build --prefix referensi-desain

npm install --prefix bot
npm run build --prefix bot
```

## 4. Edit `.env` untuk mode lokal

Buka `.env`:

```powershell
notepad .env
```

Jangan ubah `BOT_TOKEN`, `API_KEY`, `SECRET_KEY`, dan `ADMIN_IDS` kecuali memang token/admin berubah.

Untuk mode lokal polling, ubah bagian URL menjadi seperti ini:

```env
MODEL_PATH=./data/model
TOKENIZER_PATH=./data/model/tokenizer.json
USE_CUDA=0

INFERENCE_API_URL=http://127.0.0.1:8000

WEBHOOK_URL=
WEBHOOK_HOST=0.0.0.0
WEBHOOK_PORT=8081
BOT_WEBHOOK_PORT=8081
WEBHOOK_PATH=/webhook

DATABASE_URL=sqlite:///./data/db.sqlite3

MINI_APP_BASE_URL=https://ISI-DARI-NGROK-NANTI
ALLOWED_ORIGINS=https://ISI-DARI-NGROK-NANTI,http://localhost:8080,http://127.0.0.1:8080
MINI_APP_DEV_MODE=true
```

Penjelasan:

- `WEBHOOK_URL=` dikosongkan supaya bot memakai polling.
- `MINI_APP_BASE_URL` harus URL HTTPS dari ngrok.
- `MINI_APP_DEV_MODE=true` memudahkan akses dashboard lokal dengan `?chat_id=...`. Jangan pakai nilai ini untuk deploy publik permanen.
- `ALLOWED_ORIGINS` boleh berisi URL ngrok dan localhost. `common/config.py` juga otomatis menambahkan `MINI_APP_BASE_URL` ke CORS.

## 5. Jalankan Core API

Buka PowerShell terminal pertama:

```powershell
cd "D:\Data cokagung\Skripsi\tele-new"
.\.venv\Scripts\Activate.ps1
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Cek health dari PowerShell lain:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

Status yang diharapkan: `status` bernilai `ok`, `model_loaded` bernilai `True`, dan `tokenizer_loaded` bernilai `True`.

## 6. Jalankan Mini App API

Buka PowerShell terminal kedua:

```powershell
cd "D:\Data cokagung\Skripsi\tele-new"
.\.venv\Scripts\Activate.ps1
uvicorn web.main:app --host 127.0.0.1 --port 8080
```

Cek lokal:

```powershell
Start-Process "http://127.0.0.1:8080"
```

Jika muncul halaman dashboard atau halaman login/akses, static hosting sudah jalan. Jika muncul pesan "Mini App belum dibuild", ulangi:

```powershell
npm run build --prefix referensi-desain
```

## 7. Buka HTTPS tunnel untuk Mini App

Buka PowerShell terminal ketiga. Terminal ini harus tetap menyala selama Mini App ingin dipakai dari Telegram.

```powershell
cd "D:\Data cokagung\Skripsi\tele-new"
.\.venv\Scripts\Activate.ps1
python scripts\dev_tunnel.py --mini-only --update-env --auth-token "ISI_NGROK_AUTHTOKEN"
```

Script akan membuat tunnel ke port `8080` dan menulis `MINI_APP_BASE_URL` ke `.env`.

Setelah URL ngrok berubah di `.env`, restart proses Mini App API di terminal kedua supaya config baru terbaca:

```powershell
Ctrl+C
uvicorn web.main:app --host 127.0.0.1 --port 8080
```

Jika ingin mengatur authtoken ngrok permanen:

```powershell
ngrok config add-authtoken ISI_NGROK_AUTHTOKEN
python scripts\dev_tunnel.py --mini-only --update-env
```

## 8. Hapus webhook lama Telegram

Karena `.env` sebelumnya mengarah ke `https://bot.gungzy.xyz/webhook`, hapus webhook lama agar polling bisa berjalan.

Jalankan di PowerShell:

```powershell
$botToken = ((Get-Content .env | Where-Object { $_ -match '^BOT_TOKEN=' }) -replace '^BOT_TOKEN=', '').Trim()
Invoke-RestMethod "https://api.telegram.org/bot$botToken/deleteWebhook?drop_pending_updates=true"
Invoke-RestMethod "https://api.telegram.org/bot$botToken/getWebhookInfo"
```

Pada hasil `getWebhookInfo`, `result.url` seharusnya kosong.

## 9. Jalankan Telegram bot dengan polling

Buka PowerShell terminal keempat:

```powershell
cd "D:\Data cokagung\Skripsi\tele-new"
npm run start --prefix bot
```

Jika ingin mode development dengan reload otomatis:

```powershell
npm run dev --prefix bot
```

Log yang diharapkan:

```text
Polling mode enabled
```

Jika log menunjukkan `Webhook mode enabled`, berarti `WEBHOOK_URL` di `.env` masih terisi. Kosongkan, lalu restart bot.

## 10. Update URL di Telegram

Ada dua jalur URL Mini App:

1. Tombol dari command `/start`
   Bot membaca `MINI_APP_BASE_URL` dari `.env`. Setelah bot direstart, `/start` akan mengirim tombol dashboard dengan URL ngrok terbaru.

2. Menu Button atau Main Mini App dari BotFather
   Jika dulu diset ke `https://mini.gungzy.xyz`, update lewat `@BotFather` ke URL ngrok terbaru. Jalur umumnya:

```text
@BotFather -> pilih bot -> Bot Settings -> Menu Button / Configure Mini App -> ganti URL ke URL ngrok
```

Karena URL ngrok gratis biasanya berubah setiap restart, update BotFather lagi setiap tunnel berubah.

## 11. Cara tes

Tes Core API:

```powershell
$apiKey = ((Get-Content .env | Where-Object { $_ -match '^API_KEY=' }) -replace '^API_KEY=', '').Trim()
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -Headers @{ "X-API-Key" = $apiKey } `
  -ContentType "application/json" `
  -Body '{"text":"tes pesan"}'
```

Tes dari Telegram:

- Kirim `/start` ke bot.
- Pastikan bot membalas tombol `Open Dashboard`.
- Buka tombol dashboard.
- Di grup, pastikan bot sudah menjadi admin jika ingin fungsi hapus/mute/ban bekerja.
- Jalankan `/getid` untuk melihat user ID dan chat ID.
- Jalankan `/test teks percobaan` untuk tes skor moderasi.

Untuk membuka dashboard lokal langsung dari browser saat `MINI_APP_DEV_MODE=true`:

```text
https://URL-NGROK/?chat_id=CHAT_ID_GRUP
```

Chat ID grup Telegram biasanya bernilai negatif.

## 12. Alternatif: tetap memakai webhook lokal

Polling lebih sederhana untuk Windows lokal. Namun jika ingin tetap memakai webhook, jalankan tunnel Mini App dan webhook sekaligus:

```powershell
cd "D:\Data cokagung\Skripsi\tele-new"
.\.venv\Scripts\Activate.ps1
python scripts\dev_tunnel.py --update-env --auth-token "ISI_NGROK_AUTHTOKEN"
```

Script akan mengisi:

```env
MINI_APP_BASE_URL=https://....
WEBHOOK_URL=https://..../webhook
```

Lalu jalankan bot:

```powershell
npm run start --prefix bot
```

Mode ini membutuhkan tunnel webhook ke port `8081`. Jika tunnel mati, bot tidak menerima update sampai webhook diperbaiki atau dihapus kembali dengan `deleteWebhook`.

## 13. Troubleshooting

### `getUpdates` error karena webhook masih aktif

Gejala: bot polling gagal atau tidak menerima pesan.

Solusi:

```powershell
$botToken = ((Get-Content .env | Where-Object { $_ -match '^BOT_TOKEN=' }) -replace '^BOT_TOKEN=', '').Trim()
Invoke-RestMethod "https://api.telegram.org/bot$botToken/deleteWebhook?drop_pending_updates=true"
```

### Tombol dashboard tidak muncul

Penyebab paling umum: `MINI_APP_BASE_URL` bukan HTTPS.

Pastikan nilainya seperti:

```env
MINI_APP_BASE_URL=https://xxxx.ngrok-free.app
```

Lalu restart bot.

### Dashboard terbuka tetapi API error `401 Init data diperlukan`

Untuk local testing, pastikan:

```env
MINI_APP_DEV_MODE=true
```

Restart Mini App API, lalu buka URL dengan query:

```text
https://URL-NGROK/?chat_id=CHAT_ID_GRUP
```

### `Core API unreachable`

Mini App API tidak bisa menghubungi Core API.

Pastikan Core API masih jalan di:

```text
http://127.0.0.1:8000
```

Dan `.env` berisi:

```env
INFERENCE_API_URL=http://127.0.0.1:8000
```

Restart Mini App API setelah mengubah `.env`.

### Port sudah dipakai

Cek proses yang memakai port:

```powershell
netstat -ano | findstr :8000
netstat -ano | findstr :8080
netstat -ano | findstr :8081
```

Matikan proses berdasarkan PID:

```powershell
taskkill /PID PID_NYA /F
```

### Model lambat atau gagal load

Project ini memuat model dari `data/model` melalui `api/model_loader.py`. Pastikan folder ini ada dan berisi file seperti:

```text
config.json
model.safetensors
tokenizer.json
vocab.txt
```

Untuk Windows tanpa GPU, gunakan:

```env
USE_CUDA=0
```

### File `.env` sudah benar tetapi proses masih membaca config lama

Config dibaca saat proses start. Setelah mengubah `.env`, restart proses yang terkait:

- Restart Core API jika mengubah `MODEL_PATH`, `DATABASE_URL`, `API_KEY`, atau `INFERENCE_API_URL`.
- Restart Mini App API jika mengubah `MINI_APP_BASE_URL`, `MINI_APP_DEV_MODE`, `ALLOWED_ORIGINS`, atau `INFERENCE_API_URL`.
- Restart bot jika mengubah `BOT_TOKEN`, `WEBHOOK_URL`, `MINI_APP_BASE_URL`, `API_KEY`, atau `ADMIN_IDS`.

## 14. Ringkasan terminal yang harus tetap menyala

Untuk mode lokal polling, minimal ada empat terminal:

```text
Terminal 1: uvicorn api.app:app --host 127.0.0.1 --port 8000
Terminal 2: uvicorn web.main:app --host 127.0.0.1 --port 8080
Terminal 3: python scripts\dev_tunnel.py --mini-only --update-env
Terminal 4: npm run start --prefix bot
```

Jika semua terminal ini hidup, bot dan Mini App bisa dipakai dari Telegram walaupun VPS sudah tidak aktif.
