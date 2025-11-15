# Informasi Sistem HateSpeechModerator

## Gambaran Umum
Stack ini dibangun untuk mengawasi grup Telegram dan secara otomatis mendeteksi/mengendalikan pesan bermuatan ujaran kebencian. Tiga komponen utama bekerja bersama:

1. **FastAPI Backend (Python)**  
   - Menyediakan REST API/admin endpoints.  
   - Menampung mini-app React (via `web/main.py`).  
   - Mengelola database SQLite (grup, event moderasi, admin, dll).  
   - Memanggil service inferensi (`api/app.py`) yang menjalankan model NLP ONNX.

2. **Node.js Telegram Bot**  
   - Menggunakan `node-telegram-bot-api` + Express webhook.  
   - Menghapus pesan, memberi peringatan, mute, dan ban secara otomatis.  
   - Berkomunikasi dengan FastAPI lewat HTTP + API key.  
   - Menyediakan tombol WebApp (Mini App) dan perintah admin.

3. **Telegram Mini App (React + Vite)**  
   - UI dashboard real-time: statistik, daftar grup, kontrol threshold/mode, pengelolaan admin/anggota, histori, manual labeling.  
   - Build React disajikan langsung oleh FastAPI (`web/main.py`).

## Teknologi Utama
- Python 3.12, FastAPI, Uvicorn/Gunicorn.  
- ONNXRuntime + Transformers (model IndoBERTweet).  
- Node.js 20, TypeScript, Express, node-telegram-bot-api.  
- React 18, Vite, Chart.js.  
- SQLite (default) atau PostgreSQL via SQLAlchemy.

## Alur Kerja
1. **Inferensi**: Pesan yang dikirim ke bot diserahkan ke endpoint `POST /predict` (ONNX).  
2. **Moderasi**: Bot menerima skor, memutuskan tindakan (peringatan, mute, ban) dan mencatat event ke FastAPI.  
3. **Mini App**: Admin membuka dashboard via WebApp untuk melihat statistik, memantau log, mengubah pengaturan, manual labeling, dll.  
4. **Webhook**: Bot menerima update dari Telegram melalui `WEBHOOK_URL` (HTTPS), sedangkan Mini App dilayani lewat `MINI_APP_BASE_URL`.

## Deployment Singkat
- Jalankan `api/app.py` (core model) via `uvicorn api.app:app --port 8000`.  
- Jalankan `web/main.py` via Gunicorn/Uvicorn (default port 8080) untuk dashboard + proxy.  
- Jalankan bot (`npm run build && pm2 start npm --name hate-guard-bot -- run start`).  
- Reverse proxy domain: `mini.gungzy.xyz → 127.0.0.1:8080`, `bot.gungzy.xyz → 127.0.0.1:8081`.  
- Pastikan `.env` berisi:  
  ```
  MINI_APP_BASE_URL=https://mini.gungzy.xyz
  WEBHOOK_URL=https://bot.gungzy.xyz/webhook
  INFERENCE_API_URL=http://127.0.0.1:8000
  API_KEY=...  # sama antara API & bot
  ```

## Fitur Mini App
- Pilih grup dan ubah mode moderasi (precision/balanced/recall).  
- Slider threshold + retensi log.  
- Metric cards, chart aktivitas, riwayat real-time, filter manual label/deteksi.  
- Manual verification panel untuk melabeli pesan (hate/non-hate).  
- Manajemen admin, mute/ban manual, ekspor CSV.  
- Realtime toggle + manual refresh.
- Riwayat + pagination, menampilkan status verifikasi tiap pesan.

## Perintah Bot dan Fungsinya
- `/start` – menampilkan informasi, tombol Open Dashboard (di DM) + link (di grup).
- `/stats 24h` atau `/stats 7d` – ringkasan tindakan moderasi di interval waktu.
- `/moderation_on` / `/moderation_off` – aktif/nonaktif moderasi otomatis.
- `/set_threshold <nilai>` – ubah ambang pendeteksian manual (0–1).
- `/set_mode <precision|balanced|recall>` – shortcut preset threshold.
- `/mute <user_id> [menit]` – mute manual (default 30 menit).
- `/ban <user_id>` – ban manual.
- `/admins` – daftar admin yang tersimpan di backend.
- `/admin_add <user_id>` / `/admin_remove <user_id>` – utak-atik admin backend.
- `/why <teks>` – tes cepat skor model untuk teks tertentu.
- `/test <teks>` – jalankan inference kontribusi (via endpoint admin/test).
- `/help` (opsional jika diaktifkan) – menampilkan bantuan.

**Automasi internal bot:**
- Menghapus pesan hate → mencatat event (`POST /admin/events`).  
- Memberi warning (maks 4/hari), setelah 5 mute otomatis, setelah 3 mute → ban.  
- Sinkronisasi grup (judul/tipe) saat menerima pesan pertama.
- Memulihkan status mute/ban manual dari backend setiap 15 detik.

## Endpoint API Admin (inti)
- `GET /admin/groups` – daftar grup yang tersinkron.
- `POST /admin/groups/{chat_id}/sync` – update metadata grup.
- `GET|POST /admin/settings/{chat_id}` – membaca & mengubah threshold/mode/retensi.
- `GET /admin/stats/{chat_id}?window=24h|7d` – statistik agregat.
- `GET /admin/events/{chat_id}` – log tindakan (limit + offset).
- `POST /admin/events/{event_id}/verify` – simpan label manual (hate/non-hate).
- `GET /admin/groups/{chat_id}/members?status=muted|banned` – daftar anggota termute/ban.
- `POST /admin/groups/{chat_id}/members` – tambahkan mute/ban manual.
- `DELETE /admin/groups/{chat_id}/members/{user_id}` – lepaskan mute/ban manual.
- `GET /admin/groups/{chat_id}/admins` beserta `POST/DELETE` – kelola admin.
- `GET /admin/activity/{chat_id}` – timeline harian (deleted/warned/blocked).
- `GET /admin/export/{chat_id}` – CSV riwayat.
- `POST /admin/test` – jalankan inference manual (digunakan di `/test` & UI).

Semua endpoint admin memerlukan header `X-API-Key` yang sama dengan konfigurasi bot.

## Catatan Operasional
- Gunakan `guide.txt` untuk daftar command start/stop (nohup, pm2, curl).  
- Atur `MINI_APP_DEV_MODE=true` saat debugging lokal untuk memakai header debug `X-Debug-Chat-Id`.  
- Model ONNX harus lengkap di `data/model/model.onnx`, jika corrupt akan muncul `InvalidProtobuf`.  
- Untuk webhook harus memakai domain HTTPS publik; Telegram tidak menerima HTTP/ngrok di produksi.
