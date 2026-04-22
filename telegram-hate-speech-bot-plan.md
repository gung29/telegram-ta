# 🚀 Telegram Hate Speech Moderation Bot + Mini App Integration Plan

## 🎯 Tujuan
Membangun **bot Telegram** yang menggunakan model ONNX moderasi *hate speech* (IndoBERTweet) untuk:
- Mendeteksi dan menindak ujaran kebencian di grup secara otomatis.
- Memberikan **dashboard mini app (WebApp)** bagi admin grup untuk mengatur threshold, mode, dan melihat statistik.

---

## 🧠 Arsitektur Umum

```
Telegram  ──>  Bot (aiogram)  ──>  Inference API (FastAPI + ONNXRuntime)
   │                 │                    │
   │  WebApp launch  └─>  Mini App (Web)  └─>  DB (SQLite)
   │
Nginx reverse proxy (SSL) di server kamu
```

---

## ⚙️ Komponen Utama

### 1. **Inference API (FastAPI)**
- Endpoint:  
  - `POST /predict` → `{ text } → { prob_hate, prob_nonhate, pred }`
  - `GET /healthz`
- Gunakan `onnxruntime` (CPU) untuk inferensi cepat.
- `SessionOptions`: `intra=1`, `inter=1`, `mem_pattern=False`, `execution_mode=ORT_SEQUENTIAL`
- Tambahkan header `X-API-Key` untuk autentikasi.

---

### 2. **Bot (aiogram)**
- Mode: **Webhook** (hemat resource)  
- Fitur:
  - Pantau semua pesan di grup → inferensi via API
  - Hapus atau mute pesan dengan `prob_hate ≥ threshold`
  - Simpan log (user, pesan, score, waktu)
  - Kirim peringatan otomatis (“⚠️ Hate speech terdeteksi”)
- Perintah admin:
  - `/moderation_on` / `/moderation_off`
  - `/set_threshold 0.65`
  - `/set_mode ketat|moderat|longgar`
  - `/stats [24h|7d]`
  - `/test <teks>`
  - `/whitelist_add` / `/whitelist_remove`

---

### 3. **Telegram Mini App (WebApp)**
- Diluncurkan via tombol **“Open Dashboard”** dari bot.
- Dijalankan di subdomain mis. `mini.gungzy.xyz`
- Fitur:
  - Tampilkan status moderasi grup.
  - Ubah **threshold** dan **mode**.
  - Lihat **statistik pelanggaran** (chart + CSV export).
  - Lakukan **simulasi teks** dengan model langsung.
- Validasi keamanan via `initData` Telegram (HMAC).

---

### 4. **Database (SQLite → PostgreSQL)**
- Tabel:
  - `group_settings(chat_id, enabled, threshold, mode, retention_days)`
  - `events(id, chat_id, user_id, prob_hate, action, ts)`
  - `whitelist(chat_id, pattern)`
- Retensi data otomatis (mis. 7–30 hari).

---

### 5. **Nginx Reverse Proxy**
- Subdomain terpisah:
  - `bot.gungzy.xyz` → FastAPI (Bot + API)
  - `mini.gungzy.xyz` → Mini App (Dashboard)
  - `app.gungzy.xyz` → Streamlit (Frontend model)
- SSL aktif untuk keamanan webhook.

---

## 🔥 Fitur Bot

| Kategori | Fitur | Deskripsi |
|-----------|-------|-----------|
| **Moderasi Otomatis** | Deteksi & hapus pesan | ONNX model mendeteksi ujaran kebencian |
| | Peringatan bertahap | Peringatan → hapus → mute user |
| **Admin Commands** | `/set_threshold`, `/set_mode` | Ubah sensitivitas model |
| | `/stats`, `/test` | Statistik pelanggaran dan uji teks manual |
| **User Interaction** | `/why` | Jelaskan alasan pesan dianggap hate speech |
| **Grup Management** | Whitelist / Bypass | Kecualikan admin atau kata tertentu |
| **Fail-Safe** | Mode pasif saat API down | Pesan tidak dihapus jika API gagal |
| **Logging** | SQLite / Postgres | Simpan event pelanggaran |

---

## 🧩 Arsitektur Detail

```plaintext
[Telegram Messages]
       │
       ▼
 [aiogram Bot]
       │
  ├── Check whitelist/admin
  ├── Call /predict → FastAPI
  ├── Log to DB
  └── Delete/Warn user (if needed)
       │
       ▼
 [Mini App Dashboard]
  ├── API → /admin/settings
  ├── API → /admin/stats
  ├── API → /admin/export
```

---

## 🔒 Keamanan
- Gunakan `API_KEY` di header antara bot ↔ API.
- Validasi `initData` WebApp dengan `HMAC-SHA256`.
- Retensi data terbatas.
- Rate-limit request dari Telegram webhook.
- Whitelist IP Telegram.

---

## 📊 Monitoring
- Endpoint `/metrics` (Prometheus)
- Metrik:
  - QPS inferensi
  - Latency (p95)
  - Jumlah pesan diproses
  - % pesan dihapus
- Logging via `structlog` atau `loguru`.

---

## 🧱 Struktur Folder

```plaintext
hate-guard-bot/
│
├── bot/
│   ├── main.py
│   ├── handlers/
│   ├── utils/
│   └── config.py
│
├── api/
│   ├── app.py
│   ├── model_loader.py
│   └── predict.py
│
├── web/
│   ├── templates/
│   └── static/
│
├── data/
│   ├── model/
│   └── db.sqlite
│
├── .env
└── README.md
```

---

## 🧩 Contoh `.env`

```bash
BOT_TOKEN=123456:ABC...
API_KEY=supersecret
MODEL_PATH=/srv/hate-guard/model
USE_CUDA=0
DEFAULT_THRESHOLD=0.62
ADMIN_IDS=12345678,9999999
```

---

## 🧠 Ide Fitur Lanjutan
- **Adaptive Threshold:** user yang sering melanggar → threshold lebih ketat.
- **Shadow Mode:** logging tanpa tindakan (evaluasi dampak).
- **Multi-language**: tambah deteksi bahasa otomatis.
- **Feedback Admin:** tombol “Mark as Not Hate” untuk re-train data.
- **Broadcast Insight**: laporan mingguan ke admin grup.

---

## ✅ Hasil Akhir yang Diharapkan
- Bot Telegram bisa **diundang ke grup** dan langsung aktif memoderasi.
- Admin bisa buka **dashboard mini app** untuk melihat laporan.
- Semua berjalan di server VPS dengan efisien (CPU-only).
- Bisa dikembangkan lebih lanjut menjadi **platform moderation SaaS ringan**.

---

## 📚 Stack Utama
| Komponen | Teknologi |
|-----------|-----------|
| Inference API | FastAPI + ONNXRuntime |
| Bot Telegram | aiogram v3 |
| Mini App | HTMX / React (opsional) |
| Database | SQLite / PostgreSQL |
| Reverse Proxy | Nginx |
| Deployment | Supervisor / systemd |
| Auth | Telegram WebApp + API Key |
| Monitor | Prometheus + Grafana (opsional) |

---

## 📎 Langkah Lanjut
1. Buat repo `hate-guard-bot`
2. Tambahkan model ONNX kamu (`model/model.onnx`, `tokenizer.json`)
3. Implementasi FastAPI `app.py`
4. Deploy bot + webhook ke `bot.gungzy.xyz`
5. Tambahkan Mini App dasar (dashboard admin)
6. Uji coba di grup Telegram

---

> 💡 *Setelah infrastruktur dasar beres, aku bisa bantu kamu bikin template kode untuk FastAPI inference service + aiogram bot (webhook-ready) langsung.*
