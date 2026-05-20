import pandas as pd
import requests
import time
import random
import sys
import re
import json

FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSccxMu-TNqrC9tWZFnO9nJnU5M0xL1SutqhkhCY4Arx3YB28g/viewform?usp=sharing&ouid=105155610012032553564"

EXCEL_SHEET = "data_simulasi_uji_lokal.xlsx"
SHEET_NAME = "Data_Simulasi"

# Default aman: hanya cek data dan preview payload, tidak submit.
# Ubah ke False hanya untuk form TEST milik sendiri.
DRY_RUN = False

FORM_RESPONSE_URL = FORM_URL.split("?")[0].replace("viewform", "formResponse")


# Kolom Excel -> awalan pertanyaan di Google Form
QUESTION_PREFIX = {
    "Jenis Kelamin": "Jenis Kelamin",
    "Usia": "Usia",
    "Frekuensi Aktif": "Seberapa sering Anda aktif di grup Telegram tersebut?",

    "PU1": "PU1.",
    "PU2": "PU2.",
    "PU3": "PU3.",
    "PU4": "PU4.",
    "PU5": "PU5.",

    "PEOU1": "PEOU1.",
    "PEOU2": "PEOU2.",
    "PEOU3": "PEOU3.",
    "PEOU4": "PEOU4.",
    "PEOU5": "PEOU5.",

    "UA1": "UA1.",
    "UA2": "UA2.",
    "UA3": "UA3.",
}


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def extract_entry_ids(form_url):
    """
    Mengambil entry ID langsung dari HTML Google Form.
    Hasil:
    {
        "Jenis Kelamin": "entry.xxxxx",
        "Usia": "entry.xxxxx",
        ...
    }
    """
    print("[i] Mengambil struktur Google Form...")

    response = requests.get(form_url, timeout=30)
    response.raise_for_status()
    html = response.text

    match = re.search(
        r"var FB_PUBLIC_LOAD_DATA_ = (.*?);</script>",
        html,
        flags=re.DOTALL
    )

    if not match:
        raise RuntimeError(
            "Gagal menemukan FB_PUBLIC_LOAD_DATA_ di HTML form. "
            "Pastikan form bisa diakses publik dan belum dibatasi login."
        )

    raw_json = match.group(1)
    data = json.loads(raw_json)

    question_to_entry = {}

    def walk(node):
        if isinstance(node, list):
            # Struktur umum item Google Forms:
            # [item_id, question_title, ..., ..., [[entry_id, ...]]]
            if len(node) > 4 and isinstance(node[1], str) and isinstance(node[4], list):
                question_title = node[1].strip()

                entry_id = None
                for sub in node[4]:
                    if isinstance(sub, list) and len(sub) > 0 and isinstance(sub[0], int):
                        entry_id = f"entry.{sub[0]}"
                        break

                if question_title and entry_id:
                    question_to_entry[question_title] = entry_id

            for child in node:
                walk(child)

    walk(data)

    entry_ids = {}

    for excel_col, prefix in QUESTION_PREFIX.items():
        prefix_norm = normalize_text(prefix)

        found = None
        for question_title, entry_id in question_to_entry.items():
            q_norm = normalize_text(question_title)

            if q_norm.startswith(prefix_norm) or prefix_norm in q_norm:
                found = entry_id
                break

        if not found:
            print("\n[!] Daftar pertanyaan yang berhasil dibaca:")
            for q, eid in question_to_entry.items():
                print(f"- {q} -> {eid}")

            raise RuntimeError(
                f"Gagal menemukan entry ID untuk kolom Excel '{excel_col}' "
                f"dengan prefix pertanyaan '{prefix}'."
            )

        entry_ids[excel_col] = found

    return entry_ids


def clean_value(column, value):
    """
    Menyesuaikan value Excel agar cocok dengan opsi Google Form.
    """
    value = str(value).strip()

    if column == "Jenis Kelamin":
        mapping = {
            "laki-laki": "Laki - Laki",
            "laki laki": "Laki - Laki",
            "laki - laki": "Laki - Laki",
            "pria": "Laki - Laki",
            "perempuan": "Perempuan",
            "wanita": "Perempuan",
        }
        return mapping.get(normalize_text(value), value)

    if column == "Usia":
        mapping = {
            "<20 tahun": "< 20 tahun",
            "< 20 tahun": "< 20 tahun",
            "20-25 tahun": "20–25 tahun",
            "20 – 25 tahun": "20–25 tahun",
            "20–25 tahun": "20–25 tahun",
            ">25 tahun": "> 25 tahun",
            "> 25 tahun": "> 25 tahun",
        }
        return mapping.get(normalize_text(value), value)

    if column in [
        "PU1", "PU2", "PU3", "PU4", "PU5",
        "PEOU1", "PEOU2", "PEOU3", "PEOU4", "PEOU5",
        "UA1", "UA2", "UA3"
    ]:
        # Google Form memakai nilai 1 sampai 5
        try:
            return str(int(float(value)))
        except ValueError:
            return value

    return value


try:
    df = pd.read_excel(EXCEL_SHEET, sheet_name=SHEET_NAME)
except Exception as e:
    print(f"[!] Gagal membaca Excel: {e}")
    sys.exit(1)


required_columns = list(QUESTION_PREFIX.keys())

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print("[✗] Kolom berikut tidak ditemukan di Excel:")
    for col in missing_columns:
        print(f"- {col}")
    sys.exit(1)


# Cek null/kosong hanya untuk kolom yang dipakai Google Form
for idx, row in df.iterrows():
    for col in required_columns:
        val = row[col]
        if pd.isna(val) or str(val).strip() == "":
            print(f"[✗] Data kosong di baris Excel {idx + 2}, kolom '{col}'.")
            sys.exit(1)


try:
    ENTRY_IDS = extract_entry_ids(FORM_URL)
except Exception as e:
    print(f"[!] Gagal mengambil entry ID: {e}")
    sys.exit(1)


print("\n[✓] Entry ID berhasil ditemukan:")
for col, eid in ENTRY_IDS.items():
    print(f"{col}: {eid}")


print("\n[i] Mulai memproses data...")

for idx, row in df.iterrows():
    form_data = {}

    for col in required_columns:
        form_data[ENTRY_IDS[col]] = clean_value(col, row[col])

    print(f"\n--- Payload responden baris {idx + 1} ---")
    print(form_data)

    if DRY_RUN:
        print("[DRY RUN] Tidak dikirim ke Google Form.")
        continue

    try:
        response = requests.post(FORM_RESPONSE_URL, data=form_data, timeout=30)

        if response.status_code in [200, 302]:
            print(f"[✓] Baris {idx + 1} berhasil dikirim ke form TEST.")
        else:
            print(f"[✗] Gagal submit baris {idx + 1}: HTTP {response.status_code}")
            print(response.text[:500])
            sys.exit(1)

    except Exception as e:
        print(f"[!] Error submit baris {idx + 1}: {e}")
        sys.exit(1)

    delay = random.randint(1, 3)
    print(f"⏳ Menunggu {delay} detik sebelum submit berikutnya...")
    time.sleep(delay)


print("\n[✓] Proses selesai.")