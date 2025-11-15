import os
import json
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import textwrap
from typing import Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============== PAGE CONFIG & THEME ==============
st.set_page_config(
    page_title="Hate Speech Moderation",
    page_icon="🛡️",
    layout="wide"
)

# Dark theme CSS + modern look
st.markdown("""
<style>
:root, .stApp {
  --bg: #0e1117;
  --panel: #161a23;
  --text: #e5e7eb;
  --muted: #9aa0a6;
  --accent: #22c55e;
  --accent2: #60a5fa;
}
.stApp { background-color: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background-color: var(--panel); }
.block-container { padding-top: 1.5rem; }
h1,h2,h3 { color: #e5e7eb; }
h3 { margin-top: 1.5rem; margin-bottom: 0.75rem; }
hr { border-top: 1px solid #2b2f3a; }
div.stButton > button {
  background: linear-gradient(90deg, #2563eb, #22c55e);
  border: 0; color: white; padding: 0.55rem 1rem; border-radius: 10px; font-weight: 600;
}
div[data-testid="stTextInput"] input, textarea, .stTextArea textarea, .stSelectbox, .stNumberInput input, .stFileUploader {
  background-color: #0b0e14 !important; color: #e5e7eb !important; border-radius: 10px !important;
}
.stMetric { background: #0b0e14; padding: 0.75rem; border-radius: 12px; border: 1px solid #222633; }
.code-blob {
  background: #0b0e14; border-radius: 12px; padding: 1rem; border: 1px solid #222633; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
.badge { display:inline-block; padding: .25rem .55rem; background:#1f2937; border:1px solid #374151;border-radius:999px; font-size:.8rem; color:#d1d5db; }
.kpill { display:inline-flex; align-items:center; gap:.35rem; padding:.3rem .6rem; border-radius:999px; border:1px solid #334155; background:#0b0e14; }
.kpill .dot { width:8px; height:8px; border-radius:50%; background:#22c55e; }
.small { color: var(--muted); font-size:.85rem; }
/* ===== Fancy loader untuk batch CSV ===== */
.loader-pill {
  display:flex;
  align-items:center;
  gap:0.75rem;
  padding:0.75rem 1rem;
  border-radius:999px;
  border:1px solid #1f2937;
  background:rgba(15,23,42,0.9);
  box-shadow:0 10px 25px rgba(0,0,0,0.45);
  margin:0.5rem 0 1rem 0;
}
.loader-dot {
  width:14px;
  height:14px;
  border-radius:999px;
  background:linear-gradient(135deg,#22c55e,#60a5fa);
  position:relative;
  overflow:hidden;
}
.loader-dot::before {
  content:"";
  position:absolute;
  inset:-4px;
  border-radius:999px;
  border:3px solid rgba(96,165,250,0.15);
  animation:pulse-ring 1.2s infinite ease-out;
}
@keyframes pulse-ring {
  0%   { transform:scale(0.7); opacity:1; }
  100% { transform:scale(1.45); opacity:0; }
}
.loader-text-main {
  font-weight:600;
  font-size:0.9rem;
}
.loader-text-sub {
  font-size:0.8rem;
  color:#9ca3af;
}
</style>
""", unsafe_allow_html=True)

# ============== HELPERS ==============
# Opsional: Streamlit cache kalau jalan di Streamlit; aman di luar Streamlit
try:
    import streamlit as st
    cache_dec = st.cache_resource(show_spinner=False)
except Exception:
    def cache_dec(fn): return fn

@cache_dec
def load_model(
    model_dir: str | None = None,
    onnx_filename: str = "model.onnx",
    use_cuda_env: str | None = None,
    intra_threads: int | None = None,
    inter_threads: int | None = None,
):
    """
    Load tokenizer + ONNX session dengan setting yang ramah container.
    - model_dir default dari env MODEL_PATH (fallback: ./model)
    - otomatis pilih CUDA kalau tersedia & diizinkan
    - threads bisa diatur via argumen atau env
    """
    # ---------- paths ----------
    model_dir = model_dir or os.getenv("MODEL_PATH", "./model")
    onnx_path = os.path.join(model_dir, onnx_filename)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found at: {onnx_path}")

    # ---------- tokenizer ----------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    # ---------- provider selection ----------
    requested_cuda = (use_cuda_env or os.getenv("USE_CUDA", "0")).lower() in {"1","true","yes"}
    avail = ort.get_available_providers()  # e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = ["CPUExecutionProvider"]
    if requested_cuda and "CUDAExecutionProvider" in avail:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # ---------- session options ----------
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # threads (hemat CPU di container)
    import multiprocessing as mp
    cpu = max(1, (mp.cpu_count() or 1))
    so.intra_op_num_threads = int(intra_threads or os.getenv("INTRA_OP_THREADS", 1))
    so.inter_op_num_threads = int(inter_threads or os.getenv("INTER_OP_THREADS", 1))
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # ---------- env runtime tunings ----------
    # kurangi contention library BLAS/OpenMP di container
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ---------- create session ----------
    try:
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    except Exception as e:
        # error yang sering: libcudart tidak ada, model mismatch opset, dll
        raise RuntimeError(f"Failed to create ONNX Runtime session with providers {providers}: {e}")

    device = "CUDA" if "CUDAExecutionProvider" in sess.get_providers() else "CPU"

    # ---------- optional warmup (biar cold-start cepat) ----------
    try:
        # bikin input dummy dari tokenizer (panjang kecil supaya cepat)
        encoded = tok("warmup", return_tensors="np", padding="max_length", truncation=True, max_length=16)
        feed = {i.name: encoded.get(i.name) for i in sess.get_inputs() if i.name in encoded}
        if feed:
            sess.run(None, feed)
    except Exception:
        # warmup gagal bukan masalah fatal; lanjut saja
        pass

    return tok, sess, device

def count_parameters(model, onnx_path: Optional[str] = None) -> Optional[int]:
    """
    - Jika model PyTorch: hitung via .parameters()
    - Jika ONNX: hitung dari jumlah elemen di semua initializer (butuh onnx_path)
    - Jika tidak diketahui: kembalikan None
    """
    # PyTorch?
    if hasattr(model, "parameters"):
        try:
            return sum(p.numel() for p in model.parameters())
        except Exception:
            pass

    # ONNX?
    if onnx_path and os.path.exists(onnx_path):
        try:
            m = onnx.load(onnx_path)
            total = 0
            for init in m.graph.initializer:
                size = 1
                for d in init.dims:
                    size *= d
                total += size
            return int(total)
        except Exception:
            return None

    return None

def human_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")

def safe_read_txt(path: str) -> Optional[str]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None
    return None

def load_metadata(model_dir: str) -> Dict:
    meta = {}
    mpath = os.path.join(model_dir, "metadata.json")
    if os.path.exists(mpath):
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
    # also read config.json for id2label/label2id if not in metadata
    cfgp = os.path.join(model_dir, "config.json")
    if os.path.exists(cfgp):
        try:
            with open(cfgp, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                meta.setdefault("id2label", cfg.get("id2label"))
                meta.setdefault("label2id", cfg.get("label2id"))
                meta.setdefault("model_name", cfg.get("_name_or_path", None))
                meta.setdefault("num_labels", cfg.get("num_labels", None))
        except Exception:
            pass
    return meta

def preprocess_IndoBERTweet(text: str) -> str:
    # ringan & konsisten dgn training
    import re, emoji
    text = str(text).lower()
    text = re.sub(r'@\w+', '@USER', text)
    text = re.sub(r'http\S+|www\S+', 'HTTPURL', text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_text(text: str, tokenizer, session: ort.InferenceSession, device_str: str, max_length: int = 128) -> Tuple[np.ndarray, int]:
    # preprocessing sama
    txt = preprocess_IndoBERTweet(text)
    enc = tokenizer(txt, return_tensors="np", truncation=True, padding=True, max_length=max_length)

    # ONNX IndoBERT-like biasanya minta int64; cast aman
    for k in ("input_ids", "attention_mask", "token_type_ids"):
        if k in enc:
            enc[k] = enc[k].astype("int64")

    # Sesuaikan nama input yg tersedia di session
    # (kebanyakan: input_ids, attention_mask, (opsional) token_type_ids)
    input_names = {i.name for i in session.get_inputs()}
    feeds = {k: v for k, v in enc.items() if k in input_names}

    logits = session.run(None, feeds)[0]   # (1, num_labels)
    # softmax manual (stabil)
    x = logits - logits.max(axis=1, keepdims=True)
    probs = (np.exp(x) / np.exp(x).sum(axis=1, keepdims=True))[0]
    pred = int(np.argmax(probs))
    return probs, pred

@st.cache_data(show_spinner=False)
def load_pr_table(model_root: str) -> Optional[pd.DataFrame]:
    # file bisa di root artifacts atau root/
    candidates = [
        os.path.join(model_root, "pr_curve_table_hate.csv"),
        os.path.join(os.path.dirname(model_root), "pr_curve_table_hate.csv")
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                # ensure columns exist
                if {"threshold","precision_hate","recall_hate","f1_hate"}.issubset(df.columns):
                    return df
            except Exception:
                pass
    return None

def pick_threshold(model_root: str, mode: str, fallback: float = 0.5) -> float:
    mapping = {
        "PRECISION-FIRST (strict)": "best_threshold_prec.txt",
        "BALANCED (F1-max)": "best_threshold_f1.txt",
        "RECALL-FIRST (catch-all)": "best_threshold_recall.txt",
    }
    if mode in mapping:
        # file bisa di root artifacts atau root/
        rel = mapping[mode]
        for p in [os.path.join(model_root, rel), os.path.join(os.path.dirname(model_root), rel)]:
            v = safe_read_txt(p)
            if v:
                try:
                    return float(v)
                except Exception:
                    continue
    return fallback

def make_pr_plot(df: pd.DataFrame):
    fig = plt.figure(figsize=(6,5))
    plt.plot(df["recall_hate"], df["precision_hate"], label="PR curve")
    # tandai kandidat terbaik di tabel (opsional)
    try:
        idx = int(df["f1_hate"].values.argmax())
        plt.scatter(df.loc[idx,"recall_hate"], df.loc[idx,"precision_hate"], s=40, label=f"F1 max @ {df.loc[idx,'threshold']:.2f}")
    except Exception:
        pass
    plt.xlabel("Recall (Hate)")
    plt.ylabel("Precision (Hate)")
    plt.title("Precision–Recall Curve — Hate (label=1)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    return fig

def render_csv_settings(default_thr: float) -> dict:
    """Tampilkan panel kontrol batch CSV dengan tata letak modern dan responsif."""
    st.markdown("##### ⚙️ Pengaturan batch (sebelum upload)")
    st.caption("Semua opsi di bawah ini diterapkan sebelum file CSV diproses.")

    st.markdown("""
    <style>
    .batch-hint {
        color:#9ca3af;
        font-size:0.85rem;
        margin-bottom:0.5rem;
    }
    [data-testid="stTabs"] button {
        font-weight:600;
    }
    [data-testid="stFileUploaderDropzone"] {
        border-radius:16px !important;
        border:1px solid #1f2937 !important;
        background:rgba(15,23,42,0.35) !important;
        backdrop-filter:blur(6px);
    }
    </style>
    """, unsafe_allow_html=True)

    tab_basic, tab_adv = st.tabs(["Dasar", "Lanjutan"])

    with tab_basic:
        base1, base2, base3 = st.columns(3)
        with base1:
            do_preprocess = st.checkbox(
                "Preprocess seperti training",
                value=st.session_state.get("csv_preprocess", True),
                help="lowercase, ganti @USER, HTTPURL, demojize, dll.",
                key="csv_preprocess",
            )
        with base2:
            batch_size = st.slider(
                "Batch size",
                min_value=8,
                max_value=512,
                value=int(st.session_state.get("csv_batch_size", 64)),
                step=8,
                help="Kontrol jumlah teks yang diproses tiap batch (untuk kompatibilitas saja saat ini).",
                key="csv_batch_size",
            )
        with base3:
            active_thr = st.slider(
                "Threshold aktif (Hate)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("csv_thr_live", default_thr)),
                step=0.01,
                help="Gunakan nilai ini untuk seluruh keputusan label berbasis threshold.",
                key="csv_thr_live",
            )
        base4, base5 = st.columns(2)
        with base4:
            preview_rows = st.slider(
                "Preview awal",
                min_value=3,
                max_value=20,
                value=int(st.session_state.get("csv_preview_rows", 3)),
                step=1,
                help="Jumlah baris yang ditampilkan sebelum inferensi dimulai.",
                key="csv_preview_rows",
            )
        with base5:
            st.metric(
                "Threshold aktif",
                f"{active_thr:.3f}",
                delta=f"{active_thr - default_thr:+.3f}",
            )

    with tab_adv:
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            limit_rows = st.number_input(
                "Limit baris diproses (0 = semua)",
                min_value=0,
                max_value=300000,
                value=int(st.session_state.get("csv_limit_rows", 0)),
                step=1000,
                help="Gunakan untuk eksperimen cepat pada subset data.",
                key="csv_limit_rows",
            )
        with adv2:
            shuffle_rows = st.checkbox(
                "Acak urutan sebelum proses",
                value=st.session_state.get("csv_shuffle_rows", False),
                help="Bagus untuk sampling agar hasil lebih representatif.",
                key="csv_shuffle_rows",
            )
        with adv3:
            loader_interval = st.slider(
                "Update animasi tiap N teks",
                min_value=1,
                max_value=20,
                value=int(st.session_state.get("csv_loader_interval", 1)),
                step=1,
                help="Interval refresh kartu loader kustom.",
                key="csv_loader_interval",
            )
        adv4, adv5 = st.columns(2)
        with adv4:
            shuffle_seed = st.number_input(
                "Seed shuffle",
                min_value=0,
                max_value=9999,
                value=int(st.session_state.get("csv_shuffle_seed", 42)),
                step=1,
                help="Gunakan seed tetap agar subset selalu sama.",
                key="csv_shuffle_seed",
                disabled=not shuffle_rows,
            )
        with adv5:
            sample_options = ["Semua baris", "Sample 10%", "Sample 1%"]
            default_sample = st.session_state.get("csv_sample_mode", sample_options[0])
            default_index = sample_options.index(default_sample) if default_sample in sample_options else 0
            sample_mode = st.selectbox(
                "Mode sampling cepat",
                sample_options,
                index=default_index,
                help="Pilih subset otomatis sebelum limit diterapkan.",
                key="csv_sample_mode",
            )

    return {
        "do_preprocess": do_preprocess,
        "batch_size": batch_size,
        "active_thr": active_thr,
        "preview_rows": preview_rows,
        "limit_rows": limit_rows,
        "shuffle_rows": shuffle_rows,
        "shuffle_seed": shuffle_seed if shuffle_rows else None,
        "loader_interval": loader_interval,
        "sample_mode": sample_mode,
    }

# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    st.markdown("Pilih **folder model** (yang berisi `model.safetensors`, `config.json`, `tokenizer_config.json`, dll).")
    default_dir = os.path.abspath("model")  # <-- pakai subfolder 'model' relatif ke app.py
    model_subdir = st.text_input("Subfolder model (opsional)", value="")
    model_dir = os.path.join(default_dir, model_subdir) if model_subdir else default_dir
    onnx_name = st.text_input("Nama file ONNX", value="model.onnx")

    threshold_mode = st.selectbox(
        "Threshold mode",
        ["PRECISION-FIRST (strict)", "BALANCED (F1-max)", "RECALL-FIRST (catch-all)", "CUSTOM"],
        index=0
    )
    custom_thr = st.slider("Custom threshold (Hate)", 0.0, 1.0, 0.5, 0.01, disabled=(threshold_mode!="CUSTOM"))
    max_len = st.slider("Max length (tokens)", 64, 256, 128, 8)
    st.caption("Tip: jika banyak teks terpotong, naikkan `max_length` → sedikit lebih lambat.")
    reload_btn = st.button("🔄 Load / Reload Model")

# ============== LOAD MODEL ==============
load_now = True
if reload_btn:
    st.cache_resource.clear()  # force reload
    st.cache_data.clear()

try:
    tokenizer, model, device = load_model(model_dir, onnx_name)
except Exception as e:
    st.error(f"Gagal memuat model dari: `{model_dir}`\n\n{e}")
    st.stop()

num_params = count_parameters(model, os.path.join(model_dir, onnx_name))
# threshold pick
thr = custom_thr if threshold_mode=="CUSTOM" else pick_threshold(model_dir, threshold_mode, fallback=0.5)

# metadata
meta = load_metadata(model_dir)
# ===== Normalisasi label maps & helper aman =====
def _normalize_label_maps(id2label_in, label2id_in):
    id2label_out = {}
    if id2label_in:
        for k, v in id2label_in.items():
            try:
                kk = int(k) if isinstance(k, str) and k.isdigit() else k
            except Exception:
                kk = k
            id2label_out[kk] = v
    if not id2label_out:
        id2label_out = {0: "Non Hate Speech", 1: "Hate Speech"}

    label2id_out = {}
    if label2id_in:
        for k, v in label2id_in.items():
            try:
                vv = int(v) if isinstance(v, str) and str(v).isdigit() else v
            except Exception:
                vv = v
            label2id_out[k] = vv

    id2label_out.setdefault(0, "Non Hate Speech")
    id2label_out.setdefault(1, "Hate Speech")
    return id2label_out, label2id_out

id2label, label2id = _normalize_label_maps(
    meta.get("id2label") or {0:"Non Hate Speech", 1:"Hate Speech"},
    meta.get("label2id") or {"Non Hate Speech":0, "Hate Speech":1},
)

import sqlite3
from datetime import datetime

DB_PATH = os.path.join(model_dir, "history.db")  # simpan per-model

def init_db():
    os.makedirs(model_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            text TEXT,
            prob_hate REAL,
            prob_nonhate REAL,
            pred_argmax TEXT,
            pred_threshold TEXT,
            threshold REAL
        )
    """)
    conn.commit(); conn.close()

def save_history_sqlite(record: dict):
    """Simpan satu record prediksi ke DB."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""INSERT INTO history
        (ts, text, prob_hate, prob_nonhate, pred_argmax, pred_threshold, threshold)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (record.get("ts"), record.get("text"), record.get("prob_hate"),
         record.get("prob_nonhate"), record.get("pred_argmax"),
         record.get("pred_threshold"), record.get("threshold"))
    )
    conn.commit(); conn.close()

def query_history(keyword:str="", label:str="ALL", prob_min:float=0.0, prob_max:float=1.0,
                  since:float=None, until:float=None, limit:int=500) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    q = "SELECT ts, text, prob_hate, prob_nonhate, pred_argmax, pred_threshold, threshold FROM history WHERE 1=1"
    params = []
    if keyword:
        q += " AND text LIKE ?"
        params.append(f"%{keyword}%")
    if label in ("Hate","Non Hate","Non Hate Speech"):
        q += " AND pred_threshold = ?"
        params.append(label)
    q += " AND prob_hate BETWEEN ? AND ?"
    params.extend([prob_min, prob_max])
    if since is not None:
        q += " AND ts >= ?"; params.append(since)
    if until is not None:
        q += " AND ts <= ?"; params.append(until)
    q += " ORDER BY ts DESC LIMIT ?"; params.append(int(limit))
    c.execute(q, params)
    rows = c.fetchall()
    conn.close()
    cols = ["ts","full_text","prob_hate","prob_nonhate","pred_argmax","pred_threshold","threshold"]
    return pd.DataFrame(rows, columns=cols)

def clear_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit(); conn.close()

# inisialisasi sekali saat app start
init_db()

def lbl(i):
    """Ambil nama label aman untuk id 0/1."""
    return id2label.get(i) or id2label.get(str(i)) or str(i)
num_params = count_parameters(model)
best_ckpt = meta.get("best_checkpoint", None)
model_name = meta.get("model_name", "indolem/indobertweet-base-uncased")

# ============== HEADER ==============
st.markdown("### 🛡️ Hate Speech Moderation — IndoBERTweet")
st.markdown(
    f"""
<div class="kpill"><span class="dot"></span> <b>Model</b>: {model_name} &nbsp;•&nbsp; 
<b>Device</b>: {device.upper()} &nbsp;•&nbsp; <b>Params</b>: {human_int(num_params) if num_params else "—"}</div>
""", unsafe_allow_html=True)

colA, colB, colC = st.columns([1,1,1])
with colA:
    st.metric("Threshold aktif (Hate)", f"{thr:.3f}", help="Jika prob(Hate) ≥ threshold → label Hate.")
with colB:
    st.metric("Label mapping", f"0={id2label.get(0,'0')}, 1={id2label.get(1,'1')}")
with colC:
    st.metric("Best checkpoint", best_ckpt or "—")

st.markdown("---")

# ============== SINGLE TEXT INFERENCE ==============
st.markdown("#### 🔎 Uji Cepat Satu Teks")
sample = st.text_area("Masukkan teks", height=120, placeholder="Ketik kalimat di sini…")
go1 = st.button("🚀 Prediksi")

if go1 and sample.strip():
    t0 = time.time()
    probs, pred = predict_text(sample, tokenizer, model, device, max_length=max_len)
    dt = (time.time() - t0) * 1000
    prob_nonhate, prob_hate = float(probs[0]), float(probs[1])
    label_final = 1 if prob_hate >= thr else 0
    st.markdown("**Hasil:**")
    c1, c2, c3 = st.columns([1,1,1])
    with c1: st.metric("Pred (argmax)", id2label.get(pred, str(pred)))
    with c2: st.metric("Label (by threshold)", id2label.get(label_final, str(label_final)))
    with c3: st.metric("Latency", f"{dt:.1f} ms")

    st.markdown("**Probabilitas:**")
    st.progress(min(1.0, prob_hate))
    st.write(f"Non Hate: **{prob_nonhate:.3f}**, Hate: **{prob_hate:.3f}**  (thr={thr:.3f})")

    st.markdown("**Teks setelah preprocessing (IndoBERTweet):**")
    st.code(preprocess_IndoBERTweet(sample), language="text")

    # simpan riwayat di session_state
    hist = st.session_state.get("history", [])
    hist.insert(0, {
        "text": sample, "prob_hate": prob_hate, "prob_nonhate": prob_nonhate,
        "pred_argmax": id2label.get(pred, str(pred)),
        "pred_threshold": id2label.get(label_final, str(label_final)),
        "threshold": thr, "ts": time.time()
    })
    st.session_state["history"] = hist[:50]
    
    # juga simpan ke SQLite (persistent)
    try:
        save_history_sqlite(hist[0])  # simpan entri terbaru
    except Exception as e:
        st.warning(f"Gagal simpan history ke DB: {e}")

    
    # 1) Confidence & Uncertainty (entropy)
    eps = 1e-12
    entropy = - (prob_nonhate * math.log(prob_nonhate + eps, 2) + prob_hate * math.log(prob_hate + eps, 2))
    # normalisasi ke [0,1] untuk 2 kelas (max entropy = 1 bit)
    confidence = 1.0 - min(1.0, entropy/1.0)
    margin_to_thr = prob_hate - thr   # + -> di atas threshold, - -> di bawah

    c4, c5, c6 = st.columns([1,1,1])
    with c4:
        st.metric("Confidence", f"{confidence:.2f}")
    with c5:
        st.metric("Uncertainty (entropy)", f"{entropy:.2f}")
    with c6:
        st.metric("Margin vs thr", f"{margin_to_thr:+.3f}",
                  help="Positif berarti berada di atas threshold (Hate).")

    # 2) Probability bar (HATE vs NON-HATE) + garis threshold (untuk kelas Hate)
    fig_bar, ax = plt.subplots(figsize=(6, 1.4))
    ax.barh(["Non Hate", "Hate"], [prob_nonhate, prob_hate])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    # Garis threshold untuk kelas Hate
    ax.axvline(thr, color="black", linestyle="--", linewidth=1)
    ax.text(thr, 0.5, f" thr={thr:.2f}", va="center", ha="left")
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)
    st.pyplot(fig_bar, use_container_width=True)

    # 3) Highlight istilah berisiko (heuristik ringan) pada teks hasil preprocessing
    RISKY = {
        # tambahkan daftar sesuai kebutuhanmu
        "bodoh","goblok","anjing","bangsat","hina","tolol","brengsek","hina","babi","kafir"
    }
    def highlight_risky(text):
        toks = text.split()
        out = []
        for t in toks:
            if t.lower() in RISKY:
                out.append(f"<span style='background:#1d4ed8; color:#fff; padding:0 .25rem; border-radius:.25rem'>{t}</span>")
            else:
                out.append(t)
        return " ".join(out)

    st.markdown("**Highlight istilah berisiko (heuristik):**", unsafe_allow_html=True)
    st.markdown(highlight_risky(preprocess_IndoBERTweet(sample)), unsafe_allow_html=True)

    # 4) (Opsional) titik “now @ threshold” pada PR-curve jika file ada
    try:
        pr_df_now = load_pr_table(model_dir)
        if pr_df_now is not None and {"threshold","precision_hate","recall_hate"}.issubset(pr_df_now.columns):
            # cari baris terdekat dengan threshold aktif
            idx_near = (pr_df_now["threshold"] - thr).abs().idxmin()
            r_now = float(pr_df_now.loc[idx_near, "recall_hate"])
            p_now = float(pr_df_now.loc[idx_near, "precision_hate"])

            fig_now = plt.figure(figsize=(6, 4.5))
            plt.plot(pr_df_now["recall_hate"], pr_df_now["precision_hate"], label="PR curve")
            plt.scatter([r_now], [p_now], s=50, label=f"now @ {thr:.2f}")
            plt.axvline(r_now, linestyle="--", alpha=0.4)
            plt.xlabel("Recall (Hate)"); plt.ylabel("Precision (Hate)")
            plt.title("Posisi Threshold Saat Ini pada PR Curve")
            plt.grid(True, linestyle="--", alpha=0.3); plt.legend()
            st.pyplot(fig_now, use_container_width=True)

            # tampilkan estimasi metrik di threshold aktif
            st.caption(f"≈ Precision: **{p_now:.3f}**, Recall: **{r_now:.3f}** (interpolasi dari PR table)")
    except Exception:
        pass

# ============== BATCH CSV INFERENCE ==============
st.markdown("---")
st.markdown("#### 📄 Batch Inferensi dari CSV")
st.caption("CSV minimal punya kolom **full_text**. (Opsional) kolom **label** untuk evaluasi.")

csv_settings = render_csv_settings(thr)
do_preprocess = csv_settings["do_preprocess"]
batch_size = csv_settings["batch_size"]
preview_rows = csv_settings["preview_rows"]
limit_rows = csv_settings["limit_rows"]
shuffle_rows = csv_settings["shuffle_rows"]
shuffle_seed = csv_settings["shuffle_seed"]
loader_interval = max(1, csv_settings["loader_interval"])
sample_mode = csv_settings["sample_mode"]
active_thr = csv_settings["active_thr"]
thr_live = active_thr  # simpan agar seksi lain mengikuti nilai terbaru

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    # ---------- LOAD CSV ----------
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        st.stop()

    if sample_mode != "Semua baris":
        frac_map = {"Sample 10%": 0.10, "Sample 1%": 0.01}
        frac = frac_map.get(sample_mode, 1.0)
        df = df.sample(frac=min(1.0, frac), random_state=shuffle_seed).reset_index(drop=True)
        st.info(f"Mode {sample_mode}: memproses sekitar {len(df):,} baris.", icon="⚡")

    if shuffle_rows:
        df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    if limit_rows > 0:
        df = df.head(limit_rows)

    if "full_text" not in df.columns:
        st.error("Kolom `full_text` tidak ditemukan.")
        st.stop()

    st.write(f"**Preview ({preview_rows} baris):**")
    st.dataframe(df.head(preview_rows))

    # ---------- INFERENCE ----------
    texts = df["full_text"].astype(str).tolist()
    n = len(texts)
    preds, phates, pnon = [], [], []

    progress = st.progress(0.0)   # progress bar bawaan
    loader_placeholder = st.empty()  # slot untuk loader animasi kustom

    import time

    def format_eta(seconds: float | int) -> str:
        """Format ETA: X jam Y menit Z detik."""
        if seconds is None or seconds <= 0:
            return "ETA ~ 0 detik"

        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        parts = []
        if h > 0:
            parts.append(f"{h} jam")
        if m > 0:
            parts.append(f"{m} menit")
        parts.append(f"{s} detik")

        return "ETA ~ " + " ".join(parts)

    def render_loader(done: int, total: int, eta_seconds: float | int):
        """Render kartu loading animasi besar via placeholder HTML."""
        if total <= 0:
            pct_int = 0
        else:
            raw_pct = done / total * 100.0
            if done > 0 and raw_pct < 1.0:
                pct_int = 1
            else:
                pct_int = int(raw_pct)
            pct_int = max(0, min(100, pct_int))

        eta_str = format_eta(eta_seconds)

        html = textwrap.dedent(f"""
        <div style="margin-top:1.5rem; display:flex; justify-content:center;">
          <div style="
              background: radial-gradient(circle at top left,#1d4ed8,#020617);
              border-radius: 18px;
              padding: 1.1rem 1.6rem;
              border: 1px solid rgba(148,163,184,0.6);
              box-shadow: 0 22px 50px rgba(15,23,42,0.9);
              max-width: 720px;
              width: 100%;
              color: #e5e7eb;
              font-family: system-ui,-apple-system,BlinkMacSystemFont,'SF Pro Text',sans-serif;">
            <div style="display:flex; gap:1.2rem; align-items:center;">
              <div style="
                  width:60px; height:60px;
                  border-radius:999px;
                  border:5px solid rgba(59,130,246,.35);
                  border-top-color:#22c55e;
                  animation: spin 0.9s linear infinite;
              "></div>
              <div style="flex:1;">
                <div style="font-size:1.25rem; font-weight:650; letter-spacing:.02em;">
                  Memproses batch CSV…
                </div>
                <div style="margin-top:0.3rem; font-size:0.98rem; color:#cbd5f5;">
                  <span style="font-variant-numeric:tabular-nums;">
                    {done:,} / {total:,} teks
                  </span>
                  <span style="opacity:.65;"> • </span>
                  <span style="font-variant-numeric:tabular-nums;">{eta_str}</span>
                </div>
                <div style="margin-top:0.9rem; width:100%; height:11px;
                            border-radius:999px; background:rgba(15,23,42,.9); overflow:hidden;">
                  <div style="
                      width:{pct_int}%;
                      height:100%;
                      background:linear-gradient(90deg,#22c55e,#38bdf8,#a855f7);
                      box-shadow:0 0 18px rgba(34,197,94,.9);
                      transition:width .3s ease-out;
                  "></div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <style>
        @keyframes spin {{
          from {{ transform: rotate(0deg); }}
          to   {{ transform: rotate(360deg); }}
        }}
        </style>
        """).strip()
        # Render HTML kustom ke placeholder agar bisa diperbarui tiap panggilan
        loader_placeholder.markdown(html, unsafe_allow_html=True)

    def _prep(x: str) -> str:
        return preprocess_IndoBERTweet(x) if do_preprocess else x

    t_start = time.time()
    render_loader(0, n, 0)  # state awal

    # Loop per TEKS → update realtime
    for idx, text in enumerate(texts, start=1):
        probs, _ = predict_text(_prep(text), tokenizer, model, device, max_length=max_len)
        phates.append(float(probs[1]))
        pnon.append(float(probs[0]))
        preds.append(1 if probs[1] >= active_thr else 0)

        done = idx
        pct = done / n if n > 0 else 0.0

        elapsed = time.time() - t_start
        eta = (elapsed / done) * (n - done) if (done > 0 and done < n) else 0

        progress.progress(pct)
        # update loader tiap beberapa teks biar gak terlalu berat
        if idx == 1 or idx == n or idx % loader_interval == 0:
            render_loader(done, n, eta)

    # setelah selesai, hilangkan loader kustom
    loader_placeholder.empty()
    progress.empty()

    # ---------- MERGE OUTPUT ----------
    out = df.copy()
    out["prob_nonhate"] = pnon
    out["prob_hate"] = phates
    out["pred_id"] = preds                    # 0/1
    out["pred_threshold"] = [lbl(x) for x in preds]  # nama label aman

    # ---------- SUMMARY ----------
    st.markdown("### 📊 Ringkasan")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total baris", f"{len(out):,}")
    with col2:
        pct_pred_hate = (np.array(preds) == 1).mean()*100.0
        st.metric("% Pred Hate", f"{pct_pred_hate:.1f}%")
    with col3:
        st.metric("Mean prob Hate", f"{out['prob_hate'].mean():.3f}")
    with col4:
        st.metric("Median prob Hate", f"{out['prob_hate'].median():.3f}")

    # ---------- DISTRIBUTION CHART ----------
    try:
        import altair as alt
        hist = alt.Chart(out).mark_bar().encode(
            x=alt.X("prob_hate:Q", bin=alt.Bin(maxbins=30), title="Prob(Hate)"),
            y=alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip("count()", title="n")]
        ).properties(height=180)
        st.altair_chart(hist, use_container_width=True)
    except Exception:
        fig_hist, axh = plt.subplots(figsize=(6,2))
        axh.hist(out["prob_hate"], bins=30)
        axh.set_xlabel("Prob(Hate)"); axh.set_ylabel("Count")
        st.pyplot(fig_hist, use_container_width=True)

    # ---------- INTERACTIVE FILTER ----------
    st.markdown("### 🔍 Filter Interaktif")
    fcol1, fcol2, fcol3 = st.columns([1,1,2])
    with fcol1:
        show_only = st.selectbox("Tampilkan", ["Semua", "Hanya pred Hate", "Hanya pred Non Hate"])
    with fcol2:
        prob_range = st.slider("Rentang Prob(Hate)", 0.0, 1.0, (0.0, 1.0), 0.01)
    with fcol3:
        keyword = st.text_input("Cari keyword (opsional)", value="")

    mask = (out["prob_hate"] >= prob_range[0]) & (out["prob_hate"] <= prob_range[1])
    if show_only == "Hanya pred Hate":
        mask &= (out["pred_id"] == 1)
    elif show_only == "Hanya pred Non Hate":
        mask &= (out["pred_id"] == 0)
    if keyword:
        mask &= out["full_text"].str.contains(keyword, case=False, na=False)

    filtered = out[mask].reset_index(drop=True)
    st.caption(f"Hasil tersaring: **{len(filtered):,}** baris")
    st.dataframe(filtered.head(50), use_container_width=True)

    # ---------- INTERESTING SAMPLES ----------
    st.markdown("### 🧪 Sample Menarik")
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        k = st.number_input("Top-N", 5, 100, 10, 5)
    with scol2:
        top_conf_hate = filtered.sort_values("prob_hate", ascending=False).head(k)
        st.write("**Paling yakin Hate**")
        st.dataframe(top_conf_hate[["full_text","prob_hate"]])
    with scol3:
        top_conf_non = filtered.sort_values("prob_nonhate", ascending=False).head(k)
        st.write("**Paling yakin Non Hate**")
        st.dataframe(top_conf_non[["full_text","prob_nonhate"]])

    filtered["_margin_vs_thr"] = (filtered["prob_hate"] - active_thr).abs()
    st.write("**Paling Ragu (dekat threshold)**")
    st.dataframe(filtered.sort_values("_margin_vs_thr", ascending=True).head(k)[["full_text","prob_hate","_margin_vs_thr"]])

    # ---------- AUTO EVAL (JIKA ADA LABEL) ----------
    if "label" in out.columns:
        from sklearn.metrics import classification_report, confusion_matrix
        labmap = {"Non Hate Speech":0, "Hate Speech":1, 0:0, 1:1}
        y_true = out["label"].map(labmap).fillna(0).astype(int).values
        y_pred = (out["prob_hate"] >= active_thr).astype(int).values

        st.markdown("### ✅ Evaluasi (berdasarkan label di CSV)")
        st.text(classification_report(y_true, y_pred, target_names=[lbl(0), lbl(1)], digits=4))

        cm = confusion_matrix(y_true, y_pred)
        fig_cm, axcm = plt.subplots(figsize=(3.8,3.5))
        im = axcm.imshow(cm, cmap="Blues")
        axcm.set_xticks([0,1]); axcm.set_yticks([0,1])
        axcm.set_xticklabels([lbl(0), lbl(1)])
        axcm.set_yticklabels([lbl(0), lbl(1)])
        for i in range(2):
            for j in range(2):
                axcm.text(j, i, cm[i,j], ha="center", va="center")
        axcm.set_title("Confusion Matrix"); plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)

        # Reliability (calibration) kecil
        try:
            from sklearn.calibration import calibration_curve
            frac_pos, mean_pred = calibration_curve(y_true, out["prob_hate"].values, n_bins=10, strategy="quantile")
            fig_cal, axcal = plt.subplots(figsize=(3.8,3.5))
            axcal.plot([0,1],[0,1],"--",alpha=0.5)
            axcal.plot(mean_pred, frac_pos, marker="o")
            axcal.set_title("Reliability Curve (Hate)")
            axcal.set_xlabel("Mean predicted prob"); axcal.set_ylabel("Fraction of positives")
            axcal.grid(True, linestyle="--", alpha=0.3)
            st.pyplot(fig_cal, use_container_width=True)
        except Exception:
            pass

    # ---------- DOWNLOADERS ----------
    st.markdown("### 💾 Unduh")
    coldl1, coldl2, coldl3 = st.columns(3)
    with coldl1:
        st.download_button(
            "⬇️ Full Results (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="inference_results_full.csv",
            mime="text/csv"
        )
    with coldl2:
        st.download_button(
            "⬇️ Filtered View (CSV)",
            data=filtered.drop(columns=["_margin_vs_thr"], errors="ignore").to_csv(index=False).encode("utf-8"),
            file_name="inference_results_filtered.csv",
            mime="text/csv"
        )
    with coldl3:
        subset_cols = ["full_text","prob_hate","pred_threshold"]
        st.download_button(
            "⬇️ Ringkas (CSV)",
            data=out[subset_cols].to_csv(index=False).encode("utf-8"),
            file_name="inference_results_compact.csv",
            mime="text/csv"
        )

# ============== ANALYTICS • ARTEFAK • HISTORY (RESPONSIVE) ==============
st.markdown("---")
tab_pr, tab_art, tab_hist = st.tabs(["📈 PR Curve", "📦 Artefak", "🗂️ History"])

# ------------- TAB: PR CURVE -------------
with tab_pr:
    st.markdown("#### 📈 Precision–Recall Curve (Hate = label 1)")

    pr_df = load_pr_table(model_dir)

    # baca threshold tersimpan (jika ada)
    thr_files = {
        "F1-max": os.path.join(model_dir, "best_threshold_f1.txt"),
        "Precision-first": os.path.join(model_dir, "best_threshold_prec.txt"),
        "Recall-first": os.path.join(model_dir, "best_threshold_recall.txt"),
    }
    thr_values = {}
    for k, p in thr_files.items():
        v = safe_read_txt(p)
        if v:
            try: thr_values[k] = float(v)
            except: pass

    active_thr = locals().get("thr_live", thr)  # pakai thr live kalau ada

    # Row PR: Chart besar kiri, ringkasan kanan
    left, right = st.columns([3, 1], vertical_alignment="top")
    with left:
        if pr_df is not None:
            try:
                import altair as alt
                pr_df_plot = pr_df.copy()
                for col in ["threshold", "recall_hate", "precision_hate"]:
                    pr_df_plot[col] = pr_df_plot[col].astype(float)

                idx_near = (pr_df_plot["threshold"] - active_thr).abs().idxmin()
                r_now = float(pr_df_plot.loc[idx_near, "recall_hate"])
                p_now = float(pr_df_plot.loc[idx_near, "precision_hate"])

                base = alt.Chart(pr_df_plot).mark_line().encode(
                    x=alt.X("recall_hate:Q", title="Recall (Hate)", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("precision_hate:Q", title="Precision (Hate)", scale=alt.Scale(domain=[0, 1])),
                    tooltip=[
                        alt.Tooltip("threshold:Q", format=".3f"),
                        alt.Tooltip("recall_hate:Q", format=".3f"),
                        alt.Tooltip("precision_hate:Q", format=".3f"),
                        alt.Tooltip("f1_hate:Q", format=".3f"),
                    ],
                ).properties(height=320)

                point_now = alt.Chart(pd.DataFrame({
                    "recall_hate": [r_now],
                    "precision_hate": [p_now],
                    "label": [f"now @ {active_thr:.2f}"],
                })).mark_point(filled=True, size=140).encode(
                    x="recall_hate:Q", y="precision_hate:Q",
                    color=alt.value("#60a5fa"),
                    tooltip=["label:N", alt.Tooltip("recall_hate:Q", format=".3f"), alt.Tooltip("precision_hate:Q", format=".3f")],
                )

                vline = alt.Chart(pd.DataFrame({"r": [r_now]})).mark_rule(strokeDash=[4, 4]).encode(x="r:Q")

                chart = (base + point_now + vline).interactive()
                st.altair_chart(chart, use_container_width=True)

            except Exception:
                fig = make_pr_plot(pr_df)
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("`pr_curve_table_hate.csv` tidak ditemukan.")

        # Tabel kecil di bawah chart
        if pr_df is not None:
            with st.expander("🔎 Sample PR table (20 baris)"):
                st.dataframe(pr_df.head(20), use_container_width=True, height=260)

    with right:
        st.markdown("**Ringkasan @ threshold aktif**")
        if pr_df is not None:
            # hitung lagi p_now/r_now jika altair gagal
            try:
                _idx = (pr_df["threshold"].astype(float) - active_thr).abs().idxmin()
                p_now = float(pr_df.loc[_idx, "precision_hate"])
                r_now = float(pr_df.loc[_idx, "recall_hate"])
            except Exception:
                p_now = r_now = float("nan")

            c1, c2 = st.columns(2)
            with c1: st.metric("Precision", f"{p_now:.3f}" if not np.isnan(p_now) else "—")
            with c2: st.metric("Recall", f"{r_now:.3f}" if not np.isnan(r_now) else "—")
            st.metric("F1 max (tabel)", f"{float(pr_df['f1_hate'].max()):.3f}")

            st.download_button(
                "⬇️ Download PR Table (CSV)",
                data=pr_df.to_csv(index=False).encode("utf-8"),
                file_name="pr_curve_table_hate.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.metric("Precision", "—"); st.metric("Recall", "—"); st.metric("F1 max (tabel)", "—")
            st.download_button(
                "⬇️ Download PR Table (CSV)",
                data="".encode(),
                file_name="pr_curve_table_hate.csv",
                mime="text/csv",
                disabled=True,
                use_container_width=True
            )

        if thr_values:
            st.markdown("<div class='small'>Threshold tersimpan:</div>", unsafe_allow_html=True)
            for k, v in thr_values.items():
                st.markdown(f"<span class='badge'>{k}: {v:.3f}</span>", unsafe_allow_html=True)

# ------------- TAB: ARTEFAK -------------
with tab_art:
    st.markdown("#### 📦 Artefak Model")
    ctop1, ctop2 = st.columns([1, 1])
    with ctop1:
        show_missing_only = st.checkbox("Tampilkan hanya yang hilang", value=False)
    with ctop2:
        st.caption("Lokasi folder:"); st.code(model_dir, language="text")

    files = []
    for name in [
        "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.txt", "metadata.json", "best_threshold_f1.txt", "best_threshold_prec.txt",
        "best_threshold_recall.txt", "pr_curve_table_hate.csv"
    ]:
        path = os.path.join(model_dir, name)
        info = {"file": name, "exists": os.path.exists(path), "path": path}
        if info["exists"]:
            try: info["size_kb"] = round(os.path.getsize(path)/1024, 1)
            except: info["size_kb"] = "-"
        else:
            info["size_kb"] = "-"
        files.append(info)

    art_df = pd.DataFrame(files)
    if show_missing_only:
        art_df = art_df[~art_df["exists"]].reset_index(drop=True)

    st.dataframe(art_df[["file", "exists", "size_kb", "path"]],
                 use_container_width=True, height=280)

# ------------- TAB: HISTORY (SQLite) -------------
with tab_hist:
    st.markdown("#### 🗂️ Riwayat Prediksi (SQLite, persistent)")

    # Control bar (live, responsif)
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        h_keyword = st.text_input("Cari keyword", key="hist_keyword", value=st.session_state.get("hist_keyword", ""))
    with c2:
        h_label = st.selectbox("Filter label", ["ALL", lbl(1), lbl(0)], index=0, key="hist_label")
    with c3:
        # toggle waktu live agar komponen di bawahnya aktif tanpa submit
        use_time = st.checkbox("Aktifkan filter waktu", value=st.session_state.get("hist_use_time", False), key="hist_use_time")
    with c4:
        clear_chk = st.checkbox("Saya yakin hapus semua", value=st.session_state.get("hist_clear_chk", False), key="hist_clear_chk")

    # Filter lain dalam form (submit untuk apply & export)
    with st.form(key="history_filters"):
        r1, r2, r3 = st.columns([1.2, 1, 1])
        with r1:
            h_prob = st.slider("Prob(Hate) range", 0.0, 1.0, st.session_state.get("hist_prob", (0.0, 1.0)), 0.01, key="hist_prob")
        with r2:
            h_limit = st.number_input("Max rows", 50, 5000, st.session_state.get("hist_limit", 500), 50, key="hist_limit")
        with r3:
            from datetime import datetime, time as dtime
            def datetime_picker(label: str):
                if hasattr(st, "datetime_input"):
                    return st.datetime_input(label, value=st.session_state.get(label), key=f"dt_{label}")
                c1_, c2_ = st.columns([2, 1])
                with c1_:
                    d = st.date_input(label + " (tanggal)", value=st.session_state.get(f"d_{label}"))
                with c2_:
                    t = st.time_input(label + " (jam)", value=st.session_state.get(f"t_{label}", dtime(0, 0)))
                return datetime.combine(d, t) if d else None

            since_dt = datetime_picker("Dari waktu") if use_time else None
            until_dt = datetime_picker("Sampai waktu") if use_time else None
            if not use_time:
                st.caption("Filter waktu dimatikan.")

        b1, b2, b3 = st.columns([1, 1, 2])
        with b1:
            submitted = st.form_submit_button("🔎 Terapkan Filter", use_container_width=True)
        with b2:
            export_btn = st.form_submit_button("⬇️ Export CSV", use_container_width=True)
        with b3:
            # tombol refresh di kanan
            refresh_btn = st.form_submit_button("🔄 Refresh", use_container_width=True)

    # Tombol Clear ALL di luar form, agar langsung merespon checkbox
    col_c1, col_c2, col_c3 = st.columns([6, 2, 2])
    with col_c3:
        clear_btn = st.button("🗑️ Clear ALL", use_container_width=True, disabled=not clear_chk)

    if clear_btn and clear_chk:
        clear_history()
        st.success("History dihapus.")

    if refresh_btn:
        submitted = True  # pakai flag yang sama biar rerun

    # Hitung timestamp jika filter waktu aktif
    since_ts = until_ts = None
    try:
        if use_time and since_dt:
            since_ts = since_dt.timestamp()
        if use_time and until_dt:
            until_ts = until_dt.timestamp()
    except Exception:
        pass

    # Query DB
    hist_df = query_history(
        keyword=h_keyword,
        label=h_label if h_label != "ALL" else "ALL",
        prob_min=h_prob[0], prob_max=h_prob[1],
        since=since_ts, until=until_ts, limit=int(h_limit)
    )

    # Ringkasan + tabel
    if not hist_df.empty:
        from datetime import datetime
        hist_view = hist_df.copy()
        hist_view["time"] = hist_view["ts"].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))

        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("Total record", f"{len(hist_view):,}")
        with k2: st.metric("% Hate", f"{(hist_view['prob_hate']>=0.5).mean()*100:.1f}%")
        with k3: st.metric("Mean prob(Hate)", f"{hist_view['prob_hate'].mean():.3f}")
        with k4: st.metric("Median prob(Hate)", f"{hist_view['prob_hate'].median():.3f}")

        cols_order = ["time", "full_text", "prob_hate", "prob_nonhate", "pred_argmax", "pred_threshold", "threshold"]
        st.dataframe(hist_view[cols_order], use_container_width=True, height=360)
    else:
        st.caption("Belum ada record di database atau tidak ada yang cocok filter.")

    # Export CSV
    if export_btn and not hist_df.empty:
        csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download History (filtered).csv",
            data=csv_bytes,
            file_name="history_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )
