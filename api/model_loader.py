from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

import emoji
import numpy as np
import onnxruntime as ort
import structlog
from transformers import AutoTokenizer

from common.config import settings

logger = structlog.get_logger(__name__)


def resolve_model_paths(model_path: str) -> Tuple[Path, Path]:
    candidate = Path(model_path)
    if candidate.is_dir():
        model_dir = candidate
        onnx_path = candidate / "model.onnx"
    else:
        model_dir = candidate.parent if candidate.parent.exists() else Path(".")
        onnx_path = candidate
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
    return model_dir, onnx_path


def load_tokenizer(model_dir: Path, tokenizer_hint: str | None) -> AutoTokenizer:
    if tokenizer_hint:
        hint_path = Path(tokenizer_hint)
        load_from = hint_path.parent if hint_path.is_file() else hint_path
        if load_from.exists():
            return AutoTokenizer.from_pretrained(load_from, use_fast=True)
    return AutoTokenizer.from_pretrained(model_dir, use_fast=True)


def build_session(onnx_path: Path) -> ort.InferenceSession:
    requested_cuda = (os.getenv("USE_CUDA", "0").lower() in {"1", "true", "yes"})
    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if requested_cuda and "CUDAExecutionProvider" in available:
        providers.insert(0, "CUDAExecutionProvider")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = int(os.getenv("INTRA_OP_THREADS", "1"))
    so.inter_op_num_threads = int(os.getenv("INTER_OP_THREADS", "1"))
    so.enable_mem_pattern = False
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # reduce contention
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    session = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)
    logger.info("onnx_session_ready", path=str(onnx_path), providers=session.get_providers())
    return session


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"http\S+|www\S+", "HTTPURL", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_text(
    text: str,
    tokenizer: AutoTokenizer,
    session: ort.InferenceSession,
    max_length: int = 128,
) -> Tuple[np.ndarray, int]:
    processed = preprocess_text(text)
    encoded = tokenizer(
        processed,
        return_tensors="np",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    input_names = {i.name for i in session.get_inputs()}
    feed = {}
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        if key in encoded and key in input_names:
            feed[key] = encoded[key].astype("int64")

    logits = session.run(None, feed)[0]
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = (np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True))[0]
    pred = int(np.argmax(probs))
    return probs, pred


class HateSpeechInference:
    def __init__(self, model_path: str, tokenizer_hint: str | None = None):
        model_dir, onnx_path = resolve_model_paths(model_path)
        self.tokenizer = load_tokenizer(model_dir, tokenizer_hint)
        self.session = build_session(onnx_path)
        self.device = "CUDA" if "CUDAExecutionProvider" in self.session.get_providers() else "CPU"
        self._warmup()

    def _warmup(self) -> None:
        try:
            encoded = self.tokenizer(
                "warmup",
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=16,
            )
            input_names = {i.name for i in self.session.get_inputs()}
            feed = {}
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                if key in encoded and key in input_names:
                    feed[key] = encoded[key].astype("int64")
            if feed:
                self.session.run(None, feed)
        except Exception as exc:  # pragma: no cover
            logger.warning("warmup_failed", error=str(exc))

    def predict(self, text: str) -> Tuple[float, float, int]:
        if not text.strip():
            return 0.01, 0.99, 0
        probs, pred = predict_text(text, self.tokenizer, self.session)
        prob_nonhate = float(probs[0])
        prob_hate = float(probs[1])
        return prob_hate, prob_nonhate, pred


classifier = HateSpeechInference(settings.model_path, settings.tokenizer_path)
