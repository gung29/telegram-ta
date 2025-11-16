from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple

import emoji
import numpy as np
import structlog
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.config import settings

logger = structlog.get_logger(__name__)


def resolve_model_dir(model_path: str) -> Path:
    candidate = Path(model_path)
    if candidate.is_dir():
        return candidate
    if candidate.is_file():
        return candidate.parent
    raise FileNotFoundError(f"Model path {model_path} not found")


def load_tokenizer(model_dir: Path, tokenizer_hint: str | None) -> AutoTokenizer:
    if tokenizer_hint:
        hint_path = Path(tokenizer_hint)
        load_from = hint_path.parent if hint_path.is_file() else hint_path
        if load_from.exists():
            return AutoTokenizer.from_pretrained(load_from, use_fast=True)
    return AutoTokenizer.from_pretrained(model_dir, use_fast=True)


def load_model(model_dir: Path, device: torch.device) -> AutoModelForSequenceClassification:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    logger.info("torch_model_ready", path=str(model_dir), device=str(device))
    return model


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
    model: AutoModelForSequenceClassification,
    device: torch.device,
    max_length: int = 128,
) -> Tuple[np.ndarray, int]:
    processed = preprocess_text(text)
    encoded = tokenizer(
        processed,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    return probs, pred


class HateSpeechInference:
    def __init__(self, model_path: str, tokenizer_hint: str | None = None):
        model_dir = resolve_model_dir(model_path)
        use_cuda = os.getenv("USE_CUDA", "0").lower() in {"1", "true", "yes"}
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.tokenizer = load_tokenizer(model_dir, tokenizer_hint)
        self.model = load_model(model_dir, self.device)
        self._warmup()

    def _warmup(self) -> None:
        try:
            encoded = self.tokenizer(
                "warmup",
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=16,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                self.model(**encoded)
        except Exception as exc:  # pragma: no cover
            logger.warning("warmup_failed", error=str(exc))

    def predict(self, text: str) -> Tuple[float, float, int]:
        if not text.strip():
            return 0.01, 0.99, 0
        probs, pred = predict_text(text, self.tokenizer, self.model, self.device)
        prob_nonhate = float(probs[0])
        prob_hate = float(probs[1])
        return prob_hate, prob_nonhate, pred


classifier = HateSpeechInference(settings.model_path, settings.tokenizer_path)
