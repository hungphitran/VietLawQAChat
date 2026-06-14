"""VRAM-aware batch size computation for known models.

Single source of truth — `scripts/update_batch_size.py` (writes to configs)
and `scripts/tune_hybrid.py` (uses at runtime) both import from here.
"""
from __future__ import annotations

import math
import sys

# ── Model registry (from real measurements) ──────────────────────────────
# peak_per_layer_mb: max(attn, ffn, hidden) memory per sample per forward pass
# all_act_mb: all layers' activations for encoding (forward-only)
# train_act_mb: activation memory per sample for training (forward + backward stored)
# max_length: default sequence length for the model
# encode_cap: empirical batch size cap for encode (padding waste on long seq)
MODELS: dict[str, dict] = {
    "intfloat/multilingual-e5-base": {
        "model_gb": 1.11, "peak_per_layer_mb": 20.4, "train_overhead_gb": 3.34,
        "all_act_mb": 245.4, "train_act_mb": 155.0, "max_length": 512, "encode_cap": 128,
    },
    "bkai-foundation-models/vietnamese-bi-encoder": {
        "model_gb": 0.54, "peak_per_layer_mb": 7.1, "train_overhead_gb": 1.62,
        "all_act_mb": 84.9, "train_act_mb": 81.2, "max_length": 256, "encode_cap": 256,
    },
    "ibm-granite/granite-embedding-97m-multilingual-r2": {
        "model_gb": 0.19, "peak_per_layer_mb": 16.5, "train_overhead_gb": 0.58,
        "all_act_mb": 198.2, "train_act_mb": 198.2, "max_length": 512, "encode_cap": 128,
    },
    "google/embeddinggemma-300m": {
        "model_gb": 1.23, "peak_per_layer_mb": 7.1, "train_overhead_gb": 3.69,
        "all_act_mb": 169.9, "train_act_mb": 640, "max_length": 512, "encode_cap": 64,
    },
    "itdainb/PhoRanker": {
        "model_gb": 0.54, "peak_per_layer_mb": 7.1, "train_overhead_gb": 1.62,
        "all_act_mb": 84.9, "train_act_mb": 81.2, "max_length": 256, "encode_cap": 256,
    },
    "Alibaba-NLP/gte-multilingual-reranker-base": {
        "model_gb": 1.19, "peak_per_layer_mb": 15.0, "train_overhead_gb": 3.57,
        "all_act_mb": 180.0, "train_act_mb": 180.0, "max_length": 512, "encode_cap": 128,
    },
}

CUDA_OVERHEAD_GB = 1.5
FRAGMENTATION = 1.15  # 15% reserve for allocator fragmentation
HEADROOM = 0.95       # use max 95% of available VRAM
MAX_BS = 512

# Aliases: uploaded model names functionally identical to a base model
MODEL_ALIASES: dict[str, str] = {
    "phatvucoder/vietnamese-bi-encoder": "bkai-foundation-models/vietnamese-bi-encoder",
    "phatvucoder/vietnamese-bi-encoder-mp": "bkai-foundation-models/vietnamese-bi-encoder",
    "phatvucoder/PhoRanker-legal-vn": "itdainb/PhoRanker",
    "phatvucoder/PhoRanker-legal-vn-v2": "itdainb/PhoRanker",
}


def resolve_model(name: str) -> str:
    """Resolve an alias (uploaded model name) to its base model key."""
    return MODEL_ALIASES.get(name, name)


def detect_vram() -> int:
    """Auto-detect GPU VRAM in GB (CUDA only)."""
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory / 1e9)
    except ImportError:
        pass
    print("Cannot auto-detect VRAM. Use --vram <GB>")
    sys.exit(1)


def _safe_bs(raw: float, cap: int) -> int:
    """Round down to power of 2, min 2, max cap."""
    if raw < 2:
        return 2
    return min(2 ** int(math.log2(raw)), cap)


def compute_batch_sizes(vram_gb: int) -> dict[str, dict[str, int]]:
    """Compute safe batch sizes for all known models given VRAM.

    Two caps applied: min(formula_limit, practical_cap)
    - formula_limit: VRAM budget / memory per sample → guarantees no OOM
    - practical_cap: empirically derived → prevents padding waste on long seqs
    """
    results = {}
    for name, info in MODELS.items():
        cap = info["encode_cap"]

        avail_encode = (vram_gb - info["model_gb"] - CUDA_OVERHEAD_GB) / FRAGMENTATION * HEADROOM
        encode_bs = _safe_bs(avail_encode * 1024 / info["peak_per_layer_mb"], min(cap, MAX_BS))
        rerank_bs = _safe_bs(avail_encode * 1024 / (info["peak_per_layer_mb"] * 2), min(cap // 2, MAX_BS))

        train_act = info.get("train_act_mb", info["all_act_mb"])
        avail_train = (vram_gb - info["train_overhead_gb"] - CUDA_OVERHEAD_GB) / FRAGMENTATION * HEADROOM
        train_bi_bs = _safe_bs(avail_train * 1024 / train_act, MAX_BS)
        train_cross_bs = _safe_bs(avail_train * 1024 / (train_act * 2), MAX_BS)

        results[name] = {
            "encode": encode_bs,
            "rerank": rerank_bs,
            "train_bi": train_bi_bs,
            "train_cross": train_cross_bs,
        }
    return results
