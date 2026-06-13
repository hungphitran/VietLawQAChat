"""Set optimal batch sizes for all configs based on target VRAM.

Two layers of safety:
  1. VRAM formula — guarantees no OOM (model + CUDA overhead + activation memory)
  2. Practical cap — prevents padding waste on long sequences (empirically derived)

Usage:
    python scripts/update_batch_size.py --vram 16           # specify VRAM in GB
    python scripts/update_batch_size.py --vram auto          # auto-detect GPU VRAM
    python scripts/update_batch_size.py --vram 16 --dry-run  # preview only

Applies safe batch sizes to:
    - configs/dense-selection/*.yaml  → encode batch_size (encode_kwargs)
    - configs/model-selection/*.yaml  → encode batch_size (encode_kwargs)
    - configs/pipeline/*.yaml         → encode + rerank batch_size
    - configs/train/bi_encoder_*.yaml    → train_bi batch_size (per model)
    - configs/train/cross_encoder_*.yaml  → train_cross batch_size (per model)
"""

import argparse
import math
import re
import sys
from pathlib import Path

# ── Model registry (from real measurements) ──────────────────────────────
# peak_per_layer_mb: max(attn, ffn, hidden) memory per sample per forward pass
# all_act_mb: all layers' activations for encoding (forward-only)
# train_act_mb: activation memory per sample for training (forward + backward stored)
#   Derived from real training observations:
#   - e5: bs=128 on A100 40GB → 155.0
#   - vietnamese: bs=256 on A100 40GB → 81.2
#   - granite: no data yet, conservative = all_act_mb → 198.2
#   - gemma: bs=64 on H100 85GB → 640.0
# max_length: default sequence length for the model
# encode_cap: empirical batch size cap for encode (padding waste on long seq)
#   - derived from benchmarks on Vietnamese legal text (avg 1174 chars)
#   - rule: max_len 512 → cap 128, max_len 256 → cap 256, slow model → lower

MODELS = {
    "intfloat/multilingual-e5-base": {
        "model_gb": 1.11,
        "peak_per_layer_mb": 20.4,
        "train_overhead_gb": 3.34,
        "all_act_mb": 245.4,
        "train_act_mb": 155.0,
        "max_length": 512,
        "encode_cap": 128,
    },
    "bkai-foundation-models/vietnamese-bi-encoder": {
        "model_gb": 0.54,
        "peak_per_layer_mb": 7.1,
        "train_overhead_gb": 1.62,
        "all_act_mb": 84.9,
        "train_act_mb": 81.2,
        "max_length": 256,
        "encode_cap": 256,
    },
    "ibm-granite/granite-embedding-97m-multilingual-r2": {
        "model_gb": 0.19,
        "peak_per_layer_mb": 16.5,
        "train_overhead_gb": 0.58,
        "all_act_mb": 198.2,
        "train_act_mb": 198.2,
        "max_length": 512,
        "encode_cap": 128,
    },
    "google/embeddinggemma-300m": {
        "model_gb": 1.23,
        "peak_per_layer_mb": 7.1,
        "train_overhead_gb": 3.69,
        "all_act_mb": 169.9,
        "train_act_mb": 640,
        "max_length": 512,
        "encode_cap": 64,
    },
    "itdainb/PhoRanker": {
        "model_gb": 0.54,
        "peak_per_layer_mb": 7.1,
        "train_overhead_gb": 1.62,
        "all_act_mb": 84.9,
        "train_act_mb": 81.2,
        "max_length": 256,
        "encode_cap": 256,
    },
    "Alibaba-NLP/gte-multilingual-reranker-base": {
        "model_gb": 1.19,
        "peak_per_layer_mb": 15.0,
        "train_overhead_gb": 3.57,
        "all_act_mb": 180.0,
        "train_act_mb": 180.0,
        "max_length": 512,
        "encode_cap": 128,
    },
}

CUDA_OVERHEAD_GB = 1.5
FRAGMENTATION = 1.15  # 15% reserve for allocator fragmentation
HEADROOM = 0.95       # use max 95% of available VRAM
MAX_BS = 512


def _safe_bs(raw: float, cap: int) -> int:
    """Round down to power of 2, min 2, max cap."""
    if raw < 2:
        return 2
    return min(2 ** int(math.log2(raw)), cap)


def compute_batch_sizes(vram_gb: int) -> dict[str, dict[str, int]]:
    """Compute safe batch sizes for all known models given VRAM.

    Two caps applied: min(formula_limit, practical_cap)
    - formula_limit: VRAM budget / memory per sample → guarantees no OOM
    - practical_cap: empirically derived from benchmarks → prevents padding waste
    """
    results = {}
    for name, info in MODELS.items():
        cap = info["encode_cap"]

        # Encode: model + cuda + peak_per_layer * batch
        avail_encode = (vram_gb - info["model_gb"] - CUDA_OVERHEAD_GB) / FRAGMENTATION * HEADROOM
        encode_bs = _safe_bs(avail_encode * 1024 / info["peak_per_layer_mb"], min(cap, MAX_BS))

        # Rerank: pairs = 2x activation per sample, cap at half of encode cap
        rerank_bs = _safe_bs(avail_encode * 1024 / (info["peak_per_layer_mb"] * 2), min(cap // 2, MAX_BS))

        # Train bi-encoder: grad + optimizer + all_layers * activations * batch
        train_act = info.get("train_act_mb", info["all_act_mb"])
        avail_train = (vram_gb - info["train_overhead_gb"] - CUDA_OVERHEAD_GB) / FRAGMENTATION * HEADROOM
        train_bi_bs = _safe_bs(avail_train * 1024 / train_act, MAX_BS)

        # Train cross-encoder: pairs = 2x activation
        train_cross_bs = _safe_bs(avail_train * 1024 / (train_act * 2), MAX_BS)

        results[name] = {
            "encode": encode_bs,
            "rerank": rerank_bs,
            "train_bi": train_bi_bs,
            "train_cross": train_cross_bs,
        }
    return results


def detect_vram() -> int:
    """Auto-detect GPU VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return int(props.total_memory / 1e9)
    except ImportError:
        pass
    print("Cannot auto-detect VRAM. Use --vram <GB>")
    sys.exit(1)


# Aliases: uploaded model names that are functionally identical to base models
MODEL_ALIASES = {
    "phatvucoder/vietnamese-bi-encoder": "bkai-foundation-models/vietnamese-bi-encoder",
    "phatvucoder/vietnamese-bi-encoder-mp": "bkai-foundation-models/vietnamese-bi-encoder",
}


def find_model(config_text: str) -> str | None:
    """Find model_name in YAML text."""
    m = re.search(r"model_name:\s*[\"']?(\S+)[\"']?", config_text)
    if m:
        return MODEL_ALIASES.get(m.group(1), m.group(1))
    m = re.search(r"model_name_or_path:\s*[\"']?(\S+)[\"']?", config_text)
    if m:
        return MODEL_ALIASES.get(m.group(1), m.group(1))
    return None


def set_yaml_batch_size(path: Path, batch_size: int, pattern: str = r"batch_size: \d+") -> bool:
    """Replace batch_size in YAML file. Returns True if changed."""
    text = path.read_text()
    new_text = re.sub(pattern, f"batch_size: {batch_size}", text)
    if new_text != text:
        path.write_text(new_text)
        return True
    return False


def process_configs(configs_dir: Path, batch_sizes: dict, dry_run: bool) -> list[str]:
    """Process all YAML configs in a directory."""
    changes = []
    for cfg in sorted(configs_dir.glob("*.yaml")):
        text = cfg.read_text()
        model = find_model(text)
        if not model:
            continue
        if model not in batch_sizes:
            changes.append(f"  ⚠ {cfg.name}: unknown model '{model}'")
            continue

        bs = batch_sizes[model]

        # Detect operation type from config structure
        if "per_device_train_batch_size" in text or "num_epochs" in text:
            op = "train_bi" if "bi_encoder" in cfg.name or "bi" in cfg.name else "train_cross"
            changed = False if dry_run else set_yaml_batch_size(cfg, bs[op])
            status = "would update" if dry_run else ("updated" if changed else "already set")
            changes.append(f"  {cfg.name}: {op} → bs={bs[op]} ({status})")
        else:
            changed = False if dry_run else set_yaml_batch_size(cfg, bs["encode"])
            status = "would update" if dry_run else ("updated" if changed else "already set")
            changes.append(f"  {cfg.name}: encode → bs={bs['encode']} ({status})")

    return changes


def process_train_configs(batch_sizes: dict, dry_run: bool) -> list[str]:
    """Process train configs — model comes from model_name field."""
    changes = []
    train_dir = Path("configs/train")

    for cfg in sorted(train_dir.glob("bi_encoder_*.yaml")):
        text = cfg.read_text()
        model = find_model(text)
        if not model or model not in batch_sizes:
            changes.append(f"  ⚠ {cfg.name}: unknown model '{model}'")
            continue
        bs = batch_sizes[model]["train_bi"]
        changed = False if dry_run else set_yaml_batch_size(cfg, bs)
        status = "would update" if dry_run else ("updated" if changed else "already set")
        changes.append(f"  {cfg.name}: train_bi → bs={bs} (from {model.split('/')[-1]}, {status})")

    for cfg in sorted(train_dir.glob("cross_encoder_*.yaml")):
        text = cfg.read_text()
        model = find_model(text)
        if not model or model not in batch_sizes:
            changes.append(f"  ⚠ {cfg.name}: unknown model '{model}'")
            continue
        bs = batch_sizes[model]["train_cross"]
        changed = False if dry_run else set_yaml_batch_size(cfg, bs)
        status = "would update" if dry_run else ("updated" if changed else "already set")
        changes.append(f"  {cfg.name}: train_cross → bs={bs} (from {model.split('/')[-1]}, {status})")

    return changes


def main():
    parser = argparse.ArgumentParser(description="Set optimal batch sizes based on VRAM")
    parser.add_argument("--vram", type=str, required=True, help="VRAM in GB, or 'auto'")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    if args.vram == "auto":
        vram = detect_vram()
    else:
        vram = int(args.vram)

    batch_sizes = compute_batch_sizes(vram)
    print(f"Target: {vram}GB VRAM")
    print()
    print(f"  {'Model':48s} | {'enc':>4s} | {'rerk':>4s} | {'trBi':>4s} | {'trX':>4s} | cap")
    print(f"  {'-' * 48} | {'-' * 4} | {'-' * 4} | {'-' * 4} | {'-' * 4} | {'-' * 3}")
    for model, bs in batch_sizes.items():
        short = model.split("/")[-1]
        cap = MODELS[model]["encode_cap"]
        print(f"  {short:48s} | {bs['encode']:>4d} | {bs['rerank']:>4d} | {bs['train_bi']:>4d} | {bs['train_cross']:>4d} | {cap}")
    print()

    # Process configs
    config_dirs = [
        Path("configs/dense-selection"),
        Path("configs/model-selection"),
        Path("configs/pipeline"),
    ]

    all_changes = []
    for d in config_dirs:
        if d.exists():
            changes = process_configs(d, batch_sizes, args.dry_run)
            if changes:
                all_changes.append(f"\n{d}/:")
                all_changes.extend(changes)

    # Train configs
    train_changes = process_train_configs(batch_sizes, args.dry_run)
    if train_changes:
        all_changes.append("\nconfigs/train/:")
        all_changes.extend(train_changes)

    if all_changes:
        print("\n".join(all_changes))
    else:
        print("No configs found to update.")

    if args.dry_run:
        print("\n(dry run — no files changed)")


if __name__ == "__main__":
    main()
