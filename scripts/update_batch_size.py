"""Set optimal batch sizes for all configs based on target VRAM.

Thin CLI over `vnlegal_rag_v2.utils.batch_size` (single source of truth).
Two layers of safety:
  1. VRAM formula — guarantees no OOM (model + CUDA overhead + activation memory)
  2. Practical cap — prevents padding waste on long sequences (empirically derived)

Usage:
    python scripts/update_batch_size.py --vram 16           # specify VRAM in GB
    python scripts/update_batch_size.py --vram auto          # auto-detect GPU VRAM
    python scripts/update_batch_size.py --vram 16 --dry-run  # preview only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from vnlegal_rag_v2.utils.batch_size import (
    MODELS,
    compute_batch_sizes,
    detect_vram,
    resolve_model,
)

# Config fields that hold the model name (checked in order)
MODEL_FIELD_PATTERNS = [
    r"model_name:\s*[\"']?(\S+)[\"']?",
    r"model_name_or_path:\s*[\"']?(\S+)[\"']?",
    r"model_path:\s*[\"']?(\S+)[\"']?",
]


def find_model(config_text: str) -> str | None:
    """Find model name in YAML text (checks model_name, model_name_or_path, model_path)."""
    for pattern in MODEL_FIELD_PATTERNS:
        m = re.search(pattern, config_text)
        if m:
            return resolve_model(m.group(1))
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

    vram = detect_vram() if args.vram == "auto" else int(args.vram)
    if vram is None:
        sys.exit(1)

    batch_sizes = compute_batch_sizes(vram)
    print(f"Target: {vram}GB VRAM\n")
    print(f"  {'Model':48s} | {'enc':>4s} | {'rerk':>4s} | {'trBi':>4s} | {'trX':>4s} | cap")
    print(f"  {'-' * 48} | {'-' * 4} | {'-' * 4} | {'-' * 4} | {'-' * 4} | {'-' * 3}")
    for model, bs in batch_sizes.items():
        short = model.split("/")[-1]
        cap = MODELS[model]["encode_cap"]
        print(f"  {short:48s} | {bs['encode']:>4d} | {bs['rerank']:>4d} | {bs['train_bi']:>4d} | {bs['train_cross']:>4d} | {cap}")
    print()

    config_dirs = [
        Path("configs/dense-selection"),
        Path("configs/model-selection"),
        Path("configs/pipeline"),
        Path("configs/data"),
    ]
    all_changes = []
    for d in config_dirs:
        if d.exists():
            changes = process_configs(d, batch_sizes, args.dry_run)
            if changes:
                all_changes.append(f"\n{d}/:")
                all_changes.extend(changes)

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
