"""Launch the Vietnamese Legal RAG chat app.

    python scripts/serve_app.py                      # default config (configs/app/app.yaml)
    APP_CONFIG=configs/app/_smoke_tiny.yaml python scripts/serve_app.py
    python scripts/serve_app.py --config configs/app/app.yaml --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse

from app.config import load_config


def main():
    ap = argparse.ArgumentParser(description="Vietnamese Legal RAG chat app")
    ap.add_argument("--config", default=None, help="Path to app config YAML (default: configs/app/app.yaml)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    # validate config loads early (clear error before uvicorn imports the app)
    cfg = load_config(args.config)
    print(f"[serve] config loaded: corpus={cfg['corpus_path']}, pipeline={cfg['pipeline_config']}")

    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
