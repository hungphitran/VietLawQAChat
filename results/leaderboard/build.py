#!/usr/bin/env python3
"""Build data.js and serve leaderboard. Stars save to favorites.txt."""

import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent
DIR = Path(__file__).resolve().parent
OUT = DIR / "data.js"
FAVS = DIR / "favorites.txt"


def parse_name(name: str) -> dict:
    parts = name.split("_")
    prefix = parts[0]
    params = {}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            params[k] = v

    result = {"params": params}
    seg_map = {"non-seg": "none", "pyvi": "pyvi", "underthesea": "underthesea"}

    if "-" in prefix:
        method, rest = prefix.split("-", 1)
        result["method"] = method
        if rest in seg_map:
            result["seg"] = seg_map[rest]
        else:
            result["model"] = rest
    else:
        result["method"] = prefix

    return result


def build():
    groups, metrics, experiments = {}, set(), []

    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir() or d.name == "leaderboard":
            continue
        gid = d.name
        exps = []
        for f in sorted(d.glob("*.json")):
            data = json.loads(f.read_text())
            for name, scores in data.items():
                if "stage" in scores:  # nested format (hybrid-tuning): stage + params + metrics
                    info = {"method": scores["stage"],
                            "params": {k: str(v) for k, v in scores.get("params", {}).items()}}
                    entry_metrics = {k: v for k, v in scores.items() if k not in ("stage", "params")}
                else:  # legacy flat format
                    info = parse_name(name)
                    entry_metrics = scores
                exps.append({"name": name, "group": gid, **info, "metrics": entry_metrics})
                metrics.update(entry_metrics.keys())
        if exps:
            groups[gid] = {"label": gid, "count": len(exps)}
            experiments.extend(exps)

    favorites = []
    if FAVS.exists():
        favorites = [l.strip() for l in FAVS.read_text().splitlines() if l.strip()]

    payload = {"groups": groups, "metrics": sorted(metrics), "experiments": experiments, "favorites": favorites}
    OUT.write_text(f"const DATA = {json.dumps(payload)};\n")
    print(f"✓ {len(experiments)} experiments · {len(groups)} groups · {len(favorites)} favorites")
    return len(experiments) > 0


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DIR), **kw)

    def do_GET(self):
        if self.path == "/api/favorites":
            names = FAVS.read_text().strip().splitlines() if FAVS.exists() else []
            self._json([n for n in names if n.strip()])
        else:
            super().do_GET()
        # No cache for data.js so leaderboard always fresh
        if self.path.startswith("/data.js"):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")

    def do_POST(self):
        if self.path == "/api/favorites":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            names = json.loads(body)
            FAVS.write_text("\n".join(names) + "\n")
            self._json({"ok": True})
        else:
            self.send_error(404)

    def do_PUT(self):
        self.do_POST()

    def _json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def serve(port=8787):
    build()
    server = HTTPServer(("localhost", port), Handler)
    print(f"🏆 http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped")


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        serve()
