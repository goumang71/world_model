"""Fetch the pinned, public CarRacing-v3 model into an isolated directory."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "6b4b3a15fab37b73d2b8fdd3e231f670e5d65bd3"
BASE = f"https://huggingface.co/flydexo/world-models-carracing-v3/resolve/{REVISION}/"
OUT = ROOT / "external_models" / "flydexo_world_models_carracing_v3"
FILES = ("README.md", "config.yaml", "model.py", "vae.pt", "rnn.pt", "controller.pt")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in FILES:
        path = OUT / name
        if not path.exists():
            request = urllib.request.Request(BASE + name, headers={"User-Agent": "world-models-baseline-research/1.0"})
            with urllib.request.urlopen(request, timeout=120) as source, path.open("wb") as target:
                while chunk := source.read(1 << 20):
                    target.write(chunk)
        blob = path.read_bytes()
        records[name] = dict(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(), source=BASE+name)
        print(name, records[name]["bytes"], records[name]["sha256"], flush=True)
    (OUT / "provenance.json").write_text(json.dumps(dict(revision=REVISION, files=records), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
