"""Release gate for the PUBLIC demo repository (runs in CI on every push).

    python check_public.py [ROOT]

Fails if the published tree contains model weights, train-derived artifacts, runtime binaries, test-split samples,
the private repository URL, external script URLs, banned overstatements, or files whose SHA-256 does not match
assets/MANIFEST.json.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
RESTRICTED = re.compile(r"(\.onnx$|\.pt$|\.pth$|\.wasm$|train_feats|train_labels|probe\.json$|^vendor/|coi-serviceworker|knn-worker\.js$|reference\.json$)")
SKIP = re.compile(r"^(\.git|\.github)/|^check_public\.py$")
BANNED = [r"exactly as in the paper", r"pre-registered", r"rules out collapse", r"state[- ]of[- ]the[- ]art", r"beats VICReg", r"outperforms VICReg"]
PRIVATE_REPO = re.compile(r"github\.com/AliAlfatemi/psbl-jepa(?!-demo)")
bad = []
files = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and not SKIP.search(p.relative_to(ROOT).as_posix()))
bad += [f"restricted file: {f}" for f in files if RESTRICTED.search(f)]
man = json.loads((ROOT / "assets/MANIFEST.json").read_text())
if man.get("release_mode") != "public-static" or man.get("ships_model_weights") or man.get("ships_train_derived_artifacts"):
    bad.append("manifest is not a public-static release")
for rel, info in man["files"].items():
    if rel != "assets/MANIFEST.json" and (not (ROOT / rel).exists() or hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != info["sha256"]):
        bad.append(f"hash mismatch: {rel}")
unlisted = [f for f in files if f not in man["files"] and f not in ("assets/MANIFEST.json", "README_REPO.md", ".nojekyll")]
bad += [f"file not in manifest: {f}" for f in unlisted]
clips = json.loads((ROOT / "assets/clips/clips.json").read_text())
if any(c["val_index"] >= 1776 for c in clips["clips"]):
    bad.append("clip outside validation split")
for f in files:
    if f.endswith((".html", ".js", ".md", ".json")):
        t = (ROOT / f).read_text(errors="ignore")
        bad += [f"{f}: banned /{b}/" for b in BANNED if re.search(b, t, re.I)]
        if PRIVATE_REPO.search(t):
            bad.append(f"{f}: private repository URL")
        if f.endswith(".js") and re.search(r"""(?:fetch|src\s*=|import\s*\(|new Worker)\s*\(?\s*["'`](https?:)?//""", t):
            bad.append(f"{f}: external script URL")
if bad:
    print("PUBLIC RELEASE GATE: FAIL\n  " + "\n  ".join(bad))
    sys.exit(1)
print(f"PUBLIC RELEASE GATE: PASS ({len(files)} files)")
