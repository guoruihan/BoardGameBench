"""Build the dependency-free static UI (HTML/CSS/JS are intentionally inline)."""
import argparse
from pathlib import Path
import shutil

from boardbench.artifacts.store import digest, write_json
from boardbench.ui import server

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", default="build/ui")
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
assets = {}
for name in ("index.html", "board.html"):
    source = Path(server.__file__).with_name(name)
    shutil.copy2(source, out / name)
    assets[name] = digest(out / name)
write_json(out / "manifest.json", {"format": "static-html", "assets": assets})
print(out.resolve())
