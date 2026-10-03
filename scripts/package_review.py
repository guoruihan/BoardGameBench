"""Bundle explicit source and acceptance evidence; never include environment/cache/secrets."""
import argparse
import hashlib
from pathlib import Path
import zipfile

from boardbench.artifacts.store import read_json, source_id, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = Path(args.evidence).resolve()
    if read_json(evidence / "audit.json")["status"] != "passed":
        raise ValueError("acceptance evidence has not passed")
    if read_json(evidence / "environment.json")["source_id"] != source_id():
        raise ValueError("source changed since acceptance; regenerate evidence")
    target = Path(args.out).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    files = {}
    for entry in ("src/boardbench", "configs", "tests", "examples", "scripts", "docs",
                  "pyproject.toml", "README.md", "IMPLEMENTATION_STATUS.md", ".gitignore"):
        path = root / entry
        if not path.exists():
            raise FileNotFoundError(path)
        for p in path.rglob("*") if path.is_dir() else [path]:
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                files[p] = str(p.relative_to(root))
    for p in evidence.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            # Repeated evaluation is deliberately included as reproducibility evidence.
            files[p] = str(Path("runs") / evidence.name / p.relative_to(evidence))
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in sorted(files.items()):
            archive.write(path, str(Path("boardbench-v0") / relative))
    write_json(target.with_suffix(".sha256.json"), {"file": target.name, "bytes": target.stat().st_size,
               "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "files": len(files)})
    print(f"Review ZIP: {target} ({target.stat().st_size} bytes, {len(files)} files)")


if __name__ == "__main__":
    main()
