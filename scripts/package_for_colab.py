"""Сборка отдельного архива исходников для Colab.

Ноутбук загружает код через git clone; архив в него не встраивается.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "dist" / "colab_bundle.zip"
NOTEBOOK = ROOT / "BTZSC_Colab.ipynb"

INCLUDE_FILES = [
    "experiment.py",
    "requirements-colab.txt",
    "docs/colab.md",
    "contracts.json",
]
INCLUDE_GLOBS = [
    "btzsc_ru/*.py",
    "tests/*.py",
    "scripts/import_colab_outputs.py",
]
FORBIDDEN_PARTS = {".git", ".work", ".venv", "dist", "incoming", "runs", "data", "__pycache__", "models"}
FORBIDDEN_NAMES = {"results.json", "docs/results_legacy_unverified.json", "BTZSC_Colab.ipynb"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def collect(root: Path = ROOT) -> list[Path]:
    files: list[Path] = []
    for rel in INCLUDE_FILES:
        p = root / rel
        if p.exists():
            files.append(p)
    for pattern in INCLUDE_GLOBS:
        files.extend(sorted(root.glob(pattern)))
    # Ноутбук загружает код из репозитория и в архив не входит.
    clean: list[Path] = []
    for p in files:
        rel = p.relative_to(root)
        if set(rel.parts) & FORBIDDEN_PARTS or rel.name in FORBIDDEN_NAMES:
            continue
        clean.append(p)
    return sorted(set(clean))


def build(out: Path = DEFAULT_OUT, root: Path = ROOT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    files = collect(root)
    manifest = {
        "bundle": out.name,
        "files": [
            {"path": str(p.relative_to(root)), "sha256": sha256_of(p), "bytes": p.stat().st_size}
            for p in files
        ],
    }
    # Архив воспроизводим независимо от времени изменения файлов.
    def _add(zf: zipfile.ZipFile, arcname: str, payload: bytes) -> None:
        info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, payload)

    with zipfile.ZipFile(out, "w") as zf:
        for p in files:
            _add(zf, str(p.relative_to(root)), p.read_bytes())
        _add(zf, "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--list", action="store_true", help="напечатать содержимое архива")
    parser.add_argument("--no-embed", action="store_true", help="устаревший флаг; архив не встраивается")
    parser.add_argument("--check-embedded", action="store_true",
                        help="проверить отсутствие встроенного архива в ноутбуке")
    args = parser.parse_args()
    if args.check_embedded:
        notebook_text = NOTEBOOK.read_text(encoding="utf-8")
        if "BUNDLE_B64" in notebook_text or "BUNDLE_PAYLOAD" in notebook_text:
            print("в ноутбуке остался встроенный архив")
            return 1
        print("ноутбук не содержит встроенного архива")
        return 0

    out = build(Path(args.out))
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    print(f"{out} ({out.stat().st_size} bytes), файлов: {len(names)}")
    if args.list:
        for n in names:
            print("  " + n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
